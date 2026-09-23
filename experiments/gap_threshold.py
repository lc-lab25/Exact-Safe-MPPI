"""Fig. 1 and Table I: gap threading against the dimensionless group kappa.

Two geometries x six widths x (eight rho + the exact arm) x 50 seeds, with the narrowest
disk width excluded, giving 4400 soft-minimum runs and 550 exact runs.  Each run is
classified with the experiment code's own definitions; threading means crossing the slice
q_x = 0.
"""
import json

import numpy as np

from _common import (base_parser, binom_ci, load_config, out_dir, run_all, write_csv,
                     FIGURES)
from exact_safe_mppi import environments as E
from exact_safe_mppi.mppi import Cost, make_filter, run_closed_loop
from exact_safe_mppi.dynamics import DoubleIntegrator
from exact_safe_mppi.utils import run_seed

LN2 = float(np.log(2.0))


def make_env(geom, w):
    return E.RoundedSquareGap(w) if geom == "box" else E.DiskGap(w)


_SUP_CACHE = {}


def sup_hex_for(geom, w):
    """sup_X h_min is a property of the cell, not of the run; compute it once per cell."""
    key = (geom, round(float(w), 4))
    if key not in _SUP_CACHE:
        _SUP_CACHE[key] = float(E.sup_slice_hex(make_env(geom, w)))
    return _SUP_CACHE[key]


def cells(cfg, quick):
    ws = cfg["widths"][2:3] if quick else cfg["widths"]
    rhos = cfg["rhos"][:2] if quick else cfg["rhos"]
    nseed = 3 if quick else cfg["n_seeds"]
    excluded = set(cfg["excluded_cells"])
    out, idx = [], 0
    for geom in cfg["geometries"]:
        for w in ws:
            if f"{geom}_{w:.2f}" in excluded:
                continue
            arms = list(rhos) + ([None] if cfg["include_exact_arm"] else [])
            for rho in arms:
                for s in range(nseed):
                    out.append(dict(geom=geom, w=w, rho=rho, run_index=idx,
                                    seed=run_seed(cfg["base_seed"], idx),
                                    delta=cfg["delta_star"][f"{geom}_{w:.2f}"],
                                    lam=cfg["mppi"]["lambda_by_geometry"][geom],
                                    K=cfg["mppi"]["K"], N=cfg["mppi"]["horizon_N"],
                                    T=(cfg["integration"]["quick_run_length_T"] if quick
                                       else cfg["integration"]["run_length_T"])))
                    idx += 1
    return out


def one(task):
    """Classification identical to experiments/r5_gap_threading.py:run_one in the source code.

    Threading is not merely crossing x = 0: the obstacles are finite, so a detour reaches the
    goal too. A run counts as threaded only if it gets past x = 0.5 AND its closest approach to
    x = 0 happens within the gap, |q_y| <= w + 0.15.
    """
    env = make_env(task["geom"], task["w"])
    x0, qg = env.start_goal()
    kind = "exact" if task["rho"] is None else "softmin"
    filt = make_filter(kind, env, rho=task["rho"], gamma=env.GAMMA,
                       delta=task["delta"], k_max=4)
    try:
        run = run_closed_loop(x0, env, env, filt, Cost(qg), seed=task["seed"], T=task["T"],
                              Ts=env.TS, dt=env.DT_INNER, N=task["N"], K=task["K"],
                              lam=task["lam"], Sigma=env.SIGMA)
    except AssertionError as e:
        return dict(geom=task["geom"], w=task["w"],
                    rho=(task["rho"] if task["rho"] is not None else -1.0),
                    seed=task["seed"], run_index=task["run_index"], kappa=float("nan"),
                    sup_hex=float("nan"), threaded=0, deadlock=0, moving=0, violation=0,
                    ok=0, min_h=float("nan"), err=float("nan"), qy_at_gap=float("nan"),
                    reason=str(e)[:60])
    x = np.asarray(run["x"])
    q = x[:, :2]
    err = float(np.linalg.norm(q[-1] - qg))
    minh = float(np.asarray(run["hj"]).min())
    crossed = bool((q[:, 0] > 0.5).any())
    i = int(np.argmin(np.abs(q[:, 0])))
    qy_at_gap = float(abs(q[i, 1]))
    threaded = bool(crossed and qy_at_gap <= task["w"] + 0.15)
    ok = bool((err < 0.5) and (minh >= -1e-9) and threaded)
    speed = float(np.abs(x[-20:, 2:]).max()) if x.shape[0] >= 20 else float("inf")
    deadlock = bool((not threaded) and (minh >= -1e-9) and (speed < 0.3))
    violation = bool(minh < -1e-9)
    moving = bool((not threaded) and (not deadlock) and (not violation))
    sup_hex = sup_hex_for(task["geom"], task["w"])
    return dict(geom=task["geom"], w=task["w"],
                rho=(task["rho"] if task["rho"] is not None else -1.0),
                seed=task["seed"], run_index=task["run_index"],
                kappa=(task["rho"] * sup_hex if task["rho"] is not None else float("nan")),
                sup_hex=sup_hex, threaded=int(threaded), deadlock=int(deadlock),
                moving=int(moving), violation=int(violation), ok=int(ok),
                min_h=minh, err=err, qy_at_gap=qy_at_gap, reason="")


def table_one(rows):
    soft = [r for r in rows if r["rho"] > 0]
    bands = [("<ln2", lambda k: k < LN2),
             ("[ln2,1)", lambda k: LN2 <= k < 1.0),
             (">=1", lambda k: k >= 1.0)]
    out = []
    for name, sel in bands:
        g = [r for r in soft if sel(r["kappa"])]
        n = len(g)
        if n == 0:
            continue
        thr = sum(r["threaded"] for r in g)
        lo, hi = binom_ci(thr, n)
        out.append(dict(band=name, runs=n,
                        threaded=thr / n, deadlocked=sum(r["deadlock"] for r in g) / n,
                        moving=sum(r["moving"] for r in g) / n,
                        violations=sum(r["violation"] for r in g),
                        threaded_ci_lo=lo, threaded_ci_hi=hi))
    return out


def _load_rows(path):
    import csv
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k in ("geom", "reason"):
                continue
            try:
                r[k] = float(v) if ("." in v or "e" in v.lower() or v in ("nan", "inf")) else int(v)
            except ValueError:
                pass
    return rows


def main():
    p = base_parser("gap_threshold")
    p.add_argument("--plot-only", action="store_true",
                   help="regenerate figures from results/gap_threshold/runs.csv; runs nothing "
                        "and writes nothing under results/")
    a = p.parse_args()
    cfg = load_config(a.config)
    d = out_dir("gap_threshold", a.out)
    if a.plot_only:
        rows = _load_rows(d / "runs.csv")
        _figure(rows, [r for r in rows if r["rho"] > 0])
        return
    tasks = cells(cfg, a.quick)
    print(f"gap_threshold: {len(tasks)} runs, workers={a.workers}, quick={a.quick}")
    rows = run_all(tasks, one, a.workers)
    write_csv(d / "runs.csv", rows)
    tab = table_one(rows)
    write_csv(d / "table1.csv", tab)
    with open(d / "table1.tex", "w") as fh:
        fh.write("\\begin{tabular}{@{}lccccc@{}}\\toprule\n")
        fh.write(" & runs & threaded & deadlocked & still moving & violations\\\\ \\midrule\n")
        for r in tab:
            fh.write(f"${r['band']}$ & ${r['runs']}$ & ${r['threaded']:.3f}$ & "
                     f"${r['deadlocked']:.3f}$ & ${r['moving']:.3f}$ & "
                     f"$\\mathbf{{{r['violations']}}}$\\\\\n")
        fh.write("\\bottomrule\\end{tabular}\n")
    soft = [r for r in rows if r["rho"] > 0]
    below = [r for r in soft if r["kappa"] < LN2]
    summary = dict(
        n_softmin_runs=len(soft), n_exact_runs=len(rows) - len(soft),
        n_below_threshold=len(below),
        threaded_below_threshold=sum(r["threaded"] for r in below),
        smallest_successful_kappa=min([r["kappa"] for r in soft if r["threaded"]],
                                      default=float("nan")),
        min_leaf_all_runs=min(r["min_h"] for r in rows),
        table1=tab)
    with open(d / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "table1"}, indent=1))
    _figure(rows, soft)


def _figure(rows, soft):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for gi, (geom, c, mk) in enumerate((("box", "tab:blue", "o"), ("disk", "tab:red", "s"))):
        g = [r for r in soft if r["geom"] == geom]
        if not g:
            continue
        pts = {}
        for r in g:
            pts.setdefault(round(r["kappa"], 6), []).append(r["threaded"])
        ks = sorted(pts)
        ax[0].plot(ks, [np.mean(pts[k]) for k in ks], mk, ms=4.5, color=c, alpha=0.85,
                   label=f"{geom}")
        pr = {}
        for r in g:
            pr.setdefault(r["rho"], []).append(r["threaded"])
        rs = sorted(pr)
        ax[1].plot(rs, [np.mean(pr[r]) for r in rs], "-o", ms=3, color=c, alpha=0.8,
                   label=f"{geom}")
    ax[0].axvline(LN2, color="k", ls="--", lw=1.8)
    ax[0].set_xscale("log"); ax[0].set_xlabel(r"$\kappa=\rho\,\sup_{\mathcal{X}}h_{\min}$")
    ax[0].set_ylabel("gap-threading success rate"); ax[0].legend(frameon=False)
    ax[0].set_title(r"(a) collapse, threshold at $\ln 2$")
    ax[1].set_xscale("log"); ax[1].set_xlabel(r"$\rho$ (not collapsed)")
    ax[1].set_title("(b) negative control"); ax[1].legend(frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig1_gap_threshold.{ext}", dpi=200)
    print(f"wrote {FIGURES}/fig1_gap_threshold.pdf")


if __name__ == "__main__":
    main()
