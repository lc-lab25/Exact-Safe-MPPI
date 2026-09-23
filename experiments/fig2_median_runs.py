"""Fig. 2: median soft-min (rho = 1) and exact runs on the disk gap at w = 0.12."""
import json

import numpy as np

from _common import base_parser, load_config, out_dir, write_csv, FIGURES
from exact_safe_mppi import environments as E
from exact_safe_mppi.dynamics import DoubleIntegrator
from exact_safe_mppi.mppi import Cost, make_filter, run_closed_loop
from exact_safe_mppi.utils import run_seed


def one(env, kind, rho, seed, delta, T):
    x0, qg = env.start_goal()
    filt = make_filter(kind, env, rho=rho, delta=delta, k_max=4)
    run = run_closed_loop(x0, DoubleIntegrator, env, filt, Cost(qg), seed=seed, T=T)
    q = np.array(run["x"])[:, :2]
    return dict(q=q, threaded=bool((q[:, 0] >= 0).any() and (q[:, 0] <= 0).any()),
                final_qx=float(q[-1, 0]), path_len=float(np.abs(np.diff(q, axis=0)).sum()),
                final_dist=float(np.linalg.norm(q[-1] - qg)), seed=seed)


def main():
    p = base_parser("fig2_median_runs")
    p.add_argument("--plot-only", action="store_true",
                   help="regenerate the figure from the stored median seeds; runs only those "
                        "two trajectories and writes nothing under results/")
    a = p.parse_args()
    cfg = load_config("gap_threshold")
    d = out_dir("fig2_median_runs", a.out)
    if a.plot_only:
        import csv
        with open(d / "runs.csv") as fh:
            rows = [r for r in csv.DictReader(fh) if r["is_median"] == "1"]
        env = E.DiskGap(0.12)
        delta = cfg["delta_star"]["disk_0.12"]
        T = cfg["integration"]["run_length_T"]
        keep = {r["arm"]: one(env, r["arm"], (float(r["rho"]) if float(r["rho"]) > 0 else None),
                              int(r["seed"]), delta, T) for r in rows}
        _plot(keep, env)
        return
    n = 3 if a.quick else cfg["n_seeds"]
    env = E.DiskGap(0.12)
    delta = cfg["delta_star"]["disk_0.12"]
    T = (cfg["integration"]["quick_run_length_T"] if a.quick
         else cfg["integration"]["run_length_T"])
    rows, keep = [], {}
    for kind, rho in (("softmin", 1.0), ("exact", None)):
        runs = [one(env, kind, rho, run_seed(cfg["base_seed"], i), delta, T) for i in range(n)]
        # median by final distance to goal for the stalling arm, by path length for threading
        key = "final_dist" if kind == "softmin" else "path_len"
        order = sorted(range(len(runs)), key=lambda i: runs[i][key])
        med = order[len(order) // 2]
        keep[kind] = runs[med]
        for r in runs:
            rows.append(dict(arm=kind, rho=(rho if rho else -1.0), seed=r["seed"],
                             threaded=int(r["threaded"]), final_qx=r["final_qx"],
                             final_dist=r["final_dist"], path_len=r["path_len"],
                             is_median=int(r is runs[med])))
    write_csv(d / "runs.csv", rows)
    summary = {f"{k}_median_final_qx": float(v["final_qx"]) for k, v in keep.items()}
    summary.update({f"{k}_median_threaded": bool(v["threaded"]) for k, v in keep.items()})
    with open(d / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    _plot(keep, env)


def _plot(keep, env):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(5.2, 6.0), sharex=True)
    for i, (kind, ttl) in enumerate((("softmin", r"(a) soft minimum, $\rho=1$"),
                                     ("exact", "(b) exact composition"))):
        q = keep[kind]["q"]
        for sgn in (+1, -1):
            th = np.linspace(0, 2 * np.pi, 200)
            ax[i].plot(env.r * np.cos(th), sgn * env.d + env.r * np.sin(th), "k-", lw=1)
        ax[i].plot(q[:, 0], q[:, 1], lw=1.6)
        ax[i].axvline(0.0, color="g", ls="-.", lw=1)
        ax[i].set_ylabel(r"$q_y$"); ax[i].set_title(ttl); ax[i].set_ylim(-2, 2)
    ax[1].set_xlabel(r"$q_x$")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig2_median_runs.{ext}", dpi=200)
    print(f"wrote {FIGURES}/fig2_median_runs.pdf")


if __name__ == "__main__":
    main()
