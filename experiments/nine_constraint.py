"""Nine-constraint benchmark: violations of min_t min_j h_j >= 0 per controller."""
import json

import numpy as np

from _common import base_parser, load_config, out_dir, run_all, write_csv
from exact_safe_mppi import environments as E
from exact_safe_mppi.dynamics import Unicycle
from exact_safe_mppi.mppi import Cost, make_filter, run_closed_loop
from exact_safe_mppi.utils import run_seed

GS = E.nine_constraint


def tasks_for(cfg, quick):
    nseed = 3 if quick else cfg["n_seeds"]
    ngoal = 1 if quick else cfg["n_goals"]
    ctrls = cfg["controllers"][:2] if quick else cfg["controllers"]
    out, idx = [], 0
    for c in ctrls:
        for g in range(ngoal):
            for s in range(nseed):
                out.append(dict(name=c["name"], rho=c["rho"], goal=g, run_index=idx,
                                seed=run_seed(cfg["base_seed"], idx), delta=cfg["delta"],
                                T=(cfg["integration"]["quick_run_length_T"] if quick
                                   else cfg["integration"]["run_length_T"])))
                idx += 1
    return out


def one(t):
    filt = make_filter(t["name"], GS, rho=t["rho"], delta=t["delta"], k_max=4)
    run = run_closed_loop(GS.X0, Unicycle, GS, filt, Cost(GS.GOALS[t["goal"]]),
                          seed=t["seed"], T=t["T"])
    hj = np.array(run["hj"])
    minh = float(hj.min())
    ess = np.asarray(run["ess"], float)
    return dict(controller=t["name"], rho=(t["rho"] if t["rho"] else -1.0), goal=t["goal"],
                seed=t["seed"], run_index=t["run_index"], min_h=minh,
                violation=int(minh < -1e-9),
                ess_first=float(ess[0]) if ess.size else float("nan"))


def main():
    a = base_parser("nine_constraint").parse_args()
    cfg = load_config(a.config)
    d = out_dir("nine_constraint", a.out)
    tk = tasks_for(cfg, a.quick)
    print(f"nine_constraint: {len(tk)} runs, workers={a.workers}, quick={a.quick}")
    rows = run_all(tk, one, a.workers)
    write_csv(d / "runs.csv", rows)
    agg = {}
    for r in rows:
        key = r["controller"] if r["rho"] < 0 else f"{r['controller']}_rho{r['rho']:g}"
        agg.setdefault(key, dict(runs=0, violations=0, min_h=float("inf")))
        agg[key]["runs"] += 1
        agg[key]["violations"] += r["violation"]
        agg[key]["min_h"] = min(agg[key]["min_h"], r["min_h"])
        agg[key].setdefault("ess_first", []).append(r["ess_first"])
    for k in agg:
        e = [x for x in agg[k].pop("ess_first", []) if x == x]
        agg[k]["ess_first_mean"] = float(np.mean(e)) if e else float("nan")
    with open(d / "summary.json", "w") as fh:
        json.dump(agg, fh, indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
