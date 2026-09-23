"""k_max ablation: single most-active constraint (k_max = 1) vs simultaneous enforcement.

Reports violations and the total variation TV(u*) of the filtered control.
"""
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
    kms = [1, 4] if quick else cfg["k_max_values"]
    lips = [False] if quick else cfg["lipschitz_activation"]
    out, idx = [], 0
    for km in kms:
        for lip in lips:
            for g in range(ngoal):
                for s in range(nseed):
                    out.append(dict(k_max=km, lip=bool(lip), goal=g, run_index=idx,
                                    seed=run_seed(cfg["base_seed"], idx),
                                    delta=cfg["delta"],
                                    T=(cfg["integration"]["quick_run_length_T"] if quick
                                   else cfg["integration"]["run_length_T"])))
                    idx += 1
    return out


def one(t):
    filt = make_filter("exact", GS, delta=t["delta"], k_max=t["k_max"],
                       lipschitz_activation=t["lip"])
    run = run_closed_loop(GS.X0, Unicycle, GS, filt, Cost(GS.GOALS[t["goal"]]),
                          seed=t["seed"], T=t["T"])
    u = np.array(run["u"])
    hj = np.array(run["hj"])
    minh = float(hj.min())
    tv = float(np.abs(np.diff(u, axis=0)).sum())
    return dict(k_max=t["k_max"], lipschitz=int(t["lip"]), goal=t["goal"], seed=t["seed"],
                run_index=t["run_index"], min_h=minh, violation=int(minh < -1e-9), tv_u=tv)


def main():
    a = base_parser("kmax_ablation").parse_args()
    cfg = load_config(a.config)
    d = out_dir("kmax_ablation", a.out)
    tk = tasks_for(cfg, a.quick)
    print(f"kmax_ablation: {len(tk)} runs, workers={a.workers}, quick={a.quick}")
    rows = run_all(tk, one, a.workers)
    write_csv(d / "runs.csv", rows)
    agg = {}
    for r in rows:
        k = f"k_max={r['k_max']}" + ("_lip" if r["lipschitz"] else "")
        agg.setdefault(k, dict(runs=0, violations=0, min_h=float("inf"), tv=[]))
        agg[k]["runs"] += 1
        agg[k]["violations"] += r["violation"]
        agg[k]["min_h"] = min(agg[k]["min_h"], r["min_h"])
        agg[k]["tv"].append(r["tv_u"])
    for k in agg:
        agg[k]["tv_mean"] = float(np.mean(agg[k].pop("tv")))
    with open(d / "summary.json", "w") as fh:
        json.dump(agg, fh, indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
