"""HDF5 logging schema: one file per (method, rho, env, goal, seed).

Every figure script reads these files and never recomputes.  Scratch, not home:
/easley/scratch/users/alexcoker1/exact-mppi/logs/
"""
import os

import h5py
import numpy as np

SCRATCH = "/easley/scratch/users/alexcoker1/exact-mppi"


def log_path(root, method, env_name, goal_idx, seed, rho=None, extra=None):
    parts = [method]
    if rho is not None:
        parts.append(f"rho{rho:g}")
    if extra:
        parts.append(extra)
    parts += [env_name, f"g{goal_idx}", f"s{seed}"]
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "__".join(parts) + ".h5")


def save_run(path, run, meta):
    """`run` is the dict returned by mppi.run_closed_loop; `meta` is a flat dict of scalars."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in meta.items():
            f.attrs[k] = v
        f.attrs["seed"] = run["seed"]
        g = f.create_group("timeseries")
        for k in ("t", "x", "v", "u", "h", "beta", "hj", "n_active",
                  "lam_min_M", "min_Lg_active", "wall", "capped",
                  "rank_B", "max_lam", "support_size", "polytope_empty",
                  "mu_beta", "lin", "slack_fraction", "in_A", "beta_act", "lam", "mu",
                  "ess", "J_mean", "J_std", "J_spread"):
            if k in run:
                g.create_dataset(k, data=np.asarray(run[k]), compression="gzip",
                                 compression_opts=4)
        sg = f.create_group("slack")
        for k, v in run.items():
            if k.startswith("slack_"):
                sg.create_dataset(k[6:], data=np.asarray(v))
        if run.get("clouds"):
            c = f.create_group("clouds")
            for it, pts in run["clouds"]:
                c.create_dataset(f"it{it:04d}", data=pts, compression="gzip",
                                 compression_opts=4)
    return path


def load_run(path):
    out = {}
    with h5py.File(path, "r") as f:
        out["attrs"] = dict(f.attrs)
        out.update({k: f["timeseries"][k][()] for k in f["timeseries"]})
        if "slack" in f:
            out.update({"slack_" + k: f["slack"][k][()] for k in f["slack"]})
        if "clouds" in f:
            out["clouds"] = {k: f["clouds"][k][()] for k in f["clouds"]}
    return out
