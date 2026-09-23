"""Closed-form checks on the disk gap slice (no simulation).

  sup_X h_min = a*w*(2r+w)            rho_crit(w) = ln 2 / (a*w*(2r+w))
  soft-min peak over X at a given rho, and the deficit sup h_min - sup h_rho = ln 2 / rho
"""
import json

import numpy as np

from _common import base_parser, out_dir
from exact_safe_mppi import environments as E
from exact_safe_mppi.barriers import CASCADE_GAIN as A
from exact_safe_mppi.composition import softmin_stable

LN2 = float(np.log(2.0))


def slice_states(env, n_q=401, n_v=2001, vmax=20.0):
    qy = np.linspace(-env.w, env.w, n_q)
    vy = np.linspace(-vmax, vmax, n_v)
    Q, V = np.meshgrid(qy, vy, indexing="ij")
    Z = np.zeros(Q.size)
    return np.stack([Z, Q.ravel(), Z, V.ravel()], 1)


def main():
    a = base_parser("closed_form_checks").parse_args()
    d = out_dir("closed_form_checks", a.out)
    w = 0.12
    env = E.DiskGap(w)
    r = env.r
    H0 = A * w * (2 * r + w)
    X = slice_states(env)
    beta, _, _ = env.beta_jet(X)
    in_C = env.h_np(X).min(-1) >= 0
    sup_hmin = float(beta.min(-1)[in_C].max())
    out = dict(width=w, disk_radius=r, cascade_gain_a=A,
               sup_hmin_closed_form=H0, sup_hmin_measured=sup_hmin,
               sup_hmin_abs_err=abs(sup_hmin - H0),
               rho_crit_closed_form=LN2 / H0)
    for rho in (1.0, 2.0):
        sup = float(softmin_stable(beta, rho)[in_C].max())
        out[f"sup_hrho_rho{rho:g}"] = sup
        out[f"deficit_rho{rho:g}"] = sup_hmin - sup
        out[f"deficit_minus_ln2_over_rho_rho{rho:g}"] = (sup_hmin - sup) - LN2 / rho
    with open(d / "closed_form_checks.json", "w") as fh:
        json.dump(out, fh, indent=1)
    for k, v in out.items():
        print(f"  {k:38s} {v!r}")


if __name__ == "__main__":
    main()
