"""Theorem 1(iii): at k_delta = 1 the LCP filter equals the scalar law with h_min.

Samples states until the target number with |A_delta| = 1 is reached, then reports the
maximum difference between the two control laws over those states.
"""
import json

import numpy as np

from _common import base_parser, out_dir
from exact_safe_mppi import environments as E
from exact_safe_mppi.lcp_filter import exact_lcp
from exact_safe_mppi.softmin_filter import softmin_closed_form

TARGET = 10030


def main():
    p = base_parser("lcp_vs_scalar")
    p.add_argument("--n-states", type=int, default=TARGET)
    a = p.parse_args()
    n_target = 300 if a.quick else a.n_states
    d = out_dir("lcp_vs_scalar", a.out)
    env = E.DiskGap(0.12)
    rng = np.random.default_rng(0)
    Xs = []
    n = 0
    while n < n_target:
        X = np.stack([rng.uniform(-3, 3, 200000), rng.uniform(-3, 3, 200000),
                      rng.uniform(-6, 6, 200000), rng.uniform(-6, 6, 200000)], 1)
        b, _, _ = env.beta_jet(X)
        k = ((b - b.min(-1, keepdims=True)) <= 0.1).sum(-1)
        m = (env.h_np(X).min(-1) >= 0) & (k == 1)
        if m.any():
            Xs.append(X[m]); n += int(m.sum())
    X = np.vstack(Xs)[:n_target]
    b, lf, lg = env.beta_jet(X)
    v = rng.standard_normal((len(X), 2))
    u_lcp, _ = exact_lcp(b, lf, lg, v, env.GAMMA, env.alpha, delta=0.1, k_max=4, check=False)
    # the scalar law at k_delta = 1 is the soft-min closed form evaluated at the ACTIVE beta,
    # i.e. rho -> infinity so that h_rho -> h_min; taking the single active column gives it
    j = b.argmin(-1)
    i = np.arange(len(X))
    b1 = b[i, j][:, None]; lf1 = lf[i, j][:, None]; lg1 = lg[i, j][:, None, :]
    u_sc, _ = softmin_closed_form(b1, lf1, lg1, v, 1.0, env.GAMMA, env.alpha)
    err = float(np.abs(u_lcp - u_sc).max())
    out = dict(n_states=len(X), k_delta=1, max_abs_difference=err)
    with open(d / "lcp_vs_scalar.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
