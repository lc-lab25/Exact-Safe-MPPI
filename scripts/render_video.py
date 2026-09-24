"""Render an mp4 from a logged run (matplotlib animation + ffmpeg).

  python scripts/render_video.py --all
  python scripts/render_video.py --name disk_gap_exact_threads

Videos are written to media/videos/<name>.mp4 with exactly the filenames the README and
the project page expect, so dropping a rendered file into place needs no other edit.
"""
import argparse, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "media" / "videos"

from exact_safe_mppi import environments as E                       # noqa: E402
from exact_safe_mppi.dynamics import DoubleIntegrator, Unicycle     # noqa: E402
from exact_safe_mppi.mppi import Cost, make_filter, run_closed_loop  # noqa: E402

GS_MAP = E.nine_constraint

# name -> (environment factory, list of arms to draw, run length)
SPECS = {
    "teaser":                        ("disk", [("softmin", 1.0), ("exact", None)], 12.0),
    "disk_gap_softmin_rho1_deadlock": ("disk", [("softmin", 1.0)], 12.0),
    "disk_gap_exact_threads":         ("disk", [("exact", None)], 12.0),
    "box_gap_softmin_vs_exact":       ("box",  [("softmin", 1.0), ("exact", None)], 12.0),
    "nine_constraint_exact_safe":     ("gs",   [("exact", None)], 10.0),
    "nine_constraint_softmin_rho200": ("gs",   [("softmin", 200.0)], 10.0),
    "nine_constraint_shield":         ("gs",   [("shield", None)], 10.0),
    "kmax1_ablation_chattering":      ("gs",   [("exact", None)], 10.0),
}


def simulate(kind, arm, rho, T, seed=0, k_max=4):
    if kind == "gs":
        env = E.nine_constraint
        dyn, x0, goal = Unicycle, env.X0, env.GOALS[0]
        delta = 0.1
    else:
        env = E.DiskGap(0.12) if kind == "disk" else E.RoundedSquareGap(0.12)
        dyn, (x0, goal) = DoubleIntegrator, env.start_goal()
        delta = 0.0666
    filt = make_filter(arm, env, rho=rho, delta=delta, k_max=k_max)
    run = run_closed_loop(x0, dyn, env, filt, Cost(goal), seed=seed, T=T)
    return env, np.array(run["x"])[:, :2], np.array(run["u"])


def _pnorm_ring(ax, bx, by, axx, ayy, c, p, **kw):
    """Outline of {|ax(qx-bx)|^p + |ay(qy-by)|^p = c^p}, the p-norm barrier's zero level."""
    t = np.linspace(0, 2 * np.pi, 400)
    ct, st = np.cos(t), np.sin(t)
    x = bx + (c / axx) * np.sign(ct) * np.abs(ct) ** (2.0 / p)
    y = by + (c / ayy) * np.sign(st) * np.abs(st) ** (2.0 / p)
    ax.plot(x, y, **kw)


def _draw_gs(ax):
    """The nine-constraint map: six p-norm obstacles and the outer wall."""
    p = GS_MAP.P_NORM
    for _, bx, by, axx, ayy, c in GS_MAP.OBSTACLES:
        _pnorm_ring(ax, bx, by, axx, ayy, c, p, color="k", lw=1.2)
    _, bx, by, axx, ayy, c = GS_MAP.WALL
    _pnorm_ring(ax, bx, by, axx, ayy, c, p, color="0.4", lw=1.2)
    lim = 11.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)


def render(name, fps=30, dpi=140):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    kind, arms, T = SPECS[name]
    k_max = 1 if name.startswith("kmax1") else 4
    panels = [(a, r, *simulate(kind, a, r, T, k_max=k_max)) for a, r in arms]
    show_u = name.startswith("kmax1")          # chattering lives in u, not in the path
    n = len(panels) + (1 if show_u else 0)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 5.0), squeeze=False)
    lines, dots = [], []
    for ax, (arm, rho, env, q, u) in zip(axes[0], panels):
        if kind in ("disk", "box"):
            th = np.linspace(0, 2 * np.pi, 200)
            for s_ in (+1, -1):
                if kind == "disk":
                    ax.plot(env.r * np.cos(th), s_ * env.d + env.r * np.sin(th), "k-", lw=1.2)
                else:
                    _pnorm_ring(ax, 0.0, s_ * env.d, 1.0, 1.0, env.c, env.p, color="k", lw=1.2)
            ax.axvline(0.0, color="g", ls="-.", lw=1)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-3, 3)
        else:
            _draw_gs(ax)
        ax.set_aspect("equal")
        ax.set_xlabel("$q_x$")
        ax.set_ylabel("$q_y$")
        ax.set_title(f"{arm}" + (f"  $\\rho={rho:g}$" if rho else "")
                     + (f"  $k_{{\\max}}={k_max}$" if show_u else ""))
        (ln,) = ax.plot([], [], lw=1.8)
        (dt,) = ax.plot([], [], "o", ms=6)
        lines.append(ln)
        dots.append(dt)
    nf = min(len(p_[3]) for p_ in panels)

    uax = None
    if show_u:
        uax = axes[0][-1]
        u = panels[0][4]
        tv = float(np.abs(np.diff(u, axis=0)).sum())
        tt = np.arange(len(u)) * (T / max(1, len(u) - 1))
        uax.plot(tt, u[:, 0], lw=0.8, color="0.7")
        uax.plot(tt, u[:, 1], lw=0.8, color="0.85")
        (ul0,) = uax.plot([], [], lw=1.2)
        (ul1,) = uax.plot([], [], lw=1.2)
        uax.set_xlim(0, T)
        m = float(np.abs(u).max()) * 1.1 + 1e-9
        uax.set_ylim(-m, m)
        uax.set_xlabel("$t$ [s]")
        uax.set_ylabel("$u^*$")
        uax.set_title(f"filtered control,  TV$(u^*)={tv:.0f}$")
        lines += [ul0, ul1]

    def upd(i):
        for ln, dt, (_a, _r, _e, q, _u) in zip(lines, dots, panels):
            ln.set_data(q[:i + 1, 0], q[:i + 1, 1])
            dt.set_data([q[i, 0]], [q[i, 1]])
        if uax is not None:
            uu = panels[0][4]
            j = min(i, len(uu) - 1)
            tt2 = np.arange(j + 1) * (T / max(1, len(uu) - 1))
            lines[-2].set_data(tt2, uu[:j + 1, 0])
            lines[-1].set_data(tt2, uu[:j + 1, 1])
        return lines + dots

    ani = animation.FuncAnimation(fig, upd, frames=range(0, nf, max(1, nf // (fps * 10))),
                                  blit=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.mp4"
    ani.save(path, writer=animation.FFMpegWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", choices=sorted(SPECS))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    names = sorted(SPECS) if a.all else ([a.name] if a.name else [])
    if not names:
        ap.error("give --name NAME or --all")
    for n in names:
        render(n)


if __name__ == "__main__":
    main()
