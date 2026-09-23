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
    return env, np.array(run["x"])[:, :2]


def render(name, fps=30, dpi=140):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    kind, arms, T = SPECS[name]
    k_max = 1 if name.startswith("kmax1") else 4
    panels = [(a, r, *simulate(kind, a, r, T, k_max=k_max)) for a, r in arms]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 5.0), squeeze=False)
    lines, dots = [], []
    for ax, (arm, rho, env, q) in zip(axes[0], panels):
        if kind in ("disk", "box"):
            th = np.linspace(0, 2 * np.pi, 200)
            for s in (+1, -1):
                if kind == "disk":
                    ax.plot(env.r * np.cos(th), s * env.d + env.r * np.sin(th), "k-", lw=1.2)
                else:
                    ax.plot(env.c * np.sign(np.cos(th)) * np.abs(np.cos(th)) ** (2 / env.p),
                            s * env.d + env.c * np.sign(np.sin(th))
                            * np.abs(np.sin(th)) ** (2 / env.p), "k-", lw=1.2)
            ax.axvline(0.0, color="g", ls="-.", lw=1)
            ax.set_xlim(-4, 4); ax.set_ylim(-3, 3)
        ax.set_aspect("equal"); ax.set_xlabel("$q_x$"); ax.set_ylabel("$q_y$")
        ax.set_title(f"{arm}" + (f"  $\\rho={rho:g}$" if rho else ""))
        (ln,) = ax.plot([], [], lw=1.8)
        (dt,) = ax.plot([], [], "o", ms=6)
        lines.append(ln); dots.append(dt)
    nf = min(len(p[3]) for p in panels)

    def upd(i):
        for ln, dt, (_, _, _, q) in zip(lines, dots, panels):
            ln.set_data(q[:i + 1, 0], q[:i + 1, 1])
            dt.set_data([q[i, 0]], [q[i, 1]])
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
