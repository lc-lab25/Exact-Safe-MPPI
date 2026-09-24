# Composing Exact Barriers Inside an MPPI Planner

**Alex Coker and Leilei Cui**
Department of Mechanical Engineering, University of New Mexico
*Submitted to the American Control Conference (ACC) 2027*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/alexkcoker/exact-safe-mppi/actions/workflows/ci.yml/badge.svg)](https://github.com/alexkcoker/exact-safe-mppi/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)

<a href="media/videos/teaser.mp4"><img src="media/gifs/teaser.gif" width="720" alt="Soft-min vs exact, disk gap"></a>

## Abstract

Exact-Safe MPPI enforces the exact minimum of multiple (high-order) control barrier functions
inside every MPPI rollout using a closed-form linear complementarity (LCP) filter. This avoids
the certified-set shrinkage of the soft-minimum composition used by GS-MPPI, which can close
feasible gaps and cause deadlock below the threshold κ_crit = ln k.

**Key results**

- Below κ_crit the soft-minimum planner does not thread the gap in **0 of 1050** runs, and its
  failure mode is deadlock rather than constraint violation.
- The LCP filter reduces to the scalar law exactly when one constraint is active: maximum
  difference **8.9×10⁻¹⁶** over 10,030 sampled states.
- On the nine-constraint map the exact composition records **0 violations in 400 runs**, against
  **34** and **53** for the soft minimum at ρ = 200 and ρ = 1000.

## Videos

Each preview below is a looping GIF that plays here in the README, no click and no navigation.
Click one to open the full-quality mp4 in `media/videos/`. Both are tracked by git.

| Preview | Content |
|---|---|
| <a href="media/videos/teaser.mp4"><img src="media/gifs/teaser.gif" width="380" alt="Soft-min vs exact, disk gap"></a> | Soft-min vs exact, disk gap |
| <a href="media/videos/disk_gap_softmin_rho1_deadlock.mp4"><img src="media/gifs/disk_gap_softmin_rho1_deadlock.gif" width="380" alt="Disk gap w=0.12, ρ=1 (κ=0.606 < ln 2): soft min halts"></a> | Disk gap w=0.12, ρ=1 (κ=0.606 < ln 2): soft min halts |
| <a href="media/videos/disk_gap_exact_threads.mp4"><img src="media/gifs/disk_gap_exact_threads.gif" width="380" alt="Disk gap w=0.12: exact composition threads the gap"></a> | Disk gap w=0.12: exact composition threads the gap |
| <a href="media/videos/box_gap_softmin_vs_exact.mp4"><img src="media/gifs/box_gap_softmin_vs_exact.gif" width="380" alt="Rounded-square gap (p=8): soft min vs exact"></a> | Rounded-square gap (p=8): soft min vs exact |
| <a href="media/videos/nine_constraint_exact_safe.mp4"><img src="media/gifs/nine_constraint_exact_safe.gif" width="380" alt="Nine-constraint map: Exact-Safe MPPI"></a> | Nine-constraint map: Exact-Safe MPPI |
| <a href="media/videos/nine_constraint_softmin_rho200.mp4"><img src="media/gifs/nine_constraint_softmin_rho200.gif" width="380" alt="Nine-constraint map: soft min, ρ=200"></a> | Nine-constraint map: soft min, ρ=200 |
| <a href="media/videos/nine_constraint_shield.mp4"><img src="media/gifs/nine_constraint_shield.gif" width="380" alt="Nine-constraint map: discrete-time repair baseline"></a> | Nine-constraint map: discrete-time repair baseline |
| <a href="media/videos/kmax1_ablation_chattering.mp4"><img src="media/gifs/kmax1_ablation_chattering.gif" width="380" alt="Single most-active constraint (k_max=1): chattering"></a> | Single most-active constraint (k_max=1): chattering |

The GIFs are built from the mp4s at 12 fps by `scripts/make_gifs.sh` (`make gifs`), which uses a
per-clip `palettegen`/`paletteuse` pass; they total about 1.4 MB. The mp4s are the reference
copies — regenerate those with `scripts/render_video.py`, then rebuild the GIFs from them.

### Adding a video

```bash
# 1. put the file in place, using exactly the name from the table above
#    (or regenerate it: python scripts/render_video.py --name disk_gap_exact_threads)
cp /path/to/clip.mp4 media/videos/disk_gap_exact_threads.mp4

# 2. rebuild the inline GIF previews
make gifs

# 3. commit both
git add media/videos/disk_gap_exact_threads.mp4 media/gifs/disk_gap_exact_threads.gif
git commit -m "Add disk gap exact threading video"
```

No other edit is needed: the README already points at `media/gifs/<name>.gif` and links to
`media/videos/<name>.mp4` under the same name.

> **Why GIFs.** GitHub does not play a relative-path `.mp4` inline in a README — it renders a
> link, not a player. A GIF autoplays and loops, so the previews above are animated without the
> reader leaving the page. The mp4s remain the higher-quality copies behind each link.

## Installation

```bash
# pip
python -m pip install -r requirements.txt
python -m pip install -e .

# conda
conda env create -f environment.yml
conda activate exact-safe-mppi
python -m pip install -e .
```

`make videos` additionally needs `ffmpeg` on `PATH` (it is in `environment.yml`).

On UNM CARC Easley the `ffmpeg/7.1` modulefile loads but does not set `PATH`; use the binary
directly:

```bash
export PATH=/opt/spack/opt/spack/linux-sapphirerapids/ffmpeg-7.1-pod7gebu4qodjdoap3rznh4bk5ihhlm3/bin:$PATH
```

## Quick start

```bash
make quick        # every experiment, 3 seeds and a reduced grid
```

## Full reproduction

```bash
make reproduce                 # serial
make reproduce WORKERS=8       # parallel; results are identical for any WORKERS
```

Every run is seeded as `seed = base_seed + run_index`, and the seed is written into each output
row, so `--workers` changes only the wall time.

| Paper item | Command | Output |
|---|---|---|
| Fig. 1, Table I | `python experiments/gap_threshold.py` | `results/gap_threshold/{runs,table1}.csv`, `table1.tex`, `figures/fig1_gap_threshold.pdf` |
| Fig. 2 | `python experiments/fig2_median_runs.py` | `results/fig2_median_runs/`, `figures/fig2_median_runs.pdf` |
| Closed-form checks | `python experiments/closed_form_checks.py` | `results/closed_form_checks/closed_form_checks.json` |
| LCP vs scalar law | `python experiments/lcp_vs_scalar.py` | `results/lcp_vs_scalar/lcp_vs_scalar.json` |
| Nine-constraint safety | `python experiments/nine_constraint.py` | `results/nine_constraint/{runs.csv,summary.json}` |
| k_max ablation | `python experiments/kmax_ablation.py` | `results/kmax_ablation/{runs.csv,summary.json}` |
| Verification | `python scripts/verify_results.py` | PASS/FAIL table on stdout |
| Figures only | `make figures` | reads `results/*/runs.csv`, rewrites `figures/`; reruns nothing |

**Runtime.** `make quick` is **measured at 9.7 minutes** on one core (Xeon-class server,
`--workers 1`):

| Experiment | quick (measured) | full scale (estimate) |
|---|---|---|
| `closed_form_checks` | 17 s | 17 s |
| `lcp_vs_scalar` | 2 s | ~2 min |
| `gap_threshold` | 134 s | **~47 h** (4950 runs x 34 s) |
| `fig2_median_runs` | 30 s | ~1 h |
| `nine_constraint` | 195 s | **~26 h** (2800 runs) |
| `kmax_ablation` | 201 s | ~7 h |

The full-scale column is an **estimate**, extrapolated from the measured cost of one
closed-loop run (2.8 s per simulated second) times the run count; it is not a measurement.
Use `make reproduce WORKERS=N` — the full sweep is embarrassingly parallel and each run is
seeded independently, so any `N` gives identical results.

Quick mode uses one gap width per geometry, three seeds, and a shortened horizon
(`quick_run_length_T: 2.0` in each config). It is a smoke test, **not** a reduced-scale
reproduction: at that horizon runs have not settled, so the outcome *rates* in Table I are not
meaningful and `scripts/verify_results.py --quick` reports them as SKIP rather than comparing
them.

## Repository structure

```
src/exact_safe_mppi/   dynamics, barriers, composition, the three filters, MPPI, environments
configs/               one YAML per experiment; every value taken from the experiment code
experiments/           one script per paper result
scripts/               reproduce_all.sh, verify_results.py, render_video.py
tests/                 pytest: Lemma 3, two-barrier slice, Theorem 1(iii), LCP validity,
                       Proposition 1, determinism
results/expected/      the paper's reported numbers, used by verify_results.py
media/videos/          one mp4 per simulation clip (reference quality)
media/gifs/            the looping previews embedded in this README
```

## Outcome definitions

Every gap run is classified exactly as in the source experiment code
(`gap_threshold.py:one`, matching `r5_gap_threading.py:run_one`). Let `q` be the position
trace, `w` the gap half-width and `minh = min_t min_j h_j`.

```
crossed    = (q_x > 0.5).any()
qy_at_gap  = |q_y| at the sample minimising |q_x|          # closest approach to the gap
threaded   = crossed and qy_at_gap <= w + 0.15
speed      = max |velocity components| over the last 20 samples
violation  = minh < -1e-9
deadlock   = (not threaded) and (minh >= -1e-9) and (speed < 0.3)
moving     = (not threaded) and (not deadlock) and (not violation)
```

**Why threading requires passing through the gap.** Reaching `q_x > 0.5` is not enough. The
obstacles are finite, so a detour *around* them also reaches the goal, and counting those as
successes inverts the trend the experiment is measuring: box runs at ρ = 0.5 "succeeded" with
`|q_y| = 2.85` at the crossing, against an obstacle extent of 2.05 — they went around a barrier
that had closed the gap, and low ρ therefore looked *good*. A run counts as threaded only if its
closest approach to `x = 0` lies inside the gap itself.

**Deadlock is separated from violation** because both fail the task but only one is the effect
being measured: deadlock is a run that never violates a constraint and comes to rest, which is
the soft minimum being excluded from a region it should be able to reach. The `1e-9` slack in
`minh` is the numerical tolerance used throughout, including the nine-constraint and ablation
experiments.

## Notes on reconstruction

These are limitations of the experimental setup, stated plainly:

- The nine-constraint benchmark geometry was **reconstructed from figure ticks** in the source
  publication, not obtained from the authors.
- The box barrier exponent **p = 8** was selected by agreement with the published trajectory
  cloud, not taken from a published value.
- The MPPI temperature **λ = 300 was used instead of the published λ = 1**. At λ = 1 the
  first-update effective sample size is 1.00 of K = 1000 on the nine-constraint map, so the
  weighted update carries no more information than a single sample; at λ = 300 it is ≈ 922.
  The ESS is logged for every nine-constraint run and measured directly by
  `experiments/closed_form_checks.py`.
- **The temperature is not the same for every experiment.** The gap sweep uses a per-geometry
  value, `LAMBDA_BY_GEOM` in the source code: **λ = 30 for the rounded-square (p-norm) gap** and
  **λ = 300 for the disk gap**. The nine-constraint benchmark and the k_max ablation use
  **λ = 300** throughout. The configs record this as `lambda_by_geometry`.
- All guarantees in the paper are **continuous-time**. Rollouts here take one Euler step per
  planner step (T_s = 0.1 s) and execution uses substeps (δt = 2×10⁻³ s), so the reported
  results are properties of the discretised planner.

## Citation

```bibtex
@inproceedings{coker2027exactsafe,
  title     = {Composing Exact Barriers Inside an MPPI Planner},
  author    = {Coker, Alex and Cui, Leilei},
  booktitle = {Submitted to the American Control Conference (ACC)},
  year      = {2027}
}
```

See `CITATION.cff` for the machine-readable form.

## License and contact

MIT, see [LICENSE](LICENSE).

Alex Coker and Leilei Cui, Department of Mechanical Engineering, University of New Mexico.
