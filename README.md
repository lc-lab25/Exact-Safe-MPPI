# Composing Exact Barriers Inside an MPPI Planner

**Alex Coker and Leilei Cui**
Department of Mechanical Engineering, University of New Mexico
*Submitted to the American Control Conference (ACC) 2027*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/alexkcoker/exact-safe-mppi/actions/workflows/ci.yml/badge.svg)](https://github.com/alexkcoker/exact-safe-mppi/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)

<a href="media/videos/teaser.mp4"><img src="media/placeholders/teaser.svg" width="640" alt="Soft-min vs exact, disk gap"></a>

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

Videos are not committed yet; each row shows a placeholder card that links to the file path it
will occupy.

| Preview | Content |
|---|---|
| <a href="media/videos/teaser.mp4"><img src="media/placeholders/teaser.svg" width="320" alt="Soft-min vs exact, disk gap"></a> | Soft-min vs exact, disk gap |
| <a href="media/videos/disk_gap_softmin_rho1_deadlock.mp4"><img src="media/placeholders/disk_gap_softmin_rho1_deadlock.svg" width="320" alt="Disk gap w=0.12, ρ=1 (κ=0.606 < ln 2): soft min halts"></a> | Disk gap w=0.12, ρ=1 (κ=0.606 < ln 2): soft min halts |
| <a href="media/videos/disk_gap_exact_threads.mp4"><img src="media/placeholders/disk_gap_exact_threads.svg" width="320" alt="Disk gap w=0.12: exact composition threads the gap"></a> | Disk gap w=0.12: exact composition threads the gap |
| <a href="media/videos/box_gap_softmin_vs_exact.mp4"><img src="media/placeholders/box_gap_softmin_vs_exact.svg" width="320" alt="Rounded-square gap (p=8): soft min vs exact"></a> | Rounded-square gap (p=8): soft min vs exact |
| <a href="media/videos/nine_constraint_exact_safe.mp4"><img src="media/placeholders/nine_constraint_exact_safe.svg" width="320" alt="Nine-constraint map: Exact-Safe MPPI"></a> | Nine-constraint map: Exact-Safe MPPI |
| <a href="media/videos/nine_constraint_softmin_rho200.mp4"><img src="media/placeholders/nine_constraint_softmin_rho200.svg" width="320" alt="Nine-constraint map: soft min, ρ=200"></a> | Nine-constraint map: soft min, ρ=200 |
| <a href="media/videos/nine_constraint_shield.mp4"><img src="media/placeholders/nine_constraint_shield.svg" width="320" alt="Nine-constraint map: discrete-time repair baseline"></a> | Nine-constraint map: discrete-time repair baseline |
| <a href="media/videos/kmax1_ablation_chattering.mp4"><img src="media/placeholders/kmax1_ablation_chattering.svg" width="320" alt="Single most-active constraint (k_max=1): chattering"></a> | Single most-active constraint (k_max=1): chattering |

> **Swapping in a real video.** GitHub does not play relative-path `.mp4` files inline in a
> README. Once a video exists, either (a) replace the placeholder `<img>` with a GIF preview
> that links to the mp4, or (b) drag the mp4 into a GitHub issue or PR comment and use the
> resulting `user-attachments` URL, which does play inline. The project page under `docs/`
> plays the mp4 files directly and needs no such workaround.

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
media/                 placeholder cards now, videos later
docs/                  single-file GitHub Pages project page
```

## Notes on reconstruction

These are limitations of the experimental setup, stated plainly:

- The nine-constraint benchmark geometry was **reconstructed from figure ticks** in the source
  publication, not obtained from the authors.
- The box barrier exponent **p = 8** was selected by agreement with the published trajectory
  cloud, not taken from a published value.
- The MPPI temperature **λ = 300 was used instead of the published λ = 1**. At λ = 1 the
  first-update effective sample size is 1.00 of K = 1000, so the weighted update carries no more
  information than a single sample. The ESS is logged for every run.
- All guarantees in the paper are **continuous-time**. Rollouts here take one Euler step per
  planner step (T_s = 0.1 s) and execution uses substeps (δt = 2×10⁻³ s), so the reported
  results are properties of the discretised planner.

## GitHub Pages

Settings → Pages → *Deploy from a branch* → branch `main`, folder `/docs`.

`docs/` cannot reach files outside itself on Pages, so run:

```bash
make site      # copies media/ into docs/assets/media/
```

The page references `assets/media/...` and falls back to the placeholder card whenever a video
file is absent.

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
