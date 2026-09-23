# Media

Videos are **not** committed yet. Drop a file into `videos/` using exactly the filename below
and it will be picked up by the README, the project page and `make site` with no other edit.

| Expected filename | Content |
|---|---|
| `teaser.mp4` | Soft-min vs exact, disk gap |
| `disk_gap_softmin_rho1_deadlock.mp4` | Disk gap  w=0.12,  rho=1  (kappa=0.606 < ln 2):  soft min halts |
| `disk_gap_exact_threads.mp4` | Disk gap  w=0.12:  exact composition threads the gap |
| `box_gap_softmin_vs_exact.mp4` | Rounded-square gap (p=8):  soft min vs exact |
| `nine_constraint_exact_safe.mp4` | Nine-constraint map:  Exact-Safe MPPI |
| `nine_constraint_softmin_rho200.mp4` | Nine-constraint map:  soft min, rho=200 |
| `nine_constraint_shield.mp4` | Nine-constraint map:  discrete-time repair baseline |
| `kmax1_ablation_chattering.mp4` | Single most-active constraint (k_max=1):  chattering |

`placeholders/` holds one 16:9 SVG card per video, shown wherever the real file is missing.
Regenerate any video with `python scripts/render_video.py --name <name>` (needs ffmpeg).
