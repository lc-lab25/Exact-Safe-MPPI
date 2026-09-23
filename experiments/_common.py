"""Shared plumbing: config loading, deterministic seeding, parallel map, CSV output."""
import argparse, csv, os, pathlib, sys
from concurrent.futures import ProcessPoolExecutor

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def load_config(name):
    with open(ROOT / "configs" / f"{name}.yaml") as fh:
        return yaml.safe_load(fh)


def base_parser(experiment):
    p = argparse.ArgumentParser(description=experiment)
    p.add_argument("--config", default=experiment)
    p.add_argument("--workers", type=int, default=1,
                   help="results are identical for any N: every run is seeded independently")
    p.add_argument("--quick", action="store_true", help="3 seeds and a reduced grid")
    p.add_argument("--out", default=None)
    return p


def out_dir(experiment, out=None):
    d = pathlib.Path(out) if out else RESULTS / experiment
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_all(tasks, fn, workers):
    """Map fn over tasks. Each task carries its own seed, so order and N are irrelevant."""
    if workers <= 1:
        return [fn(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, tasks))


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows)")


def binom_ci(k, n, alpha=0.05):
    """Clopper-Pearson 95% interval."""
    from scipy.stats import beta as B
    lo = 0.0 if k == 0 else float(B.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(B.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi
