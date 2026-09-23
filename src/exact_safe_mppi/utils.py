"""Seeding, logging and effective sample size."""
import numpy as np

from ._logging import log_path, save_run, load_run          # noqa: F401


def run_seed(base_seed, run_index):
    """Every run is seeded independently, so results do not depend on --workers."""
    return int(base_seed) + int(run_index)


def rng_for(base_seed, run_index):
    return np.random.default_rng(run_seed(base_seed, run_index))


def effective_sample_size(weights):
    """ESS = 1 / sum_j w_j^2 for normalised weights; K for a uniform weighting."""
    w = np.asarray(weights, float)
    w = w / w.sum(-1, keepdims=True)
    return float(1.0 / (w * w).sum(-1))


__all__ = ["run_seed", "rng_for", "effective_sample_size",
           "log_path", "save_run", "load_run"]
