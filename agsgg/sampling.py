"""Repeat Factor Sampling (LVIS) thuần numpy: r_c = max(1, sqrt(t/f_c)); r_i = min(cap, max_c∈i r_c)."""
import numpy as np


def rfs_repeat_factors(presence, t=0.1, cap=10.0):
    """presence: [N, C] bool (frame i có lớp c). Trả (r_i [N], r_c [C], f_c [C])."""
    f = presence.mean(0)
    rc = np.where(f > 0, np.maximum(1.0, np.sqrt(t / np.maximum(f, 1e-12))), 1.0)
    ri = np.where(presence, rc[None, :], 1.0).max(1)
    return np.minimum(ri, cap), rc, f
