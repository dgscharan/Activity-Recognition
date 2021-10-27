import warnings
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis, iqr

import numpy as np
from scipy.stats import entropy
# from ..lib.entropy import perm_entropy, svd_entropy
from numba import jit
from math import log, floor
all = ['_embed', '_linear_regression', '_log_n']

def _embed(x, order=3, delay=1):
    N = len(x)
    if order * delay > N:
        raise ValueError("Error: order * delay should be lower than x.size")
    if delay < 1:
        raise ValueError("Delay has to be at least 1.")
    if order < 2:
        raise ValueError("Order has to be at least 2.")
    Y = np.zeros((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T
def perm_entropy(x, order=3, delay=1, normalize=False):
    x = np.array(x)
    ran_order = range(order)
    hashmult = np.power(order, ran_order)
    # Embed x and sort the order of permutations
    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind='quicksort')
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    # Return the counts
    _, c = np.unique(hashval, return_counts=True)
    # Use np.true_divide for Python 2 compatibility
    p = np.true_divide(c, c.sum())
    pe = -np.multiply(p, np.log2(p)).sum()
    if normalize:
        pe /= np.log2(factorial(order))
    return pe
def svd_entropy(x, order=3, delay=1, normalize=False):
    x = np.array(x)
    mat = _embed(x, order=order, delay=delay)
    W = np.linalg.svd(mat, compute_uv=False)
    # Normalize the singular values
    W /= sum(W)
    svd_e = -np.multiply(W, np.log2(W)).sum()
    if normalize:
        svd_e /= np.log2(order)
    return svd_e
ENTROPIES = {
    'entropy': lambda x, order, delay: entropy(x),
    'perm_entropy': lambda x, order, delay: perm_entropy(x, order=order, delay=delay),
    # 'spectral_entropy': lambda x, order, delay: spectral_entropy(x),
    'svd_entropy': lambda x, order, delay: svd_entropy(x, order=order, delay=delay),
}

def get_entropies(data, emb_dim: int = 2, tau: int = 3):
    results = {}
    if len(data) > emb_dim * tau:
        for key, value in ENTROPIES.items():
            results[key] = value(data, emb_dim, tau)
    else:
        for key in ENTROPIES.keys():
            results[key] = np.nan
    return results

FUNCTIONS = {
    'mean': np.mean,
    'std': np.std,
    'min': np.min,
    'max': np.max,
    'ptp': np.ptp,
    'sum': np.sum,
    'energy': lambda x: np.sum(x ** 2),
    'skewness': skew,
    'kurtosis': kurtosis,
    'peaks': lambda x: len(find_peaks(x, prominence=0.9)[0]),
    'rms': lambda x: np.sqrt(np.sum(x ** 2) / len(x)),
    'lineintegral': lambda x: np.abs(np.diff(x)).sum(),
    'n_above_mean': lambda x: np.sum(x > np.mean(x)),
    'n_below_mean': lambda x: np.sum(x < np.mean(x)),
    'n_sign_changes': lambda x: np.sum(np.diff(np.sign(x)) != 0),
    'iqr': iqr,
    'iqr_5_95': lambda x: iqr(x, rng=(5, 95)),
    'pct_5': lambda x: np.percentile(x, 5),
    'pct_95': lambda x: np.percentile(x, 95),
}

def get_stats(data, key_suffix: str = None, entropies: bool = True):
    data = np.asarray(data)
    data_nans = np.isnan(data)
    if np.any(data_nans):
        warnings.warn(f'input data contains {np.count_nonzero(data_nans)} NaNs which will be removed')
    data = data[~np.isnan(data)]
    results = {}
    if len(data) > 0:
        for key, value in FUNCTIONS.items():
            results[key] = value(data)
    else:
        for key in FUNCTIONS.keys():
            results[key] = np.nan
    # Update with entropies
    if entropies:
        results.update(get_entropies(data))
    if key_suffix is not None:
        results = {key_suffix+'_'+k: v for k, v in results.items()}
    return results
