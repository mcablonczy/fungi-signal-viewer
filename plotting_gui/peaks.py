
import numpy as np

def estimate_noise_sigma_mad(y: np.ndarray, min_sigma: float = 1e-12) -> float:
    """
    Robust noise estimate using Median Absolute Deviation (MAD).

    Parameters
    ----------
    y : np.ndarray
        1D array with signal samples (any baseline is fine).
    min_sigma : float
        Minimum sigma to return if the estimate degenerates.

    Returns
    -------
    float
        Estimated noise sigma.
    """
    y = np.asarray(y, dtype=float)

    if y.size == 0:
        return min_sigma

    median = float(np.median(y))
    mad = float(np.median(np.abs(y - median)))

    if mad > 0:
        sigma = 1.4826 * mad  # robust Gaussian-equivalent "std"
    else:
        sigma = float(np.std(y))

    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(min_sigma)

    return sigma
