import numpy as np


def suppress_close_peaks(peaks, scores, refractory_samples):
    """
    Select non-overlapping peaks with greedy symmetric suppression.

    Parameters:
        peaks: np.array of peak indices
        scores: np.array of same shape as mean_score
        refractory_samples: int, half-width of suppression window

    Returns:
        np.array of filtered peak indices
    """
    if len(peaks) == 0:
        return peaks

    sorted_peaks = peaks[np.argsort(scores[peaks])[::-1]]  # descending score
    keep = []
    occupied = np.zeros_like(scores, dtype=bool)

    for p in sorted_peaks:
        if not occupied[p]:
            keep.append(p)
            start = max(p - refractory_samples, 0)
            end = min(p + refractory_samples + 1, len(scores))
            occupied[start:end] = True

    return np.array(sorted(keep))

