import numpy as np
import matplotlib.pyplot as plt

def compute_and_plot_xcorr(spike_times_1, spike_times_2, bin_size_ms=1.0, max_lag_ms=100.0):
    """
    Compute and plot cross-correlogram between two spike trains.

    Parameters:
        spike_times_1 : array-like, spike times in seconds
        spike_times_2 : array-like, spike times in seconds
        bin_size_ms   : bin width in milliseconds
        max_lag_ms    : max lag to compute (one-sided), total window = 2*max_lag

    Returns:
        bins_centers  : center time of each bin (in ms)
        counts        : histogram of spike time differences
    """
    spike_times_1 = np.asarray(spike_times_1)
    spike_times_2 = np.asarray(spike_times_2)

    # Convert to milliseconds
    spike_times_1 *= 1000
    spike_times_2 *= 1000

    max_lag = max_lag_ms
    bin_size = bin_size_ms
    bins = np.arange(-max_lag, max_lag + bin_size, bin_size)

    diffs = []
    for t1 in spike_times_1:
        # Only take spikes within window for speed
        mask = (spike_times_2 >= t1 - max_lag) & (spike_times_2 <= t1 + max_lag)
        diffs.extend(spike_times_2[mask] - t1)

    diffs = np.array(diffs)
    counts, edges = np.histogram(diffs, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    plt.figure(figsize=(6, 3))
    plt.bar(centers, counts, width=bin_size, align='center', color='gray', edgecolor='black')
    plt.title("Cross-Correlogram")
    plt.xlabel("Time lag (ms)")
    plt.ylabel("Count")
    plt.xlim(-max_lag, max_lag)
    plt.tight_layout()
    plt.show()

    return centers, counts
