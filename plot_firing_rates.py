import numpy as np
import matplotlib.pyplot as plt

def plot_firing_rates(spike_trains, labels=None, bin_size_ms=10, total_duration_s=None):
    """
    Plot binned firing rates for one or more spike trains.

    Parameters:
        spike_trains      : list of arrays of spike times (in seconds)
                            or a single array for one spike train.
        labels            : optional list of labels for each trace
        bin_size_ms       : bin width for rate calculation (milliseconds)
        total_duration_s  : total duration of recording (seconds); 
                            if None, will use max spike time seen

    Returns:
        bin_centers       : array of bin center times (seconds)
        rate_matrix       : array of shape [n_units x n_bins] with rates in Hz
    """
    if not isinstance(spike_trains, list):
        spike_trains = [spike_trains]
    n_units = len(spike_trains)

    # Determine duration
    if total_duration_s is None:
        total_duration_s = max([np.max(train) if len(train) > 0 else 0 for train in spike_trains]) + 0.01

    bin_size_s = bin_size_ms / 1000
    bins = np.arange(0, total_duration_s + bin_size_s, bin_size_s)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    rate_matrix = np.zeros((n_units, len(bin_centers)))

    for i, train in enumerate(spike_trains):
        counts, _ = np.histogram(train, bins=bins)
        rate_matrix[i] = counts / bin_size_s  # convert to Hz

    # Plot
    plt.figure(figsize=(8, 3))
    for i in range(n_units):
        label = labels[i] if labels and i < len(labels) else f"Unit {i+1}"
        plt.plot(bin_centers, rate_matrix[i], label=label)

    plt.xlabel("Time (s)")
    plt.ylabel("Firing rate (Hz)")
    plt.title("Binned Firing Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return bin_centers, rate_matrix
