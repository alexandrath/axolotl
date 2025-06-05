import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def plot_smoothed_firing_rates(spike_trains, labels=None, sigma_ms=100, dt_ms=10, total_duration_s=None):
    """
    Plot smoothed firing rates using Gaussian kernel.

    Parameters:
        spike_trains      : list of arrays of spike times (in seconds)
                            or a single array for one spike train.
        labels            : optional list of labels for each trace
        sigma_ms          : standard deviation of Gaussian (in ms)
        dt_ms             : sampling resolution for output trace (in ms)
        total_duration_s  : total recording time in seconds

    Returns:
        time_vector       : array of time points (in seconds)
        rate_matrix       : [n_units x n_timepoints] array of smoothed firing rates (Hz)
    """
    if not isinstance(spike_trains, list):
        spike_trains = [spike_trains]
    n_units = len(spike_trains)

    # Determine duration
    if total_duration_s is None:
        total_duration_s = max([np.max(train) if len(train) > 0 else 0 for train in spike_trains]) + 0.01

    dt = dt_ms / 1000
    sigma_samples = sigma_ms / dt_ms
    time_vector = np.arange(0, total_duration_s, dt)
    rate_matrix = np.zeros((n_units, len(time_vector)))

    for i, train in enumerate(spike_trains):
        counts, _ = np.histogram(train, bins=np.append(time_vector, total_duration_s))
        smoothed = gaussian_filter1d(counts / dt, sigma=sigma_samples)
        rate_matrix[i] = smoothed

    # Plot
    plt.figure(figsize=(10, 3))
    for i in range(n_units):
        label = labels[i] if labels and i < len(labels) else f"Unit {i+1}"
        plt.plot(time_vector, rate_matrix[i], label=label)

    plt.xlabel("Time (s)")
    plt.ylabel("Firing rate (Hz)")
    plt.title("Smoothed Firing Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return time_vector, rate_matrix
