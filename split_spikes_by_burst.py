import numpy as np


def split_spikes_by_burst(spike_times, sampling_rate, max_burst_len=4):
    """
    Split spikes into isolated and burst-based subclusters.

    Parameters:
        spike_times     : sorted 1D numpy array of spike times (in samples)
        sampling_rate   : in Hz (e.g., 20000)
        max_burst_len   : maximum number of burst groups (e.g., burst_1 to burst_4+)

    Returns:
        clusters : list of dicts with key 'inds', compatible with analyze_clusters
    """
    isi_thresh = int(0.002 * sampling_rate)  # 2 ms in samples
    n = len(spike_times)

    group_assignments = np.zeros(n, dtype=np.int32)
    current_burst = []
    
    for i in range(n):
        if i == 0:
            next_isi = spike_times[i+1] - spike_times[i]
            if next_isi < isi_thresh:
                current_burst = [i]
            else:
                group_assignments[i] = 0
        elif i == n - 1:
            prev_isi = spike_times[i] - spike_times[i-1]
            if prev_isi < isi_thresh:
                current_burst.append(i)
                for j, idx in enumerate(current_burst):
                    group_idx = min(j + 1, max_burst_len)
                    group_assignments[idx] = group_idx
            else:
                group_assignments[i] = 0
        else:
            prev_isi = spike_times[i] - spike_times[i-1]
            next_isi = spike_times[i+1] - spike_times[i]

            if prev_isi < isi_thresh:
                current_burst.append(i)
                if next_isi >= isi_thresh:
                    for j, idx in enumerate(current_burst):
                        group_idx = min(j + 1, max_burst_len)
                        group_assignments[idx] = group_idx
                    current_burst = []
            elif next_isi < isi_thresh:
                current_burst = [i]
            else:
                group_assignments[i] = 0

    clusters = []
    for k in range(max_burst_len + 1):  # 0 = isolated
        inds = np.where(group_assignments == k)[0]
        clusters.append({'inds': inds})

    return clusters
