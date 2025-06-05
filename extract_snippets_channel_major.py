import numpy as np
import os
from scipy.signal import find_peaks
import multiprocessing as mp
from functools import partial

def extract_snippets_channel_major(dat_path_chmajor, spike_times, selected_channels, window, total_samples, dtype=np.int16):
    """
    Extracts spike-centered snippets from a channel-major binary file.

    Parameters:
    - dat_path_chmajor: Path to channel-major .dat file
    - spike_times: list or array of spike center indices (in samples)
    - selected_channels: list of channel indices to extract
    - window: (pre, post) tuple of samples before/after spike
    - total_samples: total number of timepoints in recording
    - dtype: data type (default: int16)

    Returns:
    - snippets: (n_channels, snip_len, n_spikes) float32 array
    - valid_spike_times: spike times that were successfully extracted
    """
    pre, post = window
    snip_len = post - pre + 1

    spike_times = np.array(spike_times)
    valid_spike_times = spike_times[(spike_times + pre >= 0) & (spike_times + post < total_samples)]

    data = np.memmap(dat_path_chmajor, dtype=dtype, mode='r', shape=(512, total_samples))

    snippets = np.zeros((len(selected_channels), snip_len, len(valid_spike_times)), dtype=np.float32)

    for ch_idx, ch in enumerate(selected_channels):
        trace = data[ch]
        for i, s in enumerate(valid_spike_times):
            t_start = s + pre
            t_end = s + post + 1
            snippets[ch_idx, :, i] = trace[t_start:t_end]

    return snippets, valid_spike_times



def compute_channel_baselines(
    dat_path_chmajor,
    n_channels,
    total_samples,
    dtype=np.int16,
    segment_len=100_000
):
    """
    Computes mean baseline per channel for consecutive non-overlapping segments.

    Parameters:
    - dat_path_chmajor: Path to channel-major .dat file
    - n_channels: Number of channels
    - total_samples: Total number of timepoints
    - dtype: Data type (default: np.int16)
    - segment_len: Number of samples per segment (default: 100000)

    Returns:
    - baselines: (n_channels, n_segments) array of mean values per segment
    """
    n_segments = (total_samples + segment_len - 1) // segment_len
    baselines = np.zeros((n_channels, n_segments), dtype=np.float32)

    data = np.memmap(
        dat_path_chmajor,
        dtype=dtype,
        mode='r',
        shape=(n_channels, total_samples)
    )

    for seg_idx in range(n_segments):
        start = seg_idx * segment_len
        end = min(start + segment_len, total_samples)
        segment = data[:, start:end]
        baselines[:, seg_idx] = segment.mean(axis=1)

    return baselines


def compute_channel_baselines_artifact(
    dat_path_chmajor,
    n_channels,
    total_samples,
    dtype=np.int16,
    segment_len=100_000,
    artifact_threshold=100.0,
    artifact_padding=1000,
    max_artifacts=5
):
    """
    Computes mean baseline per channel with artifact suppression.

    Returns:
    - baselines: (n_channels, n_segments) array of mean values per segment
    - artifact_locs: dict of channel → list of artifact center indices
    """


    n_segments = (total_samples + segment_len - 1) // segment_len
    baselines = np.zeros((n_channels, n_segments), dtype=np.float32)
    artifact_locs = {}

    data = np.memmap(
        dat_path_chmajor,
        dtype=dtype,
        mode='r',
        shape=(n_channels, total_samples)
    )

    global_max = -np.inf
    global_min = np.inf

    for ch in range(n_channels):

        # if ch != 369:
        #     continue
        trace = data[ch, :].astype(np.float32)
        trace_mean = trace.mean()
        trace_std = trace.std()

        # Artifact detection via peak-finding
        threshold = artifact_threshold * trace_std
        peaks_high, _ = find_peaks(trace, height=trace_mean + threshold, distance=100)
        peaks_low, _ = find_peaks(-trace, height=-(trace_mean - threshold), distance=100)
        peaks = np.sort(np.concatenate([peaks_high, peaks_low]))

        valid = ~np.isnan(trace)
        if valid.any():
            ch_max = trace[valid].max()
            ch_min = trace[valid].min()
            global_max = max(global_max, ch_max)
            global_min = min(global_min, ch_min)


        # if ch==369:
        #     print(trace_std)
        #     print(threshold)
        #     print(len(peaks))
        #     print(f"Max amplitude: {trace.max()}, Min amplitude: {trace.min()}, Mean: {trace_mean}, Threshold: {threshold}")


        if len(peaks) <= max_artifacts:
            artifact_locs[ch] = peaks
            for peak in peaks:
                start = max(0, peak - artifact_padding)
                end = min(total_samples, peak + artifact_padding)
                trace[start:end] = np.nan  # blank artifact region with NaNs

        # Compute mean ignoring NaNs
        for seg_idx in range(n_segments):
            start = seg_idx * segment_len
            end = min(start + segment_len, total_samples)
            segment = trace[start:end]
            valid = ~np.isnan(segment)
            if valid.any():
                baselines[ch, seg_idx] = segment[valid].mean()
            else:
                baselines[ch, seg_idx] = trace_mean  # fallback to global mean


    print(f"Global max: {global_max}")
    print(f"Global min: {global_min}")

    return baselines, artifact_locs
