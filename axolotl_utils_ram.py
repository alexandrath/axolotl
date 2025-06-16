import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx
from compare_eis import compare_eis
import scipy.io as sio
import os
from run_multi_gpu_ei_scan import run_multi_gpu_ei_scan
from scipy.signal import argrelextrema
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.signal import correlate
from typing import Union, Tuple
from matplotlib import gridspec
from matplotlib.table import Table
from plot_ei_waveforms import plot_ei_waveforms
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from compute_sta_from_spikes import compute_sta_chunked
from benchmark_c_rgb_generation import RGBFrameGenerator
from matplotlib import rcParams
import itertools
from scipy.stats import trim_mean
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from scipy.stats import gaussian_kde

def silverman_test(data, num_bootstrap=500, random_state=None):
    """
    Silverman's test for unimodality.
    
    Parameters:
        data : array-like, 1D
            The data to test for unimodality.
        num_bootstrap : int
            Number of bootstrap samples.
        random_state : int or None
            Seed for reproducibility.
    
    Returns:
        p_value : float
            P-value for the null hypothesis (unimodal distribution).
    """
    rng = np.random.default_rng(random_state)
    data = np.asarray(data)
    n = len(data)

    # Step 1: Compute h_obs (smallest bandwidth for unimodal KDE)
    def min_unimodal_bandwidth(data):
        # Start with Silverman's rule of thumb
        std_dev = np.std(data, ddof=1)
        iqr = np.subtract(*np.percentile(data, [75, 25]))
        sigma = min(std_dev, iqr / 1.349)
        h0 = 0.9 * sigma * n ** (-1/5)

        # Search for min bandwidth producing unimodal KDE
        hs = np.linspace(h0 * 0.1, h0 * 2, 20)
        for h in hs:
            kde = gaussian_kde(data, bw_method=h / data.std(ddof=1))
            x_grid = np.linspace(data.min(), data.max(), 1000)
            y = kde.evaluate(x_grid)
            modes = np.sum((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))
            if modes <= 1:
                return h
        return hs[-1]

    h_obs = min_unimodal_bandwidth(data)

    # Step 2: Bootstrap
    count = 0
    for _ in range(num_bootstrap):
        resample = rng.choice(data, size=n, replace=True)
        h_boot = min_unimodal_bandwidth(resample)
        if h_boot >= h_obs:
            count += 1

    p_value = count / num_bootstrap
    return p_value

def compute_baselines_int16(raw_data, segment_len=100_000):
    """
    Compute mean baseline per channel over non-overlapping segments.
    No artifact suppression. Operates directly on int16 data.
    
    Input:
    - raw_data: [T, C], int16
    - segment_len: samples per segment (default 100,000)

    Output:
    - baselines: [C, n_segments] float32
    """
    total_samples, n_channels = raw_data.shape
    n_segments = (total_samples + segment_len - 1) // segment_len

    baselines = np.zeros((n_channels, n_segments), dtype=np.float32)

    for seg_idx in range(n_segments):
        start = seg_idx * segment_len
        end = min(start + segment_len, total_samples)
        segment = raw_data[start:end, :]  # [segment_len, C]
        baselines[:, seg_idx] = segment.mean(axis=0)

    return baselines


def compute_baselines_int16_deriv_robust(raw_data, segment_len=100_000, diff_thresh=50, trim_fraction=0.05):
    """
    Compute mean baseline per channel over non-overlapping segments,
    using derivative masking + trimmed mean to suppress spike influence.

    Input:
    - raw_data: [T, C], int16
    - segment_len: samples per segment (default 100,000)
    - diff_thresh: derivative threshold in raw units (default 50 µV/sample)
    - trim_fraction: fraction to trim from both ends (default 5%)

    Output:
    - baselines: [C, n_segments] float32
    """
    total_samples, n_channels = raw_data.shape
    n_segments = (total_samples + segment_len - 1) // segment_len

    baselines = np.zeros((n_channels, n_segments), dtype=np.float32)

    for seg_idx in range(n_segments):
        start = seg_idx * segment_len
        end = min(start + segment_len, total_samples)
        segment = raw_data[start:end, :]  # [S, C]

        if segment.shape[0] < 2:
            baselines[:, seg_idx] = 0  # or np.nan
            continue

        # Compute absolute derivative
        diff_segment = np.abs(np.diff(segment, axis=0))  # [S-1, C]
        # Pad to match original length
        diff_segment = np.vstack([diff_segment, diff_segment[-1]])  

        # Mask: keep only low-derivative points
        flat_mask = diff_segment < diff_thresh  # [S, C]

        # Apply mask and compute trimmed mean per channel
        for c in range(n_channels):
            flat_vals = segment[flat_mask[:, c], c].astype(np.float32)
            if len(flat_vals) > 0:
                baselines[c, seg_idx] = trim_mean(flat_vals, proportiontocut=trim_fraction)
            else:
                baselines[c, seg_idx] = 0  # fallback

    return baselines


def find_dominant_channel_ram(
    raw_data: np.ndarray,
    positions: np.ndarray,                  # [C,2] electrode x-y (µm)
    segment_len: int = 100_000,
    n_segments: int = 10,
    peak_window: int = 30,
    top_k_neg: int = 20,
    top_k_events: int = 5,
    seed: int = 42,
    use_negative_peak: bool = False,
    top_n: int = 10,                        # how many channels to return
    min_spacing: float = 150.0              # min µm separation
) -> tuple[list[int], list[float]]:
    """
    Pick up to `top_n` channels with the largest spike-like amplitudes that are
    at least `min_spacing` µm apart.

    Returns
    -------
    top_channels   : list[int]   indices of selected electrodes
    top_amplitudes : list[float] score for each returned channel
    """

    total_samples, n_channels = raw_data.shape
    rng = np.random.default_rng(seed)

    # deterministic + random segment starts
    first_start = rng.integers(0, min(100_000, total_samples - segment_len))
    other_starts = rng.integers(0, total_samples - segment_len, size=n_segments - 1)
    starts = np.concatenate([[first_start], other_starts])

    channel_amps = [[] for _ in range(n_channels)]

    for start in starts:
        seg = raw_data[start:start + segment_len, :]
        for ch in range(n_channels):
            trace = seg[:, ch].astype(np.float32)
            trace -= trace.mean()

            neg_peaks, _ = find_peaks(-trace, distance=20)
            if neg_peaks.size == 0:
                continue

            strongest = neg_peaks[np.argsort(trace[neg_peaks])[:top_k_neg]]

            for p in strongest:
                valley = trace[p]
                if use_negative_peak:
                    amp = -valley                                   # bigger = stronger
                else:
                    w0, w1 = max(0, p - peak_window), min(segment_len, p + peak_window + 1)
                    local_max = trace[w0:w1].max()
                    amp = local_max - valley                        # peak-to-peak
                channel_amps[ch].append(amp)

    # mean of top-k events per channel
    mean_amp = np.zeros(n_channels, dtype=np.float32)
    for ch in range(n_channels):
        amps = np.asarray(channel_amps[ch], dtype=np.float32)
        if amps.size:
            mean_amp[ch] = amps[np.argsort(amps)][-top_k_events:].mean()

    # ----------------------------------------------------------------
    # spacing-aware greedy selection
    # ----------------------------------------------------------------
    sorted_idx = np.argsort(mean_amp)[::-1]       # high → low
    selected   = []
    for idx in sorted_idx:
        if len(selected) >= top_n:
            break
        if all(np.linalg.norm(positions[idx] - positions[s]) >= min_spacing
               for s in selected):
            selected.append(idx)

    # if not enough well-spaced channels, pad with next best
    for idx in sorted_idx:
        if len(selected) == top_n:
            break
        if idx not in selected:
            selected.append(idx)

    top_channels   = selected
    top_amplitudes = mean_amp[top_channels].tolist()

    return top_channels, top_amplitudes






def estimate_spike_threshold_ram(
    raw_data: np.ndarray,
    ref_channel: int,
    window: int = 30,
    total_samples_to_read: int = 10_000_000,
    refractory: int = 30,
    top_n: int = 100,
    threshold_scale: float = 0.5
) -> tuple[float, np.ndarray]:
    """
    Estimate spike threshold and detect spike times using adaptive threshold crossing logic.

    Parameters:
        raw_data: [T, C] int16 array
        ref_channel: channel to analyze
        window: unused now but kept for API consistency
        total_samples_to_read: length of trace to analyze
        refractory: minimum spacing between spikes
        top_n: used to estimate threshold from largest events
        threshold_scale: scaling for final detection threshold

    Returns:
        threshold: estimated threshold
        spike_times: array of detected spike centers (global minima within threshold-crossing windows)
    """
    trace = raw_data[:total_samples_to_read, ref_channel].astype(np.float32)
    trace -= np.mean(trace)

    segment_len = 20000  # 1s at 20kHz
    n_segments = len(trace) // segment_len

    trace_filtered = np.empty_like(trace)

    for i in range(n_segments):
        start = i * segment_len
        end = start + segment_len
        segment = trace[start:end]
        segment_mean = np.mean(segment)
        trace_filtered[start:end] = segment - segment_mean


    # --- Estimate threshold from top-N negative peaks ---
    neg_peaks, _ = find_peaks(-trace_filtered, distance=2 * refractory)
    peak_vals = trace_filtered[neg_peaks]
    if len(peak_vals) >= top_n:
        top_idx = np.argsort(np.abs(peak_vals))[-top_n:]
    else:
        top_idx = np.argsort(np.abs(peak_vals))
    threshold = threshold_scale * np.mean(np.abs(peak_vals[top_idx]))
    # threshold = 50

    # --- Detect threshold-crossing windows ---
    above = trace_filtered > -threshold
    crossings_down = np.where(above[:-1] & ~above[1:])[0] + 1
    crossings_up   = np.where(~above[:-1] & above[1:])[0] + 1

    # --- Pair crossings into spike windows ---
    entry_ptr, exit_ptr = 0, 0
    spike_windows = []

    while entry_ptr < len(crossings_down) and exit_ptr < len(crossings_up):
        entry = crossings_down[entry_ptr]
        exit = crossings_up[exit_ptr]

        if exit <= entry:
            exit_ptr += 1
            continue

        spike_windows.append((entry, exit))
        entry_ptr += 1
        exit_ptr += 1

    # --- Find minima in each window ---
    spike_times = []
    last_spike = -np.inf

    for entry, exit in spike_windows:
        if exit - entry < 1:
            continue

        min_idx = np.argmin(trace_filtered[entry:exit]) + entry
        if min_idx - last_spike > refractory:
            spike_times.append(min_idx)
            last_spike = min_idx

    spike_times = np.array(spike_times)

    # --- Enforce spike count cap ---
    max_spikes = 50000

    if len(spike_times) > max_spikes:
        spike_amps = -trace_filtered[spike_times]  # negate because spikes are negative
        top_idx = np.argsort(spike_amps)[-max_spikes:]  # top N amplitudes
        spike_times = spike_times[np.sort(top_idx)]  # preserve temporal order


        # TEMP: check values at known missed spikes
#     missed_spike_times = np.array([
#             42099,  1008006,  1305723,  6505023,  9755242, 12277345, 12857675, 13192348,
#             13439089, 14105963, 15881850, 16335736, 16385159, 16436369, 16562586, 17206720,
#             17224743, 18191108, 18294350, 19138550, 20034161, 21572611, 22001075, 22542134,
#             23148769, 23216961, 23356572, 23485098, 24394617, 24936663, 25230985, 25337944,
#             25812998, 26706395, 27558521, 28022213, 28369137, 28469633, 28486336, 29716292,
#             30211815, 32174362, 32316809, 32974069, 34168038, 34413741, 34426669, 34462695,
#             34489819, 34903635, 35249228
#     ])
#  # fill in the 173 sample indices

#     # Make sure you don't exceed the trace bounds
#     import matplotlib.pyplot as plt

#     for i, center in enumerate(missed_spike_times[:3]):
#         win_start = max(center - 60, 0)
#         win_end = min(center + 61, len(trace))
#         t_range = np.arange(win_start, win_end)

#         local_trace = trace[win_start:win_end]

#         # Find event_indices within this window
#         local_events = spike_times[(spike_times >= win_start) & (spike_times < win_end)]
#         local_event_rel = local_events - win_start  # relative to window

#         plt.figure(figsize=(10, 3))
#         plt.plot(t_range - center, local_trace, label='Filtered trace', color='black')
#         plt.axvline(0, color='red', linestyle='--', label='Missing spike center')
#         plt.plot(local_event_rel - 60, local_trace[local_event_rel], 'g*', markersize=10, label='Detected peaks')
#         plt.title(f"Missed spike #{i+1} at sample {center}")
#         plt.xlabel("Time (samples relative to center)")
#         plt.ylabel("Amplitude")
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()
#         plt.show()



    return threshold, spike_times




def extract_snippets_ram(
    raw_data: np.ndarray,
    spike_times: np.ndarray,
    window: tuple[int, int],
    selected_channels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract snippets from time-major raw_data around spike_times.

    Parameters:
        raw_data: ndarray of shape [T, C] (int16)
        spike_times: array of spike indices (int)
        window: tuple (pre, post) in samples
        selected_channels: array of channel indices to include

    Returns:
        snippets: [n_channels, n_samples, n_spikes] float32
        valid_spike_times: spike_times for which the snippet was valid
    """
    pre, post = window
    total_samples, _ = raw_data.shape

    valid_times = []
    snippets = []

    for t in spike_times:
        if t + pre < 0 or t + post >= total_samples:
            continue
        snippet = raw_data[t + pre : t + post + 1, selected_channels]  # [T, C]
        snippets.append(snippet.T.astype(np.float32))  # [C, T]
        valid_times.append(t)

    snippets = np.stack(snippets, axis=2)  # [C, T, N]
    valid_times = np.array(valid_times)

    return snippets, valid_times



def sub_sample_align_ei(ei_template, ei_candidate, ref_channel, upsample=10, max_shift=2.0):
    """
    Align ei_candidate to ei_template using sub-sample alignment on the reference channel.

    Parameters:
        ei_template : np.ndarray [C x T]
        ei_candidate : np.ndarray [C x T]
        ref_channel : int — channel to use for alignment
        upsample : int — interpolation factor (e.g., 10 for 0.1 sample resolution)
        max_shift : float — maximum shift allowed (in samples)

    Returns:
        aligned_candidate : np.ndarray [C x T] — shifted ei_candidate
    """
    C, T = ei_template.shape
    assert ei_candidate.shape == (C, T), "Shape mismatch"

    t = np.arange(T)
    t_interp = np.linspace(0, T - 1, T * upsample)

    # Interpolate both waveforms
    interp_template = interp1d(t, ei_template[ref_channel], kind='cubic', bounds_error=False, fill_value=0.0)
    interp_candidate = interp1d(t, ei_candidate[ref_channel], kind='cubic', bounds_error=False, fill_value=0.0)

    template_highres = interp_template(t_interp)
    candidate_highres = interp_candidate(t_interp)

    # Cross-correlation to find best fractional lag
    full_corr = correlate(candidate_highres, template_highres, mode='full')
    lags = np.arange(-len(candidate_highres) + 1, len(template_highres))
    center = len(full_corr) // 2
    lag_window = int(max_shift * upsample)
    search_range = slice(center - lag_window, center + lag_window + 1)

    best_lag_index = np.argmax(full_corr[search_range])
    fractional_shift = lags[search_range][best_lag_index] / upsample

    # Apply same shift to all channels
    aligned_candidate = np.zeros_like(ei_candidate)
    for ch in range(C):
        interp_func = interp1d(t, ei_candidate[ch], kind='cubic', bounds_error=False, fill_value=0.0)
        shifted_time = t + fractional_shift
        aligned_candidate[ch] = interp_func(shifted_time)

    return aligned_candidate, fractional_shift


def compare_ei_subtraction(ei_a, ei_b, max_lag=3, p2p_thresh=30.0):
    """
    Compare two EIs using subtraction and cosine similarity, with variance-based residual thresholding.

    Parameters:
        ei_a : np.ndarray
            Reference EI, shape (n_channels, n_samples)
        ei_b : np.ndarray
            Test EI, shape (n_channels, n_samples)
        max_lag : int
            Max lag (in samples) for alignment
        p2p_thresh : float
            Minimum peak-to-peak threshold to consider a channel for comparison
        scale_factor : float
            Multiplier for variance-based residual threshold

    Returns:
        result : dict with keys:
            - mean_residual
            - max_abs_residual
            - good_channels
            - per_channel_residuals
            - per_channel_cosine_sim
            - p2p_a
    """
    C, T = ei_a.shape
    assert ei_b.shape == (C, T), "EIs must have same shape"

    # Identify dominant channel from A
    ref_chan = np.argmax(np.max(np.abs(ei_a), axis=1))

    # Align B to A using sub-sample alignment
    aligned_b, fractional_shift = sub_sample_align_ei(ei_template=ei_a, ei_candidate=ei_b, ref_channel=ref_chan, upsample=10, max_shift=max_lag)

    # Select meaningful channels based on P2P of A
    p2p_a = ei_a.max(axis=1) - ei_a.min(axis=1)
    good_channels = np.where(p2p_a > p2p_thresh)[0]

    per_channel_residuals = []
    per_channel_cosine_sim = []
    all_residuals = []

    for ch in good_channels:
        a = ei_a[ch]
        b = aligned_b[ch]

        mask = np.abs(a) > 0.1 * np.max(np.abs(a))
        if not np.any(mask):
            continue

        a_masked = a[mask]
        b_masked = b[mask]

        residual = b_masked - a_masked
        per_channel_residuals.append(np.mean(residual))
        all_residuals.extend(residual)

        dot = np.dot(a_masked, b_masked)
        norm_product = np.linalg.norm(a_masked) * np.linalg.norm(b_masked) + 1e-8
        cosine_sim = dot / norm_product
        per_channel_cosine_sim.append(cosine_sim)

    all_residuals = np.array(all_residuals)
    mean_residual = np.mean(all_residuals)
    max_abs_residual = np.max(np.abs(all_residuals))

    import matplotlib.pyplot as plt


    # plt.figure(figsize=(5,5))
    # plt.plot(ei_a[ref_chan,:], color='black', alpha=0.8, linewidth=0.5)
    # plt.plot(ei_b[ref_chan,:], color='blue', alpha=0.8, linewidth=0.5)
    # plt.plot(aligned_b[ref_chan,:], color='red', alpha=0.8, linewidth=0.5)
    # plt.show()

    return {
        'mean_residual': mean_residual,
        'max_abs_residual': max_abs_residual,
        'good_channels': good_channels,
        'per_channel_residuals': per_channel_residuals,
        'per_channel_cosine_sim': per_channel_cosine_sim,
        'fractional_shift': fractional_shift,
        'p2p_a': p2p_a
    }

import numpy as np

def merge_similar_clusters(snips, labels, max_lag=3, p2p_thresh=30.0, amp_thresh=-20, cos_thresh=0.8):
    cluster_ids = sorted(np.unique(labels))
    cluster_spike_indices = {k: np.where(labels == k)[0] for k in cluster_ids}
    cluster_eis = []
    cluster_vars = []
    for k in cluster_ids:
        inds = cluster_spike_indices[k]
        ei_k = np.mean(snips[:, :, inds], axis=2)
        #ei_k -= ei_k[:, :5].mean(axis=1, keepdims=True)
        cluster_eis.append(ei_k)

        # --- Variance at peak +/-1 sample ---
        peak_idxs = np.argmin(ei_k, axis=1)  # (n_channels,)
        n_channels, n_samples = ei_k.shape
        channel_var = np.zeros(n_channels)
        for ch in range(n_channels):
            idx = peak_idxs[ch]
            if 1 <= idx < n_samples - 1:
                local_waveform = snips[ch, idx-1:idx+2, inds]  # shape (3, n_spikes)
                channel_var[ch] = np.var(local_waveform)
            else:
                channel_var[ch] = 0.0

        cluster_vars.append(channel_var)


    n_clusters = len(cluster_ids)
    sim = np.eye(n_clusters)
    n_bad_channels = np.zeros((n_clusters, n_clusters), dtype=int)

    # --- Compute similarity matrix and bad channels ---
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                ei_a = cluster_eis[i]
                ei_b = cluster_eis[j]
                var_a = cluster_vars[i]
                res_ab = compare_ei_subtraction(ei_a, ei_b, max_lag=max_lag, p2p_thresh=p2p_thresh)
                res = np.array(res_ab['per_channel_residuals'])
                p2p_a = res_ab['p2p_a']
                good_channels = res_ab['good_channels']
                cos_sim = np.array(res_ab['per_channel_cosine_sim'])
                ch_weights = np.array(p2p_a[good_channels])

                snr_score = 1 / (1 + var_a[good_channels] / (p2p_a[good_channels] ** 2 + 1e-3))
                # snr = p2p_a[good_channels] ** 2 / (var_a[good_channels] + 1e-3)
                snr_mask = snr_score > 0.5  # or whatever cutoff you choose

                res_subset = res[snr_mask]

                cos_sim_masked = cos_sim[snr_mask]
                ch_weights_masked = ch_weights[snr_mask]

                if len(cos_sim_masked) > 0:
                    weighted_cos_sim = np.average(cos_sim_masked, weights=ch_weights_masked)
                else:
                    weighted_cos_sim = 0.0  # Or np.nan, or skip this pair entirely
                #adaptive_thresh = amp_thresh * (var_a[good_channels]/ (p2p_a[good_channels] + 1e-3))
                # res_subset = res[good_channels]
                #res_subset = res
                neg_inds = np.where(res_subset < amp_thresh)[0]
                # neg_inds = np.where(res_subset < adaptive_thresh)[0]

                # neg_inds = np.where(res < amp_thresh)[0]
                sim[i, j] = weighted_cos_sim
                n_bad_channels[i, j] = len(neg_inds)





                # noise_score = var_a[good_channels] / (p2p_a[good_channels] + 1e-3)
                # if weighted_cos_sim <0.92:
                #     fig, axs = plt.subplots(5, 1, figsize=(16, 8), sharex=True)

                #     axs[0].plot(good_channels, snr_score, marker='o')
                #     axs[0].set_ylabel("Normalized Variance (SNR)")
                #     axs[0].set_title("Channel-wise Variance / P2P")

                #     axs[1].plot(good_channels, res, marker='o')
                #     axs[1].set_ylabel("Residuals (B - A)")
                #     axs[1].set_title("Channel-wise Mean Residuals")

                #     axs[2].plot(good_channels[snr_mask], cos_sim_masked * ch_weights_masked, marker='o')
                #     axs[2].set_ylabel("Cosine Similarity")
                #     axs[2].set_title(f"Weighted Cosine Similarity, mean = {weighted_cos_sim}")
                #     axs[2].set_xlabel("Channel ID")

                #     axs[3].plot(good_channels, res_ab['per_channel_cosine_sim'], marker='o')
                #     axs[3].set_ylabel("Cosine Similarity")
                #     axs[3].set_title(f"Unweighted Cosine Similarity, mean {np.mean(res_ab['per_channel_cosine_sim'])}")
                #     axs[3].set_xlabel("Channel ID")

                #     axs[4].plot(good_channels[snr_mask], ch_weights_masked, marker='o')
                #     axs[4].set_ylabel("weights")
                #     axs[4].set_title("Channel weights")
                #     axs[4].set_xlabel("Channel ID")

                #     for ax in axs:
                #         ax.grid(True)
                #         ax.set_xticks(good_channels)
                #         ax.set_xticklabels(good_channels, rotation=45)

                #     plt.tight_layout()
                #     plt.show()

    # --- Merge clusters using precomputed similarities ---
    cluster_sizes = {i: len(cluster_spike_indices[i]) for i in cluster_ids}
    sorted_cluster_ids = sorted(cluster_ids, key=lambda i: cluster_sizes[i], reverse=True)

    assigned = set()
    merged_clusters = []

    id_to_index = {cid: idx for idx, cid in enumerate(cluster_ids)}

    for i in sorted_cluster_ids:
        if i in assigned:
            continue

        base_group = [i]
        assigned.add(i)

        # Try to grow the group transitively via the star logic
        added = True
        while added:
            added = False
            for j in sorted_cluster_ids:
                if j in assigned:
                    continue

                # Check similarity to ANY already-accepted cluster in the group
                for existing in base_group:
                    sim_ij = sim[existing, j]
                    n_bad = n_bad_channels[existing, j]

                    accept = False
                    if sim_ij >= 0.95 and n_bad <= 7:
                        accept = True
                    elif sim_ij >= 0.80 and n_bad <= 5:
                        accept = True
                    elif sim_ij >= 0.70 and n_bad == 3:
                        accept = True
                    elif sim_ij >= cos_thresh and n_bad == 1:
                        accept = True

                    if accept:
                        base_group.append(j)
                        assigned.add(j)
                        added = True
                        break  # Stop checking other members once one match is found

        # Merge all spikes from group
        merged_spikes = np.concatenate([cluster_spike_indices[k] for k in base_group])
        merged_clusters.append(np.sort(merged_spikes))

    # for i in sorted_cluster_ids:
    #     if i in assigned:
    #         continue

    #     i_idx = id_to_index[i]
    #     base_inds = cluster_spike_indices[i]
    #     group = list(base_inds)
    #     assigned.add(i)

    #     for j in sorted_cluster_ids:
    #         if j in assigned or j == i:
    #             continue

    #         j_idx = id_to_index[j]
    #         sim_ij = sim[i_idx, j_idx]
    #         n_bad = n_bad_channels[i_idx, j_idx]

    #         if sim_ij >= 0.95 and n_bad <= 4:
    #             accept = True
    #         elif sim_ij >= 0.90 and n_bad <= 2:
    #             accept = True
    #         elif sim_ij >= cos_thresh and n_bad == 0:
    #             accept = True
    #         else:
    #             accept = False

    #         if accept:
    #             group.extend(cluster_spike_indices[j])
    #             assigned.add(j)

    #     merged_clusters.append(np.sort(np.array(group)))

    return merged_clusters, sim, n_bad_channels





def cluster_spike_waveforms(
    snips: np.ndarray,
    ei: np.ndarray,
    k_start: int = 3,
    p2p_threshold: float = 15,
    min_chan: int = 30,
    max_chan: int = 80,
    sim_threshold: float = 0.9,
    merge: bool = True,
    return_debug: bool = False,   # ← new
    plot_diagnostic: bool = False
) -> Union[list[dict], Tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]]:
    """
    Cluster spike waveforms based on selected EI channels and merge using EI similarity.

    Returns:
        List of cluster dicts with 'inds', 'ei', and 'channels' keys.
    """
    ei_p2p = ei.max(axis=1) - ei.min(axis=1)
    selected_channels = np.where(ei_p2p > p2p_threshold)[0]
    if len(selected_channels) > max_chan:
        selected_channels = np.argsort(ei_p2p)[-max_chan:]
    elif len(selected_channels) < min_chan:
        selected_channels = np.argsort(ei_p2p)[-min_chan:]
    selected_channels = np.sort(selected_channels)

    main_chan = np.argmax(ei_p2p)

    snips_sel = snips[selected_channels, :, :]
    C, T, N = snips_sel.shape
    # snips_centered = snips_sel - snips_sel.mean(axis=1, keepdims=True) # commented out because baseline subtraction happens before this function and I don't like per-spike offsets
    snips_centered = snips_sel.copy()
    snips_flat = snips_centered.transpose(2, 0, 1).reshape(N, -1)

    # --- Focused PCA on main channel in spike zone ---
    spike_zone = slice(10, 40)  # or whatever range captures the peak

    main_snips = snips[main_chan, spike_zone, :].T  # shape: (N, T_spike)
    main_pc1 = PCA(n_components=1).fit_transform(main_snips).flatten()

    # --- Append to flattened snippets before clustering ---
    pcs = PCA(n_components=10).fit_transform(snips_flat)

    pcs_z = StandardScaler().fit_transform(pcs)
    main_pc1_z = StandardScaler().fit_transform(main_pc1[:, None]).flatten()

    pcs_aug = np.hstack((pcs_z, main_pc1_z[:, None]))
    # pcs_aug = np.hstack((pcs, main_pc1[:, None]))

    # --- Clustering
    kmeans = KMeans(n_clusters=k_start, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pcs_aug)

    labels_updated = labels.copy()
    next_label = labels_updated.max() + 1

    # Initialize list of clusters to check
    to_check = list(np.unique(labels_updated))

    while to_check:
        cl = to_check.pop(0)
        mask = labels_updated == cl
        pc_vals = pcs_aug[mask, :2]

        if len(pc_vals) < 20:
            continue

        split_result = check_2d_gap_peaks_valley(pc_vals, angle_step=10, min_valley_frac=0.25)

        if split_result is not None:
            group1_mask, group2_mask = split_result
            cluster_indices = np.where(mask)[0]

            n1 = np.sum(group1_mask)
            n2 = np.sum(group2_mask)

            if n1 < 20 or n2 < 20:
                print(f"  Cluster {cl} split discarded: would create small cluster (group1={n1}, group2={n2})")
                continue

            # Assign new label to group2 (or vice versa)
            labels_updated[cluster_indices[group2_mask]] = next_label
            print(f"  Cluster {cl} split into {cl} and {next_label}")

            # Add both parts back to check if large enough
            if np.sum(group1_mask) >= 20:
                to_check.append(cl)
            if np.sum(group2_mask) >= 20:
                to_check.append(next_label)

            next_label += 1

    labels = labels_updated




    # pca = PCA(n_components=10)
    # pcs = pca.fit_transform(snips_flat)

    # kmeans = KMeans(n_clusters=k_start, n_init=10, random_state=42)
    # labels = kmeans.fit_predict(pcs)

    if plot_diagnostic == True:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 5))
        for i in np.unique(labels):
            cluster_points = pcs[labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i+1}", s=10)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA on spike waveforms")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    merged_clusters, sim, n_bad_channels = merge_similar_clusters(snips, labels, max_lag=3, p2p_thresh=30.0, amp_thresh=-20, cos_thresh=0.65)

    # cluster_spike_indices = {k: np.where(labels == k)[0] for k in np.unique(labels)}

    # cluster_eis = []
    # cluster_ids = sorted(cluster_spike_indices.keys())
    # for k in cluster_ids:
    #     inds = cluster_spike_indices[k]
    #     ei_k = np.mean(snips[:, :, inds], axis=2)
    #     ei_k -= ei_k[:, :5].mean(axis=1, keepdims=True)
    #     cluster_eis.append(ei_k)

    # sim = compare_eis(cluster_eis)
    # if plot_diagnostic == True:
    #     print("Similarity before merge:\n")
    #     print(sim)

    # if not merge:
    #     output = []
    #     for k in cluster_ids:
    #         inds = cluster_spike_indices[k]
    #         ei_cluster = np.mean(snips[:, :, inds], axis=2)
    #         ei_cluster -= ei_cluster[:, :5].mean(axis=1, keepdims=True)
    #         output.append({
    #             'inds': inds,
    #             'ei': ei_cluster,
    #             'channels': selected_channels
    #         })
    #     if return_debug:
    #         return output, pcs, labels, sim
    #     else:
    #         return output


    # G = nx.Graph()
    # G.add_nodes_from(range(len(cluster_ids)))
    # for i in range(len(cluster_ids)):
    #     for j in range(i + 1, len(cluster_ids)):
    #         if sim[i, j] >= sim_threshold:
    #             G.add_edge(i, j)

    # merged_groups = list(nx.connected_components(G))
    # merged_clusters = []

    # for group in merged_groups:
    #     group = sorted(list(group))
    #     all_inds = np.concatenate([cluster_spike_indices[cluster_ids[i]] for i in group])
    #     merged_clusters.append(np.sort(all_inds))

    output = []
    for inds in merged_clusters:
        ei_cluster = np.mean(snips[:, :, inds], axis=2)
        # ei_cluster -= ei_cluster[:, :5].mean(axis=1, keepdims=True)
        output.append({
            'inds': inds,
            'ei': ei_cluster,
            'channels': selected_channels
        })

    if return_debug:

        cluster_spike_indices = {k: np.where(labels == k)[0] for k in np.unique(labels)}
        cluster_eis = []
        cluster_ids = sorted(cluster_spike_indices.keys())
        for k in cluster_ids:
            inds = cluster_spike_indices[k]
            ei_k = np.mean(snips[:, :, inds], axis=2)
            # ei_k -= ei_k[:, :5].mean(axis=1, keepdims=True)
            cluster_eis.append(ei_k)

        cluster_to_merged_group = {}

        for orig_id, orig_inds in cluster_spike_indices.items():
            orig_set = set(orig_inds)

            for group_idx, merged_inds in enumerate(merged_clusters):
                merged_set = set(merged_inds)

                if orig_set.issubset(merged_set):
                    cluster_to_merged_group[orig_id] = group_idx
                    break


        return output, pcs, labels, sim, n_bad_channels, cluster_eis, cluster_to_merged_group
    else:
        return output


def select_cluster_with_largest_waveform(
    clusters: list[dict],
    ref_channel: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Select the cluster with the largest peak-to-peak EI amplitude on the reference channel.

    Parameters:
        clusters: List of cluster dictionaries with 'ei', 'inds', 'channels'
        ref_channel: Channel to evaluate for dominance

    Returns:
        ei: [channels x timepoints] EI of selected cluster
        spikes: [N] array of spike indices from selected cluster
        selected_channels: [K] array of channels used for clustering
    """
    amplitudes = []
    for cl in clusters:
        ei = cl['ei']
        p2p = ei[ref_channel, :].max() - ei[ref_channel, :].min()
        amplitudes.append(p2p)
    #print(amplitudes)

    best_idx = int(np.argmax(amplitudes))
    best = clusters[best_idx]
    return best['ei'], best['inds'], best['channels'], best_idx


import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import norm
import scipy.io as sio
from typing import Union, Tuple

def ei_pursuit_ram(
    raw_data: np.ndarray,
    spikes: np.ndarray,
    ei_template: np.ndarray,
    save_prefix: str = '/tmp/ei_scan_unit0',
    alignment_offset: int = 20,
    fit_percentile: float = 40,
    sigma_thresh: float = 5.0,
    return_debug: bool = False
) -> Union[np.ndarray, Tuple]:
    """
    RAM-resident version of EI pursuit using pre-loaded time-major data and EI template.

    Parameters:
        raw_data: [T, C] int16 full data in RAM
        spikes: Initial spike times (absolute sample indices)
        ei_template: [channels x timepoints] template
        save_prefix: Where to save temp .mat file for GPU
        alignment_offset: Used to re-align accepted spike times
        fit_percentile: Lower-tail fitting percentile
        sigma_thresh: How strict to be when setting threshold
        return_debug: Whether to return intermediate results

    Returns:
        final_spike_times or full debug tuple
    """
    from run_multi_gpu_ei_scan import run_multi_gpu_ei_scan  # uses raw file but loads template

    # Save EI template to MAT file for GPU code
    ei_template_path = f"{save_prefix}_template.mat"
    sio.savemat(ei_template_path, {'ei_template': ei_template.astype(np.float32)})

    # Run multi-GPU matching
    mean_score, max_score, valid_score, selected_channels, _ = run_multi_gpu_ei_scan(
        ei_mat_path=ei_template_path,
        dat_path=None,  # Not used; raw_data is in RAM
        total_samples=raw_data.shape[0],
        save_prefix=save_prefix,
        dtype='int16',  # Unused if you're bypassing dat_path
        block_size=None,
        baseline_start_sample=0,
        channel_major=False,  # We are now time-major
        raw_data_override=raw_data  # This requires modification inside run_multi_gpu_ei_scan
    )

    #print(selected_channels)
    # Adjust for alignment offset
    adjusted_selected_inds = spikes - alignment_offset
    adjusted_selected_inds = adjusted_selected_inds[
        (adjusted_selected_inds >= 0) & (adjusted_selected_inds < len(mean_score))
    ]

    def fit_threshold(scores):
        cutoff = np.percentile(scores, fit_percentile, method='nearest')
        left_tail = scores[scores <= cutoff]
        mu, sigma = norm.fit(left_tail)
        return mu - sigma_thresh * sigma

    mean_scores = mean_score[adjusted_selected_inds]
    valid_scores = valid_score[adjusted_selected_inds]

    clean_mean = mean_scores[~np.isnan(mean_scores)]
    clean_valid = valid_scores[~np.isnan(valid_scores)]

    mean_threshold = fit_threshold(clean_mean)
    valid_threshold = fit_threshold(clean_valid)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.hist(mean_score, bins=200, alpha=0.5, label='All mean scores', color='gray')
    plt.hist(mean_scores, bins=200, alpha=0.5, label='KS spike scores', color='red')
    plt.axvline(mean_threshold, color='red', linestyle='--', label=f"Mean threshold = {mean_threshold:.2f}")
    plt.xlabel("Mean EI Match Score")
    plt.ylabel("Count")
    plt.title("Mean EI Scores: Global vs. KS-aligned")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(10, 5))
    plt.hist(valid_score, bins=200, alpha=0.5, label='All mean scores', color='gray')
    plt.hist(valid_scores, bins=200, alpha=0.5, label='KS spike scores', color='red')
    plt.axvline(valid_threshold, color='red', linestyle='--', label=f"Mean threshold = {mean_threshold:.2f}")
    plt.xlabel("Mean EI Match Score")
    plt.ylabel("Count")
    plt.title("Mean EI Scores: Global vs. KS-aligned")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    peaks = argrelextrema(mean_score, np.greater_equal, order=1)[0]
    valid_inds = peaks[
        (mean_score[peaks] > mean_threshold) &
        (valid_score[peaks] > valid_threshold)
    ]

    # Cap accepted events to 1.2× original spike count
    if len(valid_inds) > len(spikes) * 1.2:
        limit = int(len(spikes) * 1.2)
        top_inds = np.argsort(mean_score[valid_inds])[::-1][:limit]
        valid_inds = valid_inds[top_inds]

    final_spike_times = valid_inds + alignment_offset

    if return_debug:
        return (
            final_spike_times,
            mean_score,
            valid_score,
            mean_scores,
            valid_scores,
            mean_threshold,
            valid_threshold
        )
    else:
        return final_spike_times


def estimate_lags_by_xcorr_ram(snippets: np.ndarray, peak_channel_idx: int, window: tuple = (-5, 10), max_lag: int = 3) -> np.ndarray:
    """
    Estimate lag for each spike by cross-correlating with the mean waveform of the peak channel.

    Parameters:
        snippets: np.ndarray
            Spike snippets of shape (N, C, T)
        peak_channel_idx: int
            Index of the peak (reference) channel
        window: tuple
            Relative window around the peak to consider for alignment
        max_lag: int
            Maximum lag to consider for alignment

    Returns:
        np.ndarray
            Array of integer lags for each spike
    """
    N, C, T = snippets.shape
    waveform = snippets[:, peak_channel_idx, :].mean(axis=0)
    peak_idx = np.argmax(np.abs(waveform))


    win_start = peak_idx + window[0]
    win_end = peak_idx + window[1]

    if win_start < 0 or win_end > T:

        raise ValueError(f"Window around peak ({win_start}:{win_end}) is out of bounds for waveform length {T}")


    ei_win = waveform[win_start:win_end]

    lags = np.zeros(N, dtype=int)
    for i in range(N):
        snip = snippets[i, peak_channel_idx, win_start - max_lag : win_end + max_lag].copy()
        if snip.shape[0] < ei_win.shape[0] + 2 * max_lag:
            lags[i] = 0  # skip if snippet is too short
            continue
        corr = correlate(snip, ei_win, mode='valid')
        lag = np.argmax(corr) - max_lag
        lags[i] = lag

    return lags




def extract_snippets_single_channel(dat_path, spike_times, ref_channel,
                                    window=(-20, 60), n_channels=512, dtype='int16'):
    """
    Extract raw data snippets from a time-major .dat file for a single channel.

    Parameters:
        dat_path: Path to the .dat file (time-major format)
        spike_times: array of spike center times (in samples)
        ref_channel: which channel to extract
        window: (pre, post) time window around each spike
        n_channels: number of electrodes in the recording
        dtype: data type in file (e.g., 'int16')

    Returns:
        snips: [snippet_len x num_spikes] float32 array
    """
    pre, post = window
    snip_len = post - pre + 1
    spike_count = len(spike_times)

    snips = np.zeros((snip_len, spike_count), dtype=np.float32)
    bytes_per_sample = np.dtype(dtype).itemsize

    with open(dat_path, 'rb') as f:
        f.seek(0, 2)
        total_samples = f.tell() // (n_channels * bytes_per_sample)

        for i, center in enumerate(spike_times):
            t_start = center + pre
            t_end = center + post
            if t_start < 0 or t_end >= total_samples:
                continue  # skip invalid spikes

            offset = (t_start * n_channels + ref_channel) * bytes_per_sample
            f.seek(offset, 0)

            # Read 1 channel every n_channels steps
            raw = np.fromfile(f, dtype=dtype, count=snip_len * n_channels)[::n_channels]
            snips[:, i] = raw.astype(np.float32)

    return snips[np.newaxis, :, :]



def select_cluster_by_ei_similarity_ram(
    clusters: list[dict],
    reference_ei: np.ndarray,
    similarity_threshold: float = 0.9
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Merge clusters based on EI similarity and select the one most similar to a reference EI.

    Parameters:
        clusters: List of cluster dicts with 'inds', 'ei', 'channels'
        reference_ei: EI to compare each cluster's EI against
        similarity_threshold: Threshold for merging based on pairwise EI similarity

    Returns:
        ei: Final EI of selected cluster
        spikes: Final spike times (indices into full spike list)
        selected_channels: Channels from selected cluster
    """
    cluster_eis = [cl['ei'] for cl in clusters]
    cluster_ids = list(range(len(clusters)))

    sim = compare_eis(cluster_eis)

    G = nx.Graph()
    G.add_nodes_from(cluster_ids)

    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            if sim[i, j] >= similarity_threshold:
                G.add_edge(i, j)

    merged_groups = list(nx.connected_components(G))
    merged_clusters = []

    for group in merged_groups:
        group = sorted(list(group))
        all_inds = np.concatenate([clusters[i]['inds'] for i in group])
        merged_clusters.append(np.sort(all_inds))

    merged_eis = []
    for inds in merged_clusters:
        ei = np.mean(clusters[0]['ei'][:, :, np.newaxis].repeat(len(inds), axis=2), axis=2)
        ei -= ei[:, :5].mean(axis=1, keepdims=True)
        merged_eis.append(ei)

    similarities = compare_eis(merged_eis, ei_template=reference_ei).flatten()


    best_idx = int(np.argmax(similarities))
    final_inds = merged_clusters[best_idx]
    final_ei = merged_eis[best_idx]
    # Pick any cluster that contributed to this merged group
    group_cluster_indices = list(merged_groups[best_idx])
    final_channels = clusters[group_cluster_indices[0]]['channels'] # assumed same across merged

    return final_ei, final_inds, final_channels, best_idx


def check_2d_gap_peaks_valley(pc_vals, angle_step=10, min_valley_frac=0.1):
    angles = np.deg2rad(np.arange(0, 180, angle_step))
    n_total = len(pc_vals)

    for theta in angles:
        proj = pc_vals[:, 0] * np.cos(theta) + pc_vals[:, 1] * np.sin(theta)
        n_bins = 10
        hist, edges = np.histogram(proj, bins=n_bins)

        # Find local maxima
        peak_inds, _ = find_peaks(hist)

        if len(peak_inds) < 2:
            continue  # Only one peak, nothing to check

        # Check valleys between pairs of peaks
        for i in range(len(peak_inds)-1):
            left_peak = peak_inds[i]
            right_peak = peak_inds[i+1]
            left_count = hist[left_peak]
            right_count = hist[right_peak]
            min_peak_count = min(left_count, right_count)

            # Get bins between peaks
            between = hist[left_peak+1 : right_peak]
            if len(between) == 0:
                continue

            min_valley = np.min(between)

            # Plot
            # plt.figure(figsize=(3,2))
            # plt.bar((edges[:-1] + edges[1:]) / 2, hist, width=(edges[1]-edges[0]))
            # plt.title(f"Angle {np.rad2deg(theta):.1f} deg,  Min valley: {min_valley}, Peaks {left_count}, {right_count}")
            # plt.show()

            if min_valley <= min_valley_frac * min_peak_count:
                # Determine valley bin edge
                valley_bin_idx = np.where(between == min_valley)[0][0] + left_peak + 1
                split_val = (edges[valley_bin_idx] + edges[valley_bin_idx + 1]) / 2

                group1_mask = proj < split_val
                group2_mask = proj >= split_val

                # print(f"Gap found at angle {np.rad2deg(theta):.1f} deg")
                # print(f"  Peaks at bins {left_peak}, {right_peak}: counts {left_count}, {right_count}")
                # print(f"  Min valley: {min_valley}")
                
                # # Plot
                # plt.figure(figsize=(3,2))
                # plt.bar((edges[:-1] + edges[1:]) / 2, hist, width=(edges[1]-edges[0]))
                # plt.title(f"Angle {np.rad2deg(theta):.1f} deg")
                # plt.show()

                return group1_mask, group2_mask
            
    return None



def subtract_pca_cluster_means_ram(snippets, baselines, spike_times, segment_len=100_000, n_clusters=5, offset_window=(-5,10)):
    """
    Subtracts PCA-clustered mean waveforms from baseline-corrected spike snippets for a single channel.

    Parameters:
    - snippets: (n_spikes, snip_len) array for a single channel
    - baselines: (n_segments,) array of mean baseline per segment for this channel
    - spike_times: (n_spikes,) array of spike times (in samples)
    - segment_len: segment size used for baseline estimation (default 100_000 samples)
    - n_clusters: number of PCA/k-means clusters to use (default 5)

    Returns:
    - residuals: (n_spikes, snip_len) int16 array of subtracted residuals with baseline added back
    - scale_factors: (n_spikes,) float32 array of amplitude scaling per spike
    - cluster_ids: (n_spikes,) int32 array of cluster IDs
    """
    # Compute baseline index per spike
     # --- Baseline subtraction ---
    segment_ids = spike_times // segment_len
    segment_ids = np.clip(segment_ids, 0, len(baselines) - 1)
    baseline_per_spike = baselines[segment_ids][:, np.newaxis]  # shape: (n_spikes, 1)
    snippets_bs = snippets - baseline_per_spike

    # --- Template and window ---
    template = np.mean(snippets_bs, axis=0)
    neg_peak_idx = np.argmin(template)
    w_start = max(0, neg_peak_idx + offset_window[0])
    w_end = min(snippets.shape[1], neg_peak_idx + offset_window[1])
    window = slice(w_start, w_end)

    # filter by amplitude
    neg_peak_amps = -snippets_bs[:, neg_peak_idx]  # flip sign to get positive amplitude
    mean_amp = np.mean(neg_peak_amps)
    lower_bound = 0.75 * mean_amp
    upper_bound = 1.25 * mean_amp
    accepted_spike_mask = (neg_peak_amps >= lower_bound) & (neg_peak_amps <= upper_bound)
    accepted_indices = np.where(accepted_spike_mask)[0]  # indices into snippets_bs

    n_accepted = len(accepted_indices)

    # Create full-length placeholders
    n_spikes = snippets_bs.shape[0]
    scale_factors = np.full(n_spikes, -1.0, dtype=np.float32)
    cluster_ids = np.full(n_spikes, -1, dtype=np.int32)


    if n_accepted < 50:
        # Too few spikes to reliably cluster, fallback to global template subtraction
        global_template = np.mean(snippets_bs, axis=0)
        accepted_spike_mask[:] = False
        accepted_indices = np.array([], dtype=int)
    else:
        global_template = np.mean(snippets_bs[accepted_indices], axis=0)
        # print(-mean_amp)
        # print(snippets_bs[:5,neg_peak_idx])
        # PCA
        snips_for_pca = snippets_bs[accepted_indices][:, window]  # shape: (n_accepted, window_len)
        pca = PCA(n_components=5)
        reduced = pca.fit_transform(snips_for_pca)
        cluster_ids_accepted = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(reduced)

        global_template_clusters = np.mean(snips_for_pca, axis=0)
        global_template_clusters /= np.linalg.norm(global_template_clusters) + 1e-8


        full_scale_factors = np.full(len(accepted_indices), -1.0, dtype=np.float32)
        full_cluster_ids = np.full(len(accepted_indices), -1, dtype=np.int32)

        # template assignment and subtraction
        for c in range(n_clusters):
            idx = np.where(cluster_ids_accepted == c)[0]
            if len(idx) == 0:
                continue
            # Extract snippets from this cluster within the window
            snippets_c = snips_for_pca[idx]  # shape: (N_c, win_len)
            # Compute cluster template
            template = np.mean(snippets_c, axis=0)  # shape: (win_len,)

            template_norm = template.copy()
            template_norm /= np.linalg.norm(template) + 1e-8
            dot = np.dot(global_template_clusters, template_norm)

            # # test plots
            # plt.figure(figsize=(4,2))
            # plt.plot(template, alpha=0.5, linewidth=2)
            # plt.grid(True)
            # plt.title(dot)
            # plt.tight_layout()
            # plt.show()

            if dot < 0.80:  # can tune this
                full_idx = accepted_indices[idx]
                accepted_spike_mask[full_idx] = False
                full_scale_factors[idx] = -1
                full_cluster_ids[idx] = -1
            else:
                # Compute scale factors
                dot_template = np.dot(template, template) + 1e-8  # avoid div by zero
                scales = np.dot(snippets_c, template) / dot_template  # shape: (N_c,)
                scales = np.clip(scales, 0.75, 1.25)
                # Subtract scaled template to get residuals
                residuals_c = snippets_c - scales[:, np.newaxis] * template  # shape: (N_c, win_len)
                # Insert residuals into full snippets
                for j, spike_idx in enumerate(idx):
                    full_idx = accepted_indices[spike_idx]  # index in original snippets_bs
                    snippets_bs[full_idx, window] = residuals_c[j]
                    full_scale_factors[spike_idx] = scales[j]
                    full_cluster_ids[spike_idx] = c

        # Fill in values only for accepted spikes
        scale_factors[accepted_indices] = full_scale_factors  # computed per cluster
        cluster_ids[accepted_indices] = full_cluster_ids      # same order as accepted_indices

    # deal with rejected spikes
    rejected_spike_mask = ~accepted_spike_mask
    rejected_indices = np.where(rejected_spike_mask)[0]
    for idx in rejected_indices:
        snippets_bs[idx, window] -= global_template[window]


    # # --- Compute scale factors (only within window) ---
    # dot_template = np.dot(template[window], template[window])
    # scale_factors = np.dot(snippets_bs[:, window].astype(np.float32), template[window].astype(np.float32)) / dot_template
    # scale_factors = np.clip(scale_factors, 0.5, 2.0).astype(np.float32)

    # # --- Fit scaled template (only within window) ---
    # scaled_snips = np.copy(snippets_bs)
    # scaled_snips[:, window] *= scale_factors[:, np.newaxis]

    # # --- PCA and clustering ---
    # pca = PCA(n_components=5)
    # reduced = pca.fit_transform(scaled_snips[:, window])
    # cluster_ids = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(reduced)

    # # --- Reassign undersized clusters ---
    # counts = np.bincount(cluster_ids)
    # main_cluster = np.argmax(counts)
    # for c in range(n_clusters):
    #     if counts[c] < 10:
    #         cluster_ids[cluster_ids == c] = main_cluster

    # # --- Subtract cluster means (only in window) ---
    # residuals = np.copy(snippets_bs)
    # for c in range(n_clusters):
    #     idx = np.where(cluster_ids == c)[0]
    #     if len(idx) == 0:
    #         continue
    #     cluster_mean = np.mean(scaled_snips[idx, window], axis=0)
    #     residuals[idx, window] = scaled_snips[idx, window] - cluster_mean

    # --- Restore baseline, clip, convert ---
    residuals = snippets_bs
    residuals += baseline_per_spike
    residuals = np.clip(residuals, -32768, 32767).astype(np.int16)

    return residuals, scale_factors, cluster_ids


import numpy as np

def apply_residuals(
    raw_data: np.ndarray = None,
    dat_path: str = None,
    residual_snips_per_channel: dict = None,
    write_locs: np.ndarray = None,
    selected_channels: np.ndarray = None,
    total_samples: int = None,
    dtype: np.dtype = np.int16,
    n_channels: int = 512,
    is_ram: bool = False,
    is_disk: bool = False
):
    """
    Applies residual snippets to time-major data (RAM and/or disk).

    Parameters:
        raw_data: RAM array [time, channels] (optional, required if is_ram)
        dat_path: Path to time-major .dat file (optional, required if is_disk)
        residual_snips_per_channel: {channel: [n_spikes, snip_len] int16 array}
        write_locs: Spike center locations [n_spikes]
        selected_channels: List/array of channels to update
        total_samples: Number of timepoints in the recording
        dtype: Data type of stored file
        n_channels: Total number of channels
        is_ram: If True, modify raw_data
        is_disk: If True, modify file at dat_path

    Returns:
        None
    """
    if not is_ram and not is_disk:
        raise ValueError("At least one of is_ram or is_disk must be True.")

    if is_disk:
        data_disk = np.memmap(dat_path, dtype=dtype, mode='r+', shape=(total_samples, n_channels))
    else:
        data_disk = None

    for ch in selected_channels:
        residuals = residual_snips_per_channel[ch]

        if residuals.shape[0] != len(write_locs):
            raise ValueError(f"Mismatch between residuals and write_locs for channel {ch}")

        for i, (snip, loc) in enumerate(zip(residuals, write_locs)):
            end = loc + snip.shape[0]
            if end > total_samples:
                print(f"    Skipping spike {i} (ends at {end}, beyond total_samples)")
                continue

            if is_ram:
                raw_data[loc:end, ch] = snip
            if is_disk:
                data_disk[loc:end, ch] = snip

    if is_disk:
        data_disk.flush()
        del data_disk


import numpy as np
from scipy.signal import correlate





def plot_unit_diagnostics(
    output_path: str,
    unit_id: int,
    pcs_pre: np.ndarray,
    labels_pre: np.ndarray,
    sim_matrix_pre: np.ndarray,
    cluster_eis_pre: np.ndarray,
    spikes_for_plot_pre: np.ndarray,
    n_bad_channels_pre: np.ndarray,
    contributing_original_ids_pre: np.ndarray,
    mean_score: np.ndarray,
    valid_score: np.ndarray,
    mean_scores_at_spikes: np.ndarray,
    valid_scores_at_spikes: np.ndarray,
    mean_thresh: float,
    valid_thresh: float,
    lags: np.ndarray,
    bad_spike_traces: np.ndarray,
    good_mean_trace: np.ndarray,
    threshold_ampl: float,
    ref_channel: int,
    snips_bad: np.ndarray,
    pcs_post: np.ndarray,
    labels_post: np.ndarray,
    sim_matrix_post: np.ndarray,
    spikes_for_plot_post: np.ndarray,
    n_bad_channels_post: np.ndarray,
    contributing_original_ids_post: np.ndarray,
    cluster_eis_post: np.ndarray,
    window: tuple,
    ei_positions: np.ndarray,
    selected_channels_count: int,
    spikes: np.ndarray,
    orig_threshold: float,
    ks_matches: list
):
    
    # --- STA generation ---
    sta_depth = 30
    sta_offset = 0
    sta_chunk_size = 1000
    sta_refresh = 2
    fs = 20000  # sampling rate in Hz

    triggers_mat_path='/Volumes/Lab/Users/alexth/axolotl/trigger_in_samples_201703151.mat'

    triggers = loadmat(triggers_mat_path)['triggers'].flatten() # triggers in s

    lut = np.array([
        [255, 255, 255],
        [255, 255,   0],
        [255,   0, 255],
        [255,   0,   0],
        [0,   255, 255],
        [0,   255,   0],
        [0,     0, 255],
        [0,     0,   0]
    ], dtype=np.uint8).flatten()
    
    generator = RGBFrameGenerator('/Volumes/Lab/Users/alexth/axolotl/sta/libdraw_rgb.so')
    generator.configure(width=20, height=40, lut=lut, noise_type=1, n_bits=3)

    # --- FIGURE setup ---

    fig = plt.figure(figsize=(16, 32))
    gs = gridspec.GridSpec(7, 4, height_ratios=[0.7, 2.0, 0.7, 0.7, 0.7, 2, 0.7], width_ratios=[1, 1, 1, 1], wspace=0.25)

    # --- Row 1: PCA pre-merge and sim matrix ---
    row1_gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0, :], wspace=0.05)

    ax1 = fig.add_subplot(row1_gs[0])
    ax1.set_title("Initial PCA (pre-merge)")
    for lbl in np.unique(labels_pre):
        pts = pcs_pre[labels_pre == lbl]
        ax1.scatter(pts[:, 0], pts[:, 1], s=5, label=f"{len(pts)} sp")
        # ax1.scatter(pts[:, 0], pts[:, 1], s=5, label=f"Cluster {lbl} (N={len(pts)})")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True)
    ax1.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),  # x=center, y=below the axis
        ncol=len(np.unique(labels_pre)),  # spread horizontally
        fontsize=14,
        frameon=False
    )

    # ax1_legend = fig.add_subplot(gs[0, 1])
    # ax1_legend.axis('off')
    # ax1_legend.legend(*ax1.get_legend_handles_labels(), loc='center left', fontsize=14)

    # Extract cluster labels
    cluster_ids_pre = sorted(np.unique(labels_pre))
    # Get default matplotlib color cycle
    default_colors = itertools.cycle(rcParams['axes.prop_cycle'].by_key()['color'])
    # Build color list in order of cluster appearance
    colors_pre = [next(default_colors) for _ in cluster_ids_pre]

    ax2 = fig.add_subplot(row1_gs[1])
    ax2.set_title("Similarity Matrix (Pre)")

    label_matrix = np.empty_like(sim_matrix_pre, dtype=object)
    n = sim_matrix_pre.shape[0]
    for i in range(n):
        for j in range(n):
            score = sim_matrix_pre[i, j]
            n_bad = n_bad_channels_pre[i, j]
            label_matrix[i, j] = f"{score:.2f}/{n_bad}"


    tb = Table(ax2, bbox=[0.2, 0.2, 0.8, 0.8])
    n = sim_matrix_pre.shape[0]
    for i in range(n):
        for j in range(n):
            tb.add_cell(i, j, 1/n, 1/n, text=label_matrix[i, j], loc='center')
    for i in range(n):
        tb.add_cell(i, -1, 1/n, 1/n, text=str(i), loc='right', edgecolor='none')
        tb.add_cell(-1, i, 1/n, 1/n, text=str(i), loc='center', edgecolor='none')
    ax2.add_table(tb)
    ax2.axis('off')

    s0 = spikes_for_plot_pre[labels_pre == 0]
    s1 = spikes_for_plot_pre[labels_pre == 1]
    s2 = spikes_for_plot_pre[labels_pre == 2]

    if len(s0) > 0 and s0[0]>0:
        ax_sta0 = fig.add_subplot(row1_gs[2])   # STA cluster 0
        sta = compute_sta_chunked(
            spikes_sec=s0/fs,
            triggers_sec=triggers,
            generator=generator,
            seed=11111,
            depth=sta_depth,
            offset=sta_offset,
            chunk_size=sta_chunk_size,
            refresh=sta_refresh
        )
        # Time course from peak pixel
        max_idx = np.unravel_index(np.abs(sta).argmax(), sta.shape)
        # Display STA frame at peak time
        peak_frame = max_idx[3]
        if peak_frame>7 or peak_frame<3:
            peak_frame = 4
        rgb = sta[:, :, :, peak_frame]
        vmax = np.max(np.abs(sta)) * 2
        norm_rgb = rgb / vmax + 0.5
        norm_rgb = np.clip(norm_rgb, 0, 1)

        ax_sta0.imshow(norm_rgb.transpose(1, 0, 2), origin='upper')
        ax_sta0.set_title(f"STA Frame {peak_frame + 1}", fontsize=10)
        ax_sta0.set_aspect('equal')
        ax_sta0.axis('off')
        ax_sta0.set_position(ax_sta0.get_position().expanded(1.1, 1.0))


    if len(s1) > 0 and s1[0]>0:
        ax_sta1 = fig.add_subplot(row1_gs[3])   # STA cluster 1
        sta = compute_sta_chunked(
            spikes_sec=s1/fs,
            triggers_sec=triggers,
            generator=generator,
            seed=11111,
            depth=sta_depth,
            offset=sta_offset,
            chunk_size=sta_chunk_size,
            refresh=sta_refresh
        )
        # Time course from peak pixel
        max_idx = np.unravel_index(np.abs(sta).argmax(), sta.shape)
        # Display STA frame at peak time
        peak_frame = max_idx[3]
        if peak_frame>7 or peak_frame<3:
            peak_frame = 4
        rgb = sta[:, :, :, peak_frame]
        vmax = np.max(np.abs(sta)) * 2
        norm_rgb = rgb / vmax + 0.5
        norm_rgb = np.clip(norm_rgb, 0, 1)

        ax_sta1.imshow(norm_rgb.transpose(1, 0, 2), origin='upper')
        ax_sta1.set_title(f"STA Frame {peak_frame + 1}", fontsize=10)
        ax_sta1.set_aspect('equal')
        ax_sta1.axis('off')
        ax_sta1.set_position(ax_sta1.get_position().expanded(1.1, 1.0))


    if len(s2) > 0 and s2[0]>0:
        ax_sta2 = fig.add_subplot(row1_gs[4])   # STA cluster 2
        sta = compute_sta_chunked(
            spikes_sec=s2/fs,
            triggers_sec=triggers,
            generator=generator,
            seed=11111,
            depth=sta_depth,
            offset=sta_offset,
            chunk_size=sta_chunk_size,
            refresh=sta_refresh
        )
        # Time course from peak pixel
        max_idx = np.unravel_index(np.abs(sta).argmax(), sta.shape)
        # Display STA frame at peak time
        peak_frame = max_idx[3]
        if peak_frame>7 or peak_frame<3:
            peak_frame = 4
        rgb = sta[:, :, :, peak_frame]
        vmax = np.max(np.abs(sta)) * 2
        norm_rgb = rgb / vmax + 0.5
        norm_rgb = np.clip(norm_rgb, 0, 1)

        ax_sta2.imshow(norm_rgb.transpose(1, 0, 2), origin='upper')
        ax_sta2.set_title(f"STA Frame {peak_frame + 1}", fontsize=10)
        ax_sta2.set_aspect('equal')
        ax_sta2.axis('off')
        ax_sta2.set_position(ax_sta2.get_position().expanded(1.1, 1.0))

    # --- Row 2: EI waveforms pre ---

    #ei_row_pre = gridspec.GridSpecFromSubplotSpec(1, max(len(ei_clusters_pre), 2), subplot_spec=gs[1, :])
    ei_row_pre = fig.add_subplot(gs[1, :])  # one full-width plot


    plot_ei_waveforms(
        ei=cluster_eis_pre,                 # list of EIs
        positions=ei_positions,
        ref_channel=ref_channel,
        scale=70.0,
        box_height=1.5,
        box_width=50,
        linewidth=1,
        alpha=0.9,
        colors=colors_pre,                 # same colors as PCA
        ax=ei_row_pre
    )

    ei_row_pre.set_title(f"Cluster EIs; clusters {contributing_original_ids_pre} merged; total spikes {len(mean_scores_at_spikes)}", fontsize=16)

    # for i in range(len(ei_clusters_pre)):
    #     ax = fig.add_subplot(ei_row_pre[i])
    #     color = 'red' if i == selected_index_pre else 'black'
    #     plot_ei_waveforms(
    #         ei=ei_clusters_pre[i],
    #         positions=ei_positions,
    #         scale=70.0,
    #         box_height=1.5,
    #         box_width=50,
    #         linewidth=1,
    #         alpha=0.9,
    #         colors=color,
    #         ax=ax
    #     )
    #     ax.set_title(f"Cluster {i} | N={spike_counts_pre[i]}")


    # --- Row 3: score histograms ---
    def clipped_hist(ax, scores, threshold, title, bins=100):
        scores = np.asarray(scores)
        valid = scores[np.isfinite(scores)]

        if valid.size == 0 or not np.isfinite(valid).any():
            ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='red')
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        counts, bins = np.histogram(valid, bins=bins)
        cutoff = np.sort(counts)[-4] if len(counts) >= 4 else max(counts)
        ax.hist(valid, bins=bins, alpha=0.6, color='gray')
        ax.axvline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold:.2f}")
        ax.set_ylim(0, cutoff * 1.1)
        ax.set_title(title)
        ax.grid(True)

    if mean_score is not None:
        ax3a = fig.add_subplot(gs[2, 0])
        clipped_hist(ax3a, mean_score, mean_thresh, "Mean Score (all)")

        ax3b = fig.add_subplot(gs[2, 1])
        clipped_hist(ax3b, mean_scores_at_spikes, mean_thresh, f"Mean Score ({len(mean_scores_at_spikes)} spikes)")

        ax3c = fig.add_subplot(gs[2, 2])
        clipped_hist(ax3c, valid_score, valid_thresh, "Valid Channels (all)", bins=np.arange(0, selected_channels_count + 2))

        ax3d = fig.add_subplot(gs[2, 3])
        clipped_hist(ax3d, valid_scores_at_spikes, valid_thresh, "Valid Channels (spikes)", bins=np.arange(0, selected_channels_count + 2))

    # --- Row 4: Lags and bad spikes ---
    row4_gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[3, :])

    ax4a = fig.add_subplot(row4_gs[0])       # Lags
    lags_nonzero = lags[lags != 0]
    if lags_nonzero.size > 0:
        ax4a.hist(lags_nonzero, bins=np.arange(-6.5, 7.5, 1), color='purple', alpha=0.7)
    else:
        ax4a.text(0.5, 0.5, 'No nonzero lags', transform=ax4a.transAxes,
                ha='center', va='center', fontsize=10, color='red')
        ax4a.set_xticks([])
        ax4a.set_yticks([])
    ax4a.set_title(f"Lags (Excl. 0); total spikes {len(lags)}")
    ax4a.set_xlabel("Lag (samples)")
    ax4a.grid(True)


    ax4b = fig.add_subplot(row4_gs[1:3])     # Bad spikes
    if isinstance(bad_spike_traces, np.ndarray) and bad_spike_traces.shape[0] > 0:
        for trace in snips_bad:
            ax4b.plot(trace, color='green', alpha=1, linewidth=1)
        for trace in bad_spike_traces:
            ax4b.plot(trace, color='black', alpha=1, linewidth=1)
    else:
        ax4b.text(0.5, 0.5, 'No bad spikes', transform=ax4b.transAxes,
                ha='center', va='center', fontsize=10, color='red')
        ax4b.set_xticks([])
        ax4b.set_yticks([])

    ax4b.plot(good_mean_trace, color='red', linewidth=1.5, label='Good Mean')
    ax4b.axhline(threshold_ampl, color='black', linestyle='--', label=f"Threshold = {threshold_ampl:.2f}")
    ax4b.set_title(f"Ref Channel {ref_channel+1} | {len(bad_spike_traces)} bad spikes | orig threshold {-orig_threshold:.1f}")
    ax4b.grid(True)

    ax4b.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),  # x=center, y=below the axis
        ncol=2,  # spread horizontally
        fontsize=14,
        frameon=False
    )

    ax4c = fig.add_subplot(row4_gs[3:5])     # KS matches
    ax4c.axis('off')

    if ks_matches:
        table_data = [
            ["KS Unit", "Vision ID", "Sim", "N spikes"]
        ] + [
            [m["unit_id"], m["vision_id"],f"{m['similarity']:.2f}", m["n_spikes"]] for m in ks_matches
        ]

        tb = ax4c.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            loc='center',
            cellLoc='center',
            bbox=[0.3, 0.0, 0.7, 1.0] 
        )
        tb.scale(0.6, 1)
        for i in range(len(ks_matches) + 1):
            for j in range(4):
                tb[(i, j)].set_fontsize(14)
    else:
        ax4c.text(0.5, 0.5, 'No match', transform=ax4c.transAxes,
                       ha='center', va='center', fontsize=14, color='gray')
    
    ax4c.set_title("Matching KS Units")

    # --- Row 5: PCA post and sim matrix ---
    ax5 = fig.add_subplot(gs[4, 0])
    ax5.set_title("Post-EI PCA")
    for lbl in np.unique(labels_post):
        pts = pcs_post[labels_post == lbl]
        ax5.scatter(pts[:, 0], pts[:, 1], s=5, label=f"{len(pts)} sp")
    ax5.set_xlabel("PC1")
    ax5.set_ylabel("PC2")
    ax5.set_aspect('equal', adjustable='box')
    ax5.grid(True)
    ax5.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),  # x=center, y=below the axis
        ncol=len(np.unique(labels_post)),  # spread horizontally
        fontsize=14,
        frameon=False
    )

    ax5a = fig.add_subplot(gs[4, 1])
    ax5a.set_title("Similarity Matrix (Post)")

    label_matrix = np.empty_like(sim_matrix_post, dtype=object)
    n = sim_matrix_post.shape[0]
    for i in range(n):
        for j in range(n):
            score = sim_matrix_post[i, j]
            n_bad = n_bad_channels_post[i, j]
            label_matrix[i, j] = f"{score:.2f}/{n_bad}"


    tb = Table(ax5a, bbox=[0.2, 0.2, 0.8, 0.8])
    n = sim_matrix_post.shape[0]
    for i in range(n):
        for j in range(n):
            tb.add_cell(i, j, 1/n, 1/n, text=label_matrix[i, j], loc='center')
    for i in range(n):
        tb.add_cell(i, -1, 1/n, 1/n, text=str(i), loc='right', edgecolor='none')
        tb.add_cell(-1, i, 1/n, 1/n, text=str(i), loc='center', edgecolor='none')
    ax5a.add_table(tb)
    ax5a.axis('off')

    # tb = Table(ax5a, bbox=[0.2, 0.2, 0.6, 0.6])
    # n = sim_matrix_post.shape[0]
    # for i in range(n):
    #     for j in range(n):
    #         tb.add_cell(i, j, 1/n, 1/n, text=f"{sim_matrix_post[i, j]:.2f}", loc='center')
    # for i in range(n):
    #     tb.add_cell(i, -1, 1/n, 1/n, text=str(i), loc='right', edgecolor='none')
    #     tb.add_cell(-1, i, 1/n, 1/n, text=str(i), loc='center', edgecolor='none')
    # ax5a.add_table(tb)
    # ax5a.axis('off')


    s0 = spikes_for_plot_post[labels_post == 0]
    s1 = spikes_for_plot_post[labels_post == 1]

    if len(s0) > 0 and s0[0]>0:
        ax_sta50 = fig.add_subplot(gs[4, 2])   # STA cluster 0
        sta = compute_sta_chunked(
            spikes_sec=s0/fs,
            triggers_sec=triggers,
            generator=generator,
            seed=11111,
            depth=sta_depth,
            offset=sta_offset,
            chunk_size=sta_chunk_size,
            refresh=sta_refresh
        )
        # Time course from peak pixel
        max_idx = np.unravel_index(np.abs(sta).argmax(), sta.shape)
        # Display STA frame at peak time
        peak_frame = max_idx[3]
        if peak_frame>7 or peak_frame<3:
            peak_frame = 4
        rgb = sta[:, :, :, peak_frame]
        vmax = np.max(np.abs(sta)) * 2
        norm_rgb = rgb / vmax + 0.5
        norm_rgb = np.clip(norm_rgb, 0, 1)

        ax_sta50.imshow(norm_rgb.transpose(1, 0, 2), origin='upper')
        ax_sta50.set_title(f"STA Frame {peak_frame + 1}", fontsize=10)
        ax_sta50.set_aspect('equal')
        ax_sta50.axis('off')
        ax_sta50.set_position(ax_sta50.get_position().expanded(1.1, 1.0))

    if len(s1) > 0 and s1[0]>0:
        ax_sta51 = fig.add_subplot(gs[4, 3])   # STA cluster 0
        sta = compute_sta_chunked(
            spikes_sec=s1/fs,
            triggers_sec=triggers,
            generator=generator,
            seed=11111,
            depth=sta_depth,
            offset=sta_offset,
            chunk_size=sta_chunk_size,
            refresh=sta_refresh
        )
        # Time course from peak pixel
        max_idx = np.unravel_index(np.abs(sta).argmax(), sta.shape)
        # Display STA frame at peak time
        peak_frame = max_idx[3]
        if peak_frame>7 or peak_frame<3:
            peak_frame = 4
        rgb = sta[:, :, :, peak_frame]
        vmax = np.max(np.abs(sta)) * 2
        norm_rgb = rgb / vmax + 0.5
        norm_rgb = np.clip(norm_rgb, 0, 1)

        ax_sta51.imshow(norm_rgb.transpose(1, 0, 2), origin='upper')
        ax_sta51.set_title(f"STA Frame {peak_frame + 1}", fontsize=10)
        ax_sta51.set_aspect('equal')
        ax_sta51.axis('off')
        ax_sta51.set_position(ax_sta51.get_position().expanded(1.1, 1.0))



    # --- Row 6: Final EI waveforms ---
    #ei_row_post = gridspec.GridSpecFromSubplotSpec(1, max(len(ei_clusters_post), 2), subplot_spec=gs[5, :])
    ei_row_post = fig.add_subplot(gs[5, :])  # one full-width plot
    plot_ei_waveforms(
        ei=cluster_eis_post,                 # list of EIs
        positions=ei_positions,
        ref_channel=ref_channel,
        scale=70.0,
        box_height=1.5,
        box_width=50,
        linewidth=1,
        alpha=0.9,
        colors=colors_pre,                 # same colors as PCA
        ax=ei_row_post
    )

    ei_row_post.set_title(f"Cluster EIs; clusters {contributing_original_ids_post} merged; total spikes {len(spikes)}", fontsize=16)

    # Row 7: Final unit firing, ISI, STA time course, STA frame
    ax7a = fig.add_subplot(gs[6, 0])
    ax7b = fig.add_subplot(gs[6, 1])
    ax7c = fig.add_subplot(gs[6, 2])
    ax7d = fig.add_subplot(gs[6, 3])

    fs = 20000  # sampling rate in Hz
    times_sec = np.sort(spikes) / fs # spikes in seconds

    # --- Firing rate plot (smoothed) ---
    if len(times_sec) > 0:
        sigma_ms=2500.0
        dt_ms=1000.0
        dt = dt_ms / 1000.0
        sigma_samples = sigma_ms / dt_ms
        total_duration = 1800.1
        time_vector = np.arange(0, total_duration, dt)
        counts, _ = np.histogram(times_sec, bins=np.append(time_vector, total_duration))
        rate = gaussian_filter1d(counts / dt, sigma=sigma_samples)
        ax7a.plot(time_vector, rate, color='black')
        ax7a.set_title(f"Smoothed Firing Rate, {len(spikes)} spikes")
        ax7a.set_xlabel("Time (s)")
        ax7a.set_ylabel("Rate (Hz)")
    else:
        ax7a.text(0.5, 0.5, 'No spikes', transform=ax7a.transAxes,
                 ha='center', va='center', fontsize=10, color='red')
        ax7a.set_xticks([])
        ax7a.set_yticks([])


    # --- ISI histogram ---

    if len(times_sec) > 1:
        isi = np.diff(times_sec) # differences in seconds
        isi_max_s = 200.0 / 1000.0  # convert to seconds
        bins = np.arange(0, isi_max_s + 0.0005, 0.0005)
        hist, _ = np.histogram(isi, bins=bins)
        fractions = hist / hist.sum() if hist.sum() > 0 else np.zeros_like(hist)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax7b.plot(bin_centers, fractions, color='blue')
        ax7b.set_xlim(0, isi_max_s)
        ax7b.set_ylim(0, np.max(fractions) * 1.1)
        ax7b.set_title("ISI Histogram", fontsize=10)
        ax7b.set_xlabel("ISI (ms)")
    else:
        ax7b.text(0.5, 0.5, 'No ISIs', transform=ax7b.transAxes,
            ha='center', va='center', fontsize=10, color='red')
        ax7b.set_xticks([])
        ax7b.set_yticks([])


    if len(times_sec) > 0 and times_sec[0]>0:
        sta = compute_sta_chunked(
            spikes_sec=times_sec,
            triggers_sec=triggers,
            generator=generator,
            seed=11111,
            depth=sta_depth,
            offset=sta_offset,
            chunk_size=sta_chunk_size,
            refresh=sta_refresh
        )
        # Time course from peak pixel
        max_idx = np.unravel_index(np.abs(sta).argmax(), sta.shape)
        y, x = max_idx[0], max_idx[1]
        red_tc = sta[y, x, 0, :][::-1]
        green_tc = sta[y, x, 1, :][::-1]
        blue_tc = sta[y, x, 2, :][::-1]

        ax7c.plot(red_tc, color='red', label='R')
        ax7c.plot(green_tc, color='green', label='G')
        ax7c.plot(blue_tc, color='blue', label='B')
        ax7c.set_title("STA Time Course", fontsize=10)
        ax7c.set_xlim(0, sta_depth - 1)
        ax7c.set_xticks([0, sta_depth - 1])
        ax7c.set_xlabel("Time (frames)")

        # Display STA frame at peak time
        peak_frame = max_idx[3]
        if peak_frame>7 or peak_frame<3:
            peak_frame = 4
        rgb = sta[:, :, :, peak_frame]
        vmax = np.max(np.abs(sta)) * 2
        norm_rgb = rgb / vmax + 0.5
        norm_rgb = np.clip(norm_rgb, 0, 1)

        ax7d.imshow(norm_rgb.transpose(1, 0, 2), origin='upper')
        ax7d.set_title(f"STA Frame {peak_frame + 1}", fontsize=10)
        ax7d.set_aspect('equal')
        ax7d.axis('off')
    

    plt.subplots_adjust(top=0.97, bottom=0.03, left=0.05, right=0.98, hspace=0.5, wspace=0.25)
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(os.path.join(output_path, f"unit_{unit_id:03d}_diagnostics_ram.png"), dpi=150)
    # plt.close(fig)





def plot_unit_diagnostics_single_cluster(
    output_path: str,
    unit_id: int,
    pcs_pre: np.ndarray,
    labels_pre: np.ndarray,
    sim_matrix_pre: np.ndarray,
    cluster_eis_pre: np.ndarray,
    spikes_for_plot_pre: np.ndarray,
    n_bad_channels_pre: np.ndarray,
    contributing_original_ids_pre: np.ndarray,
    lags: np.ndarray,
    bad_spike_traces: np.ndarray,
    bad_spike_traces_easy: np.ndarray,
    pcs: np.ndarray,
    outlier_inds_easy: np.ndarray,
    outlier_inds: np.ndarray,
    good_mean_trace: np.ndarray,
    ref_channel: int,
    final_ei: np.ndarray,
    ei_positions: np.ndarray,
    spikes: np.ndarray,
    orig_threshold: float,
    ks_matches: list
):
    
    # --- STA generation ---
    sta_depth = 30
    sta_offset = 0
    sta_chunk_size = 1000
    sta_refresh = 2
    fs = 20000  # sampling rate in Hz

    triggers_mat_path='/Volumes/Lab/Users/alexth/axolotl/trigger_in_samples_201703151.mat'

    triggers = loadmat(triggers_mat_path)['triggers'].flatten() # triggers in s

    lut = np.array([
        [255, 255, 255],
        [255, 255,   0],
        [255,   0, 255],
        [255,   0,   0],
        [0,   255, 255],
        [0,   255,   0],
        [0,     0, 255],
        [0,     0,   0]
    ], dtype=np.uint8).flatten()
    
    generator = RGBFrameGenerator('/Volumes/Lab/Users/alexth/axolotl/sta/libdraw_rgb.so')
    generator.configure(width=20, height=40, lut=lut, noise_type=1, n_bits=3)

    # --- FIGURE setup ---

    fig = plt.figure(figsize=(16, 30))
    gs = gridspec.GridSpec(6, 4, height_ratios=[0.7, 0.7, 2.0, 0.7, 2, 0.7], width_ratios=[1, 1, 1, 1], wspace=0.25)

    # --- Row 1: PCA pre-merge and sim matrix ---
    row1_gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0, :], wspace=0.05)

    color_cycle = itertools.cycle(rcParams['axes.prop_cycle'].by_key()['color'])
    cluster_colors = {}

    ax1 = fig.add_subplot(row1_gs[0])
    ax1.set_title("Initial PCA (pre-merge)")
    for lbl in np.unique(labels_pre):
        color = next(color_cycle)
        cluster_colors[lbl] = color
        pts = pcs_pre[labels_pre == lbl]
        ax1.scatter(pts[:, 0], pts[:, 1], s=5, color=color, label=f"{len(pts)} sp")
        # ax1.scatter(pts[:, 0], pts[:, 1], s=5, label=f"Cluster {lbl} (N={len(pts)})")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True)
    # ax1.legend(
    #     loc='upper center',
    #     bbox_to_anchor=(1, -0.15),  # x=center, y=below the axis
    #     ncol=len(np.unique(labels_pre)),  # spread horizontally
    #     fontsize=14,
    #     frameon=False
    # )

    # Extract cluster labels
    cluster_ids_pre = sorted(np.unique(labels_pre))
    # Get default matplotlib color cycle
    default_colors = itertools.cycle(rcParams['axes.prop_cycle'].by_key()['color'])
    # Build color list in order of cluster appearance
    colors_pre = [next(default_colors) for _ in cluster_ids_pre]


    ax2 = fig.add_subplot(row1_gs[1])
    ax2.set_title("Similarity Matrix (Pre)")

    label_matrix = np.empty_like(sim_matrix_pre, dtype=object)
    n = sim_matrix_pre.shape[0]
    for i in range(n):
        for j in range(n):
            score = sim_matrix_pre[i, j]
            n_bad = n_bad_channels_pre[i, j]
            label_matrix[i, j] = f"{score:.2f}/{n_bad}"


    tb = Table(ax2, bbox=[0.2, 0.2, 0.8, 0.8])
    n = sim_matrix_pre.shape[0]
    for i in range(n):
        for j in range(n):
            tb.add_cell(i, j, 1/n, 1/n, text=label_matrix[i, j], loc='center')
    for i in range(n):
        tb.add_cell(i, -1, 1/n, 1/n, text=str(i), loc='right', edgecolor='none')
        tb.add_cell(-1, i, 1/n, 1/n, text=str(i), loc='center', edgecolor='none')
    ax2.add_table(tb)
    ax2.axis('off')

    # --- STA plotting loop ---
    max_clusters_row1 = 3
    max_clusters_row2 = 4

    cluster_spike_lists = [spikes_for_plot_pre[labels_pre == i] for i in np.unique(labels_pre)]

    # --- Row 1 and 2: STAs ---
    for idx, sN in enumerate(cluster_spike_lists):
        if len(sN) == 0 or sN[0] <= 0:
            continue

        # Select subplot position
        if idx < max_clusters_row1:
            ax = fig.add_subplot(row1_gs[idx+2])
        elif idx < max_clusters_row1 + max_clusters_row2:
            row2_idx = idx - max_clusters_row1
            ax = fig.add_subplot(gs[1, row2_idx])
        else:
            print(f"Skipping cluster {idx}, no space for more plots")
            continue

        # Compute STA
        sta = compute_sta_chunked(
            spikes_sec=sN / fs,
            triggers_sec=triggers,
            generator=generator,
            seed=11111,
            depth=sta_depth,
            offset=sta_offset,
            chunk_size=sta_chunk_size,
            refresh=sta_refresh
        )

        max_idx = np.unravel_index(np.abs(sta).argmax(), sta.shape)
        peak_frame = max_idx[3]
        if peak_frame > 7 or peak_frame < 3:
            peak_frame = 4

        rgb = sta[:, :, :, peak_frame]
        vmax = np.max(np.abs(sta)) * 2
        norm_rgb = rgb / vmax + 0.5
        norm_rgb = np.clip(norm_rgb, 0, 1)

        title_color = cluster_colors.get(idx, 'black')  # default to black if not found

        ax.imshow(norm_rgb.transpose(1, 0, 2), origin='upper')
        ax.set_title(f"Cluster {idx}. Frame {peak_frame + 1} (N={len(sN)})", fontsize=10, color=title_color)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_position(ax.get_position().expanded(1.1, 1.0))
    
    # --- Row 3: EI waveforms pre ---

    #ei_row_pre = gridspec.GridSpecFromSubplotSpec(1, max(len(ei_clusters_pre), 2), subplot_spec=gs[1, :])
    ei_row_pre = fig.add_subplot(gs[2, :])  # one full-width plot

    plot_ei_waveforms(
        ei=cluster_eis_pre,                 # list of EIs
        positions=ei_positions,
        ref_channel=ref_channel,
        scale=70.0,
        box_height=1.5,
        box_width=50,
        linewidth=1,
        alpha=0.9,
        colors=colors_pre,                 # same colors as PCA
        ax=ei_row_pre
    )
    n_selected_spikes = np.sum(np.isin(labels_pre, contributing_original_ids_pre))

    ei_row_pre.set_title(f"Cluster EIs; clusters {contributing_original_ids_pre} merged; {n_selected_spikes} spikes out of total {pcs_pre.shape[0]}", fontsize=16)

    # --- Row 4: Bad spikes ---
    row4_gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[3, :])

    ax4a = fig.add_subplot(row4_gs[0])       # Lags

    ax4a.scatter(pcs[:, 0], pcs[:, 1], s=2, alpha=0.5)
    ax4a.scatter(pcs[outlier_inds_easy, 0], pcs[outlier_inds_easy, 1], color='green', s=6, alpha=1)
    ax4a.scatter(pcs[outlier_inds, 0], pcs[outlier_inds, 1], color='red', s=6, alpha=1)
    ax4a.set_title("PCA on Ref Channel Waveforms")
    ax4a.set_xlabel("PC1")
    ax4a.set_ylabel("PC2")
    ax4a.grid(True)


    ax4b = fig.add_subplot(row4_gs[1:3])     # Bad spikes
    if isinstance(bad_spike_traces_easy, np.ndarray) and bad_spike_traces_easy.shape[0] > 0:
        for trace in bad_spike_traces_easy:
            ax4b.plot(trace, color='green', alpha=1, linewidth=1)
        if isinstance(bad_spike_traces, np.ndarray) and bad_spike_traces.shape[0] > 0:
            for trace in bad_spike_traces:
                ax4b.plot(trace, color='black', alpha=1, linewidth=1)
    else:
        ax4b.text(0.5, 0.5, 'No bad spikes', transform=ax4b.transAxes,
                ha='center', va='center', fontsize=10, color='red')
        ax4b.set_xticks([])
        ax4b.set_yticks([])

    ax4b.plot(good_mean_trace, color='red', linewidth=2, label='Good Mean')
    ax4b.set_title(f"Ref Channel {ref_channel+1} | {len(bad_spike_traces)} bad spikes | orig threshold {-orig_threshold:.1f}")
    ax4b.grid(True)

    ax4b.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),  # x=center, y=below the axis
        ncol=2,  # spread horizontally
        fontsize=14,
        frameon=False
    )

    ax4c = fig.add_subplot(row4_gs[3:5])     # KS matches
    ax4c.axis('off')

    if ks_matches:
        table_data = [
            ["KS Unit", "Vision ID", "Sim", "N spikes"]
        ] + [
            [m["unit_id"], m["vision_id"],f"{m['similarity']:.2f}", m["n_spikes"]] for m in ks_matches
        ]

        tb = ax4c.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            loc='center',
            cellLoc='center',
            bbox=[0.3, 0.0, 0.7, 1.0] 
        )
        tb.scale(0.6, 1)
        for i in range(len(ks_matches) + 1):
            for j in range(4):
                tb[(i, j)].set_fontsize(14)
    else:
        ax4c.text(0.5, 0.5, 'No match', transform=ax4c.transAxes,
                       ha='center', va='center', fontsize=14, color='gray')
    
    ax4c.set_title("Matching KS Units")

    # --- Row 5: Final EI waveforms ---

    ei_row_final = fig.add_subplot(gs[4, :])  # one full-width plot
    plot_ei_waveforms(
        ei=final_ei,                 # list of EIs
        positions=ei_positions,
        ref_channel=ref_channel,
        scale=70.0,
        box_height=1.5,
        box_width=50,
        linewidth=1,
        alpha=0.9,
        colors=colors_pre,                 # same colors as PCA
        ax=ei_row_final
    )

    ei_row_final.set_title(f"Final EI; total spikes {len(spikes)}", fontsize=16)

    # Row 6: Final unit firing, ISI, STA time course, STA frame
    ax7a = fig.add_subplot(gs[5, 0])
    ax7b = fig.add_subplot(gs[5, 1])
    ax7c = fig.add_subplot(gs[5, 2])
    ax7d = fig.add_subplot(gs[5, 3])

    fs = 20000  # sampling rate in Hz
    times_sec = np.sort(spikes) / fs # spikes in seconds

    # --- Firing rate plot (smoothed) ---
    if len(times_sec) > 0:
        sigma_ms=2500.0
        dt_ms=1000.0
        dt = dt_ms / 1000.0
        sigma_samples = sigma_ms / dt_ms
        total_duration = 1800.1
        time_vector = np.arange(0, total_duration, dt)
        counts, _ = np.histogram(times_sec, bins=np.append(time_vector, total_duration))
        rate = gaussian_filter1d(counts / dt, sigma=sigma_samples)
        ax7a.plot(time_vector, rate, color='black')
        ax7a.set_title(f"Smoothed Firing Rate, {len(spikes)} spikes")
        ax7a.set_xlabel("Time (s)")
        ax7a.set_ylabel("Rate (Hz)")
    else:
        ax7a.text(0.5, 0.5, 'No spikes', transform=ax7a.transAxes,
                 ha='center', va='center', fontsize=10, color='red')
        ax7a.set_xticks([])
        ax7a.set_yticks([])


    # --- ISI histogram ---

    if len(times_sec) > 1:
        isi = np.diff(times_sec) # differences in seconds
        isi_max_s = 200.0 / 1000.0  # convert to seconds
        bins = np.arange(0, isi_max_s + 0.0005, 0.0005)
        hist, _ = np.histogram(isi, bins=bins)
        fractions = hist / hist.sum() if hist.sum() > 0 else np.zeros_like(hist)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax7b.plot(bin_centers, fractions, color='blue')
        ax7b.set_xlim(0, isi_max_s)
        ax7b.set_ylim(0, np.max(fractions) * 1.1)
        ax7b.set_title("ISI Histogram", fontsize=10)
        ax7b.set_xlabel("ISI (ms)")
    else:
        ax7b.text(0.5, 0.5, 'No ISIs', transform=ax7b.transAxes,
            ha='center', va='center', fontsize=10, color='red')
        ax7b.set_xticks([])
        ax7b.set_yticks([])


    if len(times_sec) > 0 and times_sec[0]>0:
        sta = compute_sta_chunked(
            spikes_sec=times_sec,
            triggers_sec=triggers,
            generator=generator,
            seed=11111,
            depth=sta_depth,
            offset=sta_offset,
            chunk_size=sta_chunk_size,
            refresh=sta_refresh
        )
        # Time course from peak pixel
        max_idx = np.unravel_index(np.abs(sta).argmax(), sta.shape)
        y, x = max_idx[0], max_idx[1]
        red_tc = sta[y, x, 0, :][::-1]
        green_tc = sta[y, x, 1, :][::-1]
        blue_tc = sta[y, x, 2, :][::-1]

        ax7c.plot(red_tc, color='red', label='R')
        ax7c.plot(green_tc, color='green', label='G')
        ax7c.plot(blue_tc, color='blue', label='B')
        ax7c.set_title("STA Time Course", fontsize=10)
        ax7c.set_xlim(0, sta_depth - 1)
        ax7c.set_xticks([0, sta_depth - 1])
        ax7c.set_xlabel("Time (frames)")

        # Display STA frame at peak time
        peak_frame = max_idx[3]
        if peak_frame>7 or peak_frame<3:
            peak_frame = 4
        rgb = sta[:, :, :, peak_frame]
        vmax = np.max(np.abs(sta)) * 2
        norm_rgb = rgb / vmax + 0.5
        norm_rgb = np.clip(norm_rgb, 0, 1)

        ax7d.imshow(norm_rgb.transpose(1, 0, 2), origin='upper')
        ax7d.set_title(f"STA Frame {peak_frame + 1}", fontsize=10)
        ax7d.set_aspect('equal')
        ax7d.axis('off')
    

    plt.subplots_adjust(top=0.97, bottom=0.03, left=0.05, right=0.98, hspace=0.5, wspace=0.25)
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(os.path.join(output_path, f"unit_{unit_id:03d}_diagnostics_ram.png"), dpi=150)
    plt.close(fig)
