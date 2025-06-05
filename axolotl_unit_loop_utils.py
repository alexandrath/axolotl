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





def find_dominant_channel(
    dat_path: str,
    n_channels: int,
    dtype: np.dtype = np.int16,
    segment_len: int = 100000,
    n_segments: int = 10,
    peak_window: int = 30,
    top_k_neg: int = 20,
    top_k_events: int = 5,
    seed: int = 42
) -> int:
    """
    Identify the channel with the largest spike-like events from a channel-major .dat file.

    Parameters:
        dat_path: Path to channel-major binary file
        n_channels: Total number of channels (should match file format)
        dtype: Data type (e.g., np.int16)
        segment_len: Samples per time segment (along time axis)
        n_segments: Total number of segments to scan
        peak_window: Half-width for local peak-to-peak measurement
        top_k_neg: Number of most negative peaks to use per channel
        top_k_events: Number of top amplitudes to average per channel
        seed: Random seed

    Returns:
        Index of the dominant channel
    """
    import numpy as np
    from scipy.signal import find_peaks

    bytes_per_sample = np.dtype(dtype).itemsize
    file_size = os.path.getsize(dat_path)
    total_samples = file_size // (n_channels * bytes_per_sample)


    # Memory-map entire channel-major file
    #data = np.memmap(dat_path, dtype=dtype, mode='r', shape=(n_channels, -1))
    #total_samples = data.shape[1]

    data = np.memmap(dat_path, dtype=dtype, mode='r', shape=(n_channels, total_samples))


    rng = np.random.default_rng(seed)
    first_segment_start = rng.integers(0, min(100000, total_samples - segment_len))
    other_starts = rng.integers(0, total_samples - segment_len, size=n_segments - 1)
    start_indices = np.concatenate([[first_segment_start], other_starts])

    channel_amplitudes = [[] for _ in range(n_channels)]

    for start in start_indices:
        segment = data[:, start:start + segment_len].astype(np.float32)

        for ch in range(n_channels):
            trace = segment[ch]
            trace -= trace.mean()

            neg_peaks, _ = find_peaks(-trace, distance=20)
            if len(neg_peaks) == 0:
                continue

            sorted_idx = np.argsort(trace[neg_peaks])
            neg_peaks = neg_peaks[sorted_idx[:top_k_neg]]

            for peak_idx in neg_peaks:
                win_start = max(peak_idx - peak_window, 0)
                win_end = min(peak_idx + peak_window + 1, segment_len)
                local_max = np.max(trace[win_start:win_end])
                amplitude = local_max - trace[peak_idx]
                channel_amplitudes[ch].append(amplitude)

    mean_amplitudes = np.zeros(n_channels)
    for ch in range(n_channels):
        amps = np.array(channel_amplitudes[ch])
        if len(amps) > 0:
            mean_amplitudes[ch] = np.mean(np.sort(amps)[-top_k_events:])

    return int(np.argmax(mean_amplitudes))



def estimate_spike_threshold(
    dat_path: str,
    ref_channel: int,
    dtype: np.dtype = np.int16,
    window: int = 30,
    n_channels: int = 512,
    total_samples_to_read: int = 10_000_000,
    block_size: int = 100_000,
    refractory: int = 30,
    top_n: int = 100
) -> tuple[float, np.ndarray]:
    """
    Estimate spike threshold on the dominant channel using negative peaks with post-peak rebound.

    Parameters:
        dat_path: Path to channel-major binary .dat file
        ref_channel: Dominant channel index from step 1
        dtype: Data type of the recording (e.g., np.int16)
        window: ±window around negative peak to detect post-peak amplitude
        n_channels: Total number of electrodes
        total_samples_to_read: How many samples to scan for spike-like events
        block_size: Chunk size for I/O
        refractory: Minimum separation (in samples) between peaks
        top_n: How many events to use to estimate threshold

    Returns:
        threshold: Estimated threshold (0.5× mean of top negative peaks)
        spike_times: Indices of all suprathreshold events detected on the ref channel
    """
    trace = np.zeros(total_samples_to_read, dtype=np.float32)
    bytes_per_sample = np.dtype(dtype).itemsize
    samples_read = 0

    with open(dat_path, 'rb') as f:
        while samples_read < total_samples_to_read:
            samples_to_load = min(block_size, total_samples_to_read - samples_read)
            offset = (ref_channel * total_samples_to_read + samples_read) * bytes_per_sample
            f.seek(offset)
            raw = np.fromfile(f, dtype=dtype, count=samples_to_load)
            if len(raw) != samples_to_load:
                break
            trace[samples_read:samples_read+samples_to_load] = raw
            samples_read += samples_to_load

    trace -= np.mean(trace)

    neg_peaks, _ = find_peaks(-trace, distance=2 * refractory)

    event_amplitudes = []
    event_indices = []

    for idx in neg_peaks:
        win_start = max(idx - window, 0)
        win_end = min(idx + window + 1, len(trace))
        pos_peak = np.max(trace[win_start:win_end])
        amp = pos_peak - trace[idx]
        event_amplitudes.append(amp)
        event_indices.append(idx)

    event_amplitudes = np.array(event_amplitudes)
    event_indices = np.array(event_indices)

    if len(event_amplitudes) >= top_n:
        top_idx = np.argsort(event_amplitudes)[-top_n:]
    else:
        top_idx = np.argsort(event_amplitudes)

    ref_neg_peaks = trace[event_indices[top_idx]]
    threshold = 0.5 * np.mean(np.abs(ref_neg_peaks))

    baseline_window = 20001  # ~1s at 20kHz
    trace_filtered = trace - uniform_filter1d(trace, size=baseline_window, mode='nearest')
    selected_peaks = [idx for idx in event_indices if trace_filtered[idx] < -threshold]
    spike_times = np.array(selected_peaks)

    return threshold, spike_times



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

    snips_sel = snips[selected_channels, :, :]
    C, T, N = snips_sel.shape
    snips_centered = snips_sel - snips_sel.mean(axis=1, keepdims=True)
    snips_flat = snips_centered.transpose(2, 0, 1).reshape(N, -1)

    pca = PCA(n_components=10)
    pcs = pca.fit_transform(snips_flat)

    kmeans = KMeans(n_clusters=k_start, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pcs)

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


    cluster_spike_indices = {k: np.where(labels == k)[0] for k in np.unique(labels)}

    cluster_eis = []
    cluster_ids = sorted(cluster_spike_indices.keys())
    for k in cluster_ids:
        inds = cluster_spike_indices[k]
        ei_k = np.mean(snips[:, :, inds], axis=2)
        ei_k -= ei_k[:, :5].mean(axis=1, keepdims=True)
        cluster_eis.append(ei_k)

    sim = compare_eis(cluster_eis)
    if plot_diagnostic == True:
        print("Similarity before merge:\n")
        print(sim)

    if not merge:
        output = []
        for k in cluster_ids:
            inds = cluster_spike_indices[k]
            ei_cluster = np.mean(snips[:, :, inds], axis=2)
            ei_cluster -= ei_cluster[:, :5].mean(axis=1, keepdims=True)
            output.append({
                'inds': inds,
                'ei': ei_cluster,
                'channels': selected_channels
            })
        if return_debug:
            return output, pcs, labels, sim
        else:
            return output


    G = nx.Graph()
    G.add_nodes_from(range(len(cluster_ids)))
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            if sim[i, j] >= sim_threshold:
                G.add_edge(i, j)

    merged_groups = list(nx.connected_components(G))
    merged_clusters = []

    for group in merged_groups:
        group = sorted(list(group))
        all_inds = np.concatenate([cluster_spike_indices[cluster_ids[i]] for i in group])
        merged_clusters.append(np.sort(all_inds))

    output = []
    for inds in merged_clusters:
        ei_cluster = np.mean(snips[:, :, inds], axis=2)
        ei_cluster -= ei_cluster[:, :5].mean(axis=1, keepdims=True)
        output.append({
            'inds': inds,
            'ei': ei_cluster,
            'channels': selected_channels
        })

    if return_debug:
        return output, pcs, labels, sim, cluster_eis
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


def ei_pursuit(
    dat_path: str,
    spikes: np.ndarray,
    ei_template: np.ndarray,
    dtype: str = 'int16',
    total_samples: int = 10_000_000,
    save_prefix: str = '/Volumes/Lab/Users/alexth/axolotl/ei_scan_unit0',
    block_size: int = None,
    baseline_start_sample: int = 0,
    alignment_offset: int = 20,
    fit_percentile: float = 40,
    sigma_thresh: float = 5.0,
    channel_major: bool = True,
    return_debug: bool = False   # ← new
) -> Union[np.ndarray, Tuple]:
    """
    Run EI template matching using multi-GPU scan, apply thresholding, and return spike times.

    Parameters:
        dat_path: Path to raw .dat file
        spikes: Initial spike times (absolute sample indices)
        ei_template: [channels x timepoints] EI to use as matching template
        dtype: Data type of the file (e.g., 'int16')
        total_samples: Length of dataset to scan
        save_prefix: Where to save temp .npy files

    Returns:
        final_spike_times: np.ndarray of accepted spike times (in samples)
    """
    ei_template_path = f"{save_prefix}_template.mat"
    sio.savemat(ei_template_path, {'ei_template': ei_template.astype(np.float32)})

    mean_score, max_score, valid_score, selected_channels, _ = run_multi_gpu_ei_scan(
        ei_mat_path=ei_template_path,
        dat_path=dat_path,
        total_samples=total_samples,
        save_prefix=save_prefix,
        dtype=dtype,
        block_size=block_size,
        baseline_start_sample=baseline_start_sample,
        channel_major=channel_major
    )

    adjusted_selected_inds = spikes - alignment_offset
    adjusted_selected_inds = adjusted_selected_inds[(adjusted_selected_inds >= 0) & (adjusted_selected_inds < len(mean_score))]

    def fit_threshold(scores):
        cutoff = np.percentile(scores, fit_percentile)
        left_tail = scores[scores <= cutoff]
        mu, sigma = norm.fit(left_tail)
        return mu - sigma_thresh * sigma

    mean_scores = mean_score[adjusted_selected_inds]
    valid_scores = valid_score[adjusted_selected_inds]

    mean_threshold = fit_threshold(mean_scores)
    valid_threshold = fit_threshold(valid_scores)

    # import matplotlib.pyplot as plt

    # center = 2666246
    # window_for_plot = 60
    # x = np.arange(center - window_for_plot, center + window_for_plot + 1)
    # y = mean_score[center - window_for_plot : center + window_for_plot + 1]

    # plt.figure(figsize=(10, 3))
    # plt.plot(x, y, marker='o')
    # plt.axvline(center, color='red', linestyle='--', label='Spike')
    # plt.title(f"Mean score around spike {center}")
    # plt.xlabel("Sample")
    # plt.ylabel("Score")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


    # import matplotlib.pyplot as plt

    # # --- Histogram for mean scores ---
    # plt.figure(figsize=(12, 5))

    # plt.subplot(1, 2, 1)
    # #plt.hist(mean_score, bins=200, alpha=0.4, label='All scores', color='gray', density=True)
    # plt.hist(mean_scores[:1400], bins=50, alpha=0.6, label='True spikes', color='blue', density=True)
    # plt.axvline(mean_threshold, color='red', linestyle='--', label='Mean threshold')
    # plt.title("Mean Score Distribution")
    # plt.xlabel("Score")
    # plt.ylabel("Density")
    # plt.legend()
    # plt.grid(True)

    # # --- Histogram for valid scores ---
    # plt.subplot(1, 2, 2)
    # #plt.hist(valid_score, bins=100, alpha=0.4, label='All valid counts', color='gray', density=True)
    # plt.hist(valid_scores[:1400], bins=50, alpha=0.6, label='True spikes', color='blue', density=True)
    # plt.axvline(valid_threshold, color='red', linestyle='--', label='Valid threshold')
    # plt.title("Valid Channel Count Distribution")
    # plt.xlabel("Valid Channels")
    # plt.ylabel("Density")
    # plt.legend()
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()

    peaks = argrelextrema(mean_score, np.greater_equal, order=1)[0]
    valid_inds = peaks[(mean_score[peaks] > mean_threshold) & (valid_score[peaks] > valid_threshold)]


    # accept at most 20% more than before...
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


def select_cluster_by_ei_similarity(
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


def subtract_unit_all_channels(
    spike_times: np.ndarray,
    ei: np.ndarray,
    dat_path: str,
    ei_positions: np.ndarray,
    start_sample: int,
    segment_length: int,
    dtype: str,
    n_channels: int,
    window: tuple = (-20, 60),
    fit_offsets: tuple = (-5, 10),
    p2p_threshold: float = 15.0
) -> dict:
    """
    Perform template subtraction for all EI-relevant channels of a unit over a given segment.

    Parameters:
        spike_times: np.ndarray of spike sample indices (absolute times)
        ei: EI waveform [n_channels x T]
        dat_path: path to the raw data file (modified or original)
        ei_positions: electrode spatial positions [n_channels x 2]
        start_sample: first sample index of the segment
        segment_length: number of samples in the segment
        dtype: data type of the raw file
        n_channels: number of channels
        window: waveform window around each spike (e.g. (-20, 60))
        fit_offsets: tuple defining fitting range relative to peak (e.g. (-5, 10))
        p2p_threshold: min peak-to-peak to include a channel

    Returns:
        all_channel_fit_params: dict[channel_idx] = array of fit parameters per spike
    """
    ei_p2p = ei.max(axis=1) - ei.min(axis=1)
    selected_channels = np.where(ei_p2p > p2p_threshold)[0]
    selected_channels = selected_channels[np.argsort(ei_p2p[selected_channels])[::-1]]

    tree = KDTree(ei_positions)

    valid_spikes = [s for s in spike_times if (s + window[0] >= start_sample) and (s + window[1] < start_sample + segment_length)]
    valid_spikes = np.array(valid_spikes)

    all_channel_fit_params = {ch: None for ch in selected_channels}

    for ch_idx in selected_channels:
        ch_coord = ei_positions[ch_idx]
        neighbor_dists, neighbor_idxs = tree.query(ch_coord, k=6)
        neighbor_idxs = neighbor_idxs[neighbor_dists > 0]

        valid_neighbors = [idx for idx in neighbor_idxs if idx in all_channel_fit_params and all_channel_fit_params[idx] is not None]

        if len(valid_neighbors) == 0:
            fallback_params = None
        else:
            fallback_params = {}
            for i, spike_time in enumerate(valid_spikes):
                per_spike_vals = [all_channel_fit_params[idx][i] for idx in valid_neighbors]
                fallback_params[spike_time] = np.mean(per_spike_vals, axis=0)

        channel_params = run_template_subtraction_on_channel.run_template_subtraction_on_channel(
            channel_idx=ch_idx,
            final_spike_times=spike_times,
            ei_waveform=ei[ch_idx],
            dat_path=dat_path,
            start_sample=start_sample,
            segment_length=segment_length,
            dtype=dtype,
            n_channels=n_channels,
            window=window,
            fit_offsets=fit_offsets,
            fallback_fit_params=fallback_params,
            diagnostic_plot_spikes=None
        )
        all_channel_fit_params[ch_idx] = channel_params

    return all_channel_fit_params




def run_template_subtraction_on_channel_accumulate(
    channel_idx,
    final_spike_times,
    ei_waveform,
    dat_path_chmajor,
    total_samples,
    dtype,
    window,
    fit_offsets,
    max_chunk_len=100_000,
    fallback_fit_params=None,
    diagnostic_plot_spikes=None
):
    """
    Subtracts warped template fits from a channel-major data file, accumulating residuals.

    Returns:
    - fit_params_per_spike: list of [A, w, delta] per spike
    - snippets_to_write: (n_spikes, snippet_len) int16 array of residuals
    - write_locations: array of start times (sample indices) for each residual
    """

    pre, post = window
    snip_len = post - pre + 1

    ei_trace = ei_waveform.copy()
    ei_trace -= np.mean(ei_trace[:5])
    t_template = np.arange(len(ei_trace))
    t_peak_template = np.argmin(ei_trace)

    valid_spikes = [s for s in final_spike_times if (s + pre >= 0) and (s + post < total_samples)]
    valid_spikes = np.array(valid_spikes)

    snippets_to_write = []
    write_locations = []
    fit_params_per_spike = []

    with open(dat_path_chmajor, 'rb') as f:
        spike_idx = 0
        while spike_idx < len(valid_spikes):
            s_start = valid_spikes[spike_idx]
            chunk_start = s_start + pre
            chunk_end = min(chunk_start + max_chunk_len, total_samples)

            chunk_spikes = []
            while (spike_idx < len(valid_spikes)) and (valid_spikes[spike_idx] + post < chunk_end):
                chunk_spikes.append(valid_spikes[spike_idx])
                spike_idx += 1

            if len(chunk_spikes) == 0:
                spike_idx += 1
                continue

            chunk_spikes = np.array(chunk_spikes)

            offset = (channel_idx * total_samples + chunk_start) * np.dtype(dtype).itemsize
            n_time = chunk_end - chunk_start
            f.seek(offset)
            chunk_data = np.fromfile(f, dtype=dtype, count=n_time).astype(np.float32)

            baseline_offset = np.mean(chunk_data)
            chunk_data -= baseline_offset

            t = np.arange(snip_len)
            t0 = max(t_peak_template + fit_offsets[0], 0)
            t1 = min(t_peak_template + fit_offsets[1], snip_len - 1)

            A_vals = np.linspace(0.8, 1.2, 5)
            w_vals = np.linspace(0.95, 1.05, 5)
            d_vals = np.linspace(-0.5, 0.5, 5)

            for s in chunk_spikes:
                t_start = s + pre
                local_offset = t_start - chunk_start
                raw_snip = chunk_data[local_offset:local_offset + snip_len]

                best_err = np.inf
                best_fit = None
                best_params = None

                for A in A_vals:
                    for w in w_vals:
                        for delta in d_vals:
                            t_shifted = (t - t_peak_template - delta) / w + t_peak_template
                            t_shifted = np.clip(t_shifted, 0, len(ei_trace) - 1)
                            warped = interp1d(t_template, ei_trace, kind='linear', bounds_error=False)(t_shifted)
                            fit = A * warped
                            err = np.sum((raw_snip[t0:t1 + 1] - fit[t0:t1 + 1]) ** 2)
                            if err < best_err:
                                best_err = err
                                best_fit = fit
                                best_params = (A, w, delta)

                A_fit, w_fit, delta_fit = best_params
                template_fit = best_fit

                if fallback_fit_params is not None and (best_params is None or best_err > 1e6) and s in fallback_fit_params:
                    A_fit, w_fit, delta_fit = fallback_fit_params[s]
                    template_fit = A_fit * ei_trace

                residual = raw_snip - template_fit
                residual += baseline_offset
                residual_clipped = np.clip(residual, -32768, 32767).astype(np.int16)

                if diagnostic_plot_spikes is not None and s in set(diagnostic_plot_spikes):
                    plt.figure(figsize=(10, 3))
                    plt.plot(raw_snip, label='Raw', color='black')
                    plt.plot(template_fit, label='Fit', color='red')
                    plt.plot(residual, label='Residual', color='green')
                    plt.axvline(t_peak_template, linestyle='--', color='gray')
                    plt.title(f"Channel {channel_idx}, Spike @ {s}")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()

                snippets_to_write.append(residual_clipped)
                write_locations.append(t_start)
                fit_params_per_spike.append([A_fit, w_fit, delta_fit])

    return fit_params_per_spike, np.stack(snippets_to_write), np.array(write_locations)




def subtract_pca_cluster_means(snippets, baselines, spike_times, segment_len=100_000, n_clusters=5, offset_window=(-5,10)):
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




def apply_residuals_to_channel_major(
    dat_path_chmajor,
    residual_snips_per_channel,
    write_locs,
    selected_channels,
    total_samples,
    dtype=np.int16,
    n_channels=512
):
    """
    Subtracts residual waveforms from a channel-major .dat file in-place.

    Parameters:
    - dat_path_chmajor: Path to channel-major binary file
    - residual_snips_per_channel: dict {channel_idx: (n_spikes, snip_len) int16 array}
    - write_locs: array of sample indices
    - selected_channels: list of channels to process
    - total_samples: total number of timepoints in recording
    - dtype: data type (default np.int16)

    Returns:
    - None
    """

    # Use correct shape based on full dataset, not subset of processed channels
    data = np.memmap(
        dat_path_chmajor,
        dtype=dtype,
        mode='r+',
        shape=(n_channels, total_samples)
    )

    for ch_idx in selected_channels:
        residuals = residual_snips_per_channel[ch_idx]

        if residuals.shape[0] != len(write_locs):
            raise ValueError(f"Mismatch in spikes and write_locs for channel {ch_idx}")

        for i, (snip, loc) in enumerate(zip(residuals, write_locs)):
            end = loc + snip.shape[0]
            if end > total_samples:
                print(f"    Skipping spike {i} (ends at {end}, beyond total_samples)")
                continue

            data[ch_idx, loc:end] = snip

    data.flush()
    del data



import numpy as np
from scipy.signal import correlate

def estimate_lags_by_xcorr(snippets: np.ndarray, peak_channel_idx: int, window: tuple = (-5, 10), max_lag: int = 3) -> np.ndarray:
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
        # import matplotlib.pyplot as plt

        # plt.figure(figsize=(10, 4))
        # plt.plot(waveform, label="Mean waveform")
        # plt.axvline(peak_idx, color='red', linestyle='--', label=f'Peak @ {peak_idx}')
        # plt.axvspan(win_start, win_end, color='orange', alpha=0.3, label=f'Window ({win_start}-{win_end})')
        # plt.title("Mean waveform on peak channel")
        # plt.xlabel("Sample")
        # plt.ylabel("Amplitude")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

        # Raise exception with details
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


def suppress_artifacts_in_dat(
    dat_path_chmajor,
    artifact_locs,
    n_channels,
    total_samples,
    dtype=np.int16,
    suppress_window=(-20, 50),
    baseline_window=(-100, -20, 50, 150)
):
    """
    Suppress artifacts by flattening the trace around each artifact location.

    Parameters:
    - dat_path_chmajor: Path to the channel-major .dat file
    - artifact_locs: Dictionary of {channel_index: list of sample indices (int)}
    - n_channels: Total number of channels
    - total_samples: Total number of samples in the file
    - dtype: Data type of the file (default: np.int16)
    - suppress_window: Tuple (start, end) relative to artifact sample for replacement
    - baseline_window: Tuple (pre_start, pre_end, post_start, post_end) relative to artifact

    Returns:
    - None. The file is modified in-place.
    """

    data = np.memmap(dat_path_chmajor, dtype=dtype, mode='r+', shape=(n_channels, total_samples))

    suppress_start, suppress_end = suppress_window
    pre_start_rel, pre_end_rel, post_start_rel, post_end_rel = baseline_window

    for ch, locs in artifact_locs.items():
        for s in locs:
            start_fill = s + suppress_start
            end_fill = s + suppress_end
            pre_start = s + pre_start_rel
            pre_end = s + pre_end_rel
            post_start = s + post_start_rel
            post_end = s + post_end_rel

            if pre_start < 0 or post_end > total_samples:
                continue  # skip if out of bounds

            pre = data[ch, pre_start:pre_end]
            post = data[ch, post_start:post_end]
            flat_value = int(np.round(np.mean(np.concatenate([pre, post]))))

            data[ch, start_fill:end_fill] = flat_value

    data.flush()



def clipped_hist(ax, scores, threshold, title, bins=100):
    scores = np.asarray(scores)
    valid = scores[np.isfinite(scores)]

    if valid.size == 0:
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


def plot_unit_diagnostics(
    output_path: str,
    unit_id: int,
    pcs_pre: np.ndarray,
    labels_pre: np.ndarray,
    sim_matrix_pre: np.ndarray,
    cluster_eis_pre: np.ndarray,
    spikes_for_plot_pre: np.ndarray,
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
    tb = Table(ax2, bbox=[0.2, 0.2, 0.8, 0.8])
    n = sim_matrix_pre.shape[0]
    for i in range(n):
        for j in range(n):
            tb.add_cell(i, j, 1/n, 1/n, text=f"{sim_matrix_pre[i, j]:.2f}", loc='center')
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

    ei_row_pre.set_title(f"Cluster EIs; spikes chosen {len(mean_scores_at_spikes)}")

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
    tb = Table(ax5a, bbox=[0.2, 0.2, 0.6, 0.6])
    n = sim_matrix_post.shape[0]
    for i in range(n):
        for j in range(n):
            tb.add_cell(i, j, 1/n, 1/n, text=f"{sim_matrix_post[i, j]:.2f}", loc='center')
    for i in range(n):
        tb.add_cell(i, -1, 1/n, 1/n, text=str(i), loc='right', edgecolor='none')
        tb.add_cell(-1, i, 1/n, 1/n, text=str(i), loc='center', edgecolor='none')
    ax5a.add_table(tb)
    ax5a.axis('off')


    s0 = spikes_for_plot_post[labels_post == 0]
    s1 = spikes_for_plot_post[labels_post == 1]
    print(len(s0))
    print(len(s1))

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

    ei_row_post.set_title("Cluster EIs")

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
    fig.savefig(os.path.join(output_path, f"unit_{unit_id:03d}_diagnostics.png"), dpi=150)
    plt.close(fig)
