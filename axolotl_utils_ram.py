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
from scipy.signal import correlate

from scipy.stats import gaussian_kde
import h5py
from scipy.optimize import nnls


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


def subtract_segment_baselines_int16(raw_data: np.ndarray,
                                     baselines_f32: np.ndarray,
                                     segment_len: int = 100_000) -> None:
    """
    In-place baseline removal for int16 raw traces.

    Parameters
    ----------
    raw_data      : [T, C]  int16   – entire recording in RAM
    baselines_f32 : [C, n_segments] float32 – from compute_baselines_int16_deriv_robust
    segment_len   : int               – same value that was passed to the baseline routine

    Returns
    -------
    None   (raw_data is modified in place)
    """

    T, C = raw_data.shape
    C_b, n_seg = baselines_f32.shape
    if C_b != C:
        raise ValueError("Channel count mismatch between raw_data and baselines")

    # Quantise baselines once; cost ≈ 1 kB
    baselines_i16 = np.rint(baselines_f32).astype(np.int16)

    for seg_idx in range(n_seg):
        start = seg_idx * segment_len
        end   = min(start + segment_len, T)          # handle last partial segment

        # Broadcast-subtract:  [end-start, C]  -=  [C]
        raw_data[start:end, :] -= baselines_i16[:, seg_idx]

    # done – raw_data now baseline-subtracted, still int16

def plot_array(ei_positions):
    plt.figure(figsize=(12,4))

    for i in range(ei_positions.shape[0]):
        x_pos = ei_positions[i, 0]
        y_pos = ei_positions[i, 1]
        plt.text(x_pos, y_pos, str(i), fontsize=8, ha='center', va='center')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('EI positions with channel IDs')
    plt.grid(True)

    # Safe axis limits to prevent auto-scaling to something weird
    plt.xlim(-1000, 1000)
    plt.ylim(-500, 500)

    plt.tight_layout()
    plt.show()

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
    total_samples_to_read: int = 10_000_000,
    refractory: int = 30,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Returns:
        threshold : float
        spikes : array of int, selected spike times passing threshold
        next_spikes : array of int, top 500 largest peaks that were excluded
    """

    trace_f = raw_data[:total_samples_to_read, ref_channel].astype(np.float32)

    # -- Step 1: Collect negative peaks --
    neg_peaks, _ = find_peaks(-trace_f, distance=2 * refractory)
    if not len(neg_peaks):
        return 0.0, np.empty(0, dtype=int), np.empty(0, dtype=int)

    peak_vals = trace_f[neg_peaks]  # negative values
    hist, edges = np.histogram(peak_vals, bins=100)
    centers = (edges[:-1] + edges[1:]) / 2

    peak_idx, _ = find_peaks(hist)
    valley_idx, _ = find_peaks(-hist)

    # -- Step 2: Valley search --
    DEPTH_RATIO = 0.25
    NEED_200 = 200
    NEED_50K = 20_000
    threshold = None

    if len(peak_idx):
        noise_peak = peak_idx[np.argmax(hist[peak_idx])]
        cand_valleys = valley_idx[valley_idx < noise_peak]

        for v in cand_valleys:
            left_peaks = peak_idx[peak_idx < v]
            right_peaks = peak_idx[peak_idx > v]
            if not len(left_peaks):
                continue

            left_cnt = hist[left_peaks[-1]]
            right_cnt = hist[right_peaks[0]] if len(right_peaks) else left_cnt
            valley_ok = hist[v] < DEPTH_RATIO * left_cnt

            spikes_left = hist[:v].sum()
            if (valley_ok and spikes_left >= NEED_200) or (spikes_left >= NEED_50K):
                threshold = centers[v]
                break

    # -- Step 3: Fallback threshold --
    if threshold is None:
        amps_sorted = np.sort(peak_vals)
        k = 4_999
        threshold = amps_sorted[k] if len(amps_sorted) > k else amps_sorted[-1]

    # -- Step 4: Threshold-crossing --
    below = trace_f < threshold
    down = np.where(~below[:-1] & below[1:])[0] + 1
    up = np.where(below[:-1] & ~below[1:])[0] + 1

    i = j = 0
    windows = []
    while i < len(down) and j < len(up):
        if up[j] <= down[i]:
            j += 1
            continue
        windows.append((down[i], up[j]))
        i += 1
        j += 1

    # ------------------------------------------------------------
    # 5.  Pick minima inside windows, enforce refractory
    # ------------------------------------------------------------
    spikes = []
    last = -np.inf
    all_spike_candidates = []

    for a, b in windows:
        if b - a < 1:
            continue
        idx = np.argmin(trace_f[a:b]) + a
        amp = -trace_f[idx]
        all_spike_candidates.append((amp, idx))
        if idx - last > refractory:
            spikes.append(idx)
            last = idx

    spikes = np.array(spikes, dtype=int)

    # ------------------------------------------------------------
    # 6.  Cap at 50 000 spikes
    # ------------------------------------------------------------
    MAX_SPIKES = 50_000
    if len(spikes) > MAX_SPIKES:
        amps = -trace_f[spikes]
        keep = np.argsort(amps)[-MAX_SPIKES:]
        spikes = np.sort(spikes[keep])

    # -- Step 7: sub-threshold peaks ------------------------------------
    # NB: threshold is still NEGATIVE at this point.
    sub_mask   = peak_vals >= threshold               # peaks that did NOT cross threshold
    sub_idx    = neg_peaks[sub_mask]                  # their sample indices
    sub_amps   = -trace_f[sub_idx]                    # positive amplitudes (< -threshold)

    if sub_amps.size:
        order       = np.argsort(sub_amps)[::-1]      # descending by amplitude
        keep_n      = min(500, order.size)
        next_spikes = sub_idx[order[:keep_n]].astype(int)
    else:
        next_spikes = np.empty(0, dtype=int)



    # Plot
    # plt.figure(figsize=(6,4))
    # plt.bar(centers, hist, width=edges[1]-edges[0], alpha=0.7)
    # plt.scatter(centers[peak_idx], hist[peak_idx], color='red', label='Peaks')
    # plt.scatter(centers[valley_idx], hist[valley_idx], color='green', label='Valleys')
    # plt.axvline(threshold, color='purple', linestyle='--', label=f'Threshold {threshold:.2f} µV')

    # plt.xlabel("Amplitude at negative peaks (µV)")
    # plt.ylabel("Count")
    # plt.title("Histogram with valleys, peaks, and threshold")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    threshold = -threshold

    # print(threshold)
    # print(len(spikes))

    return threshold, spikes, next_spikes




def load_units_from_h5(h5_path):
    """
    Load spike_times, ei, and selected_channels for all units in the file.

    Returns:
        units: dict of dicts
            units[unit_id] = {
                'spike_times': np.ndarray,
                'ei': np.ndarray,
                'selected_channels': np.ndarray,
                'peak_channel': int
            }
    """
    units = {}

    with h5py.File(h5_path, 'r') as h5:
        for unit_name in h5.keys():
            group = h5[unit_name]
            unit_id = int(unit_name.split('_')[-1])
            
            spike_times = group['spike_times'][()]
            ei = group['ei'][()]
            selected_channels = group['selected_channels'][()]
            peak_channel = group.attrs['peak_channel']
            
            units[unit_id] = {
                'spike_times': spike_times,
                'ei': ei,
                'selected_channels': selected_channels,
                'peak_channel': peak_channel
            }

    return units



def candidate_pairs_simple(p2p, thr=0.1):
    """
    Find pairs of units that share channels above threshold, 
    but only compare units with different dominant channels.
    
    Parameters:
    - p2p: [units, channels] peak-to-peak amplitude
    - thr: threshold fraction (default 0.1)
    
    Returns:
    - pairs: list of (unit_a, unit_b, shared_channels)
    """
    U, C = p2p.shape
    
    p2p_max = p2p.max(axis=1)  # [units]
    p2p_max_ch = p2p.argmax(axis=1)  # [units]
    
    pairs = []
    
    for a in range(U):
        for b in range(a + 1, U):
            if p2p_max_ch[a] == p2p_max_ch[b]:
                continue  # skip pairs with same dominant channel
            
            shared = np.where(
                (p2p[a] >= thr * p2p_max[a]) &
                (p2p[b] >= thr * p2p_max[b])
            )[0]
            
            if shared.size:
                pairs.append((a, b, shared))
    
    return pairs

def find_collisions_1ch(tA, tB, pkA, pkB, delta=30):
    """Return index pairs (i,j) where corrected times ≤delta."""
    # arrival time series
    tA_corr = tA + pkA
    tB_corr = tB + pkB
    # two-pointer intersect
    i = j = 0
    hitsA, hitsB = [], []
    while i < len(tA_corr) and j < len(tB_corr):
        d = tB_corr[j] - tA_corr[i]
        if abs(d) <= delta:
            hitsA.append(i); hitsB.append(j)
            i += 1; j += 1
        elif d > 0:
            i += 1
        else:
            j += 1
    return np.array(hitsA), np.array(hitsB)

def find_solo_1ch(tA, tB, pkA, pkB, delta=60):
    """
    Return indices of tA and tB that are at least delta samples away
    from spikes of the other cell (on channel C corrected times).
    """
    tA_corr = tA + pkA
    tB_corr = tB + pkB

    i = j = 0
    soloA = []
    soloB = []

    while i < len(tA_corr) and j < len(tB_corr):
        d = tB_corr[j] - tA_corr[i]
        if abs(d) <= delta:
            # Too close → not solo → skip both (or move on appropriately)
            # We move the one that's earlier in time
            if d > 0:
                i += 1
            else:
                j += 1
        elif d > delta:
            # tA[i] is solo (B too far ahead)
            soloA.append(i)
            i += 1
        else:  # d < -delta
            # tB[j] is solo (A too far ahead)
            soloB.append(j)
            j += 1

    # Any remaining spikes in A or B are solo (no more of the other left to conflict)
    while i < len(tA_corr):
        soloA.append(i)
        i += 1
    while j < len(tB_corr):
        soloB.append(j)
        j += 1

    return np.array(soloA), np.array(soloB)


def extract_snippets_fast_ram(
        raw_data: np.ndarray,           # [T, C]  int16
        spike_times: np.ndarray,        # [N]     int64 / int32
        window: tuple[int, int],        # (pre, post)  e.g. (-20, 40)
        selected_channels: np.ndarray   # [K]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return snippets [K, L, N] (float32) and the spike times that were
    inside bounds.  No Python loop, single memory copy.
    """
    pre, post = window
    win_len = post - pre + 1
    total_samples, _ = raw_data.shape

    m = (spike_times + pre >= 0) & (spike_times + post < total_samples)
    valid_times = spike_times[m]
    N = valid_times.size
    K = len(selected_channels)
    if N == 0:
        return np.empty((K, win_len, 0), np.float32), valid_times

    offsets = np.arange(pre, post + 1, dtype=np.int64)
    rows_idx = (valid_times[:, None] + offsets[None, :]).reshape(-1)  # [N*L]
    snips = raw_data[rows_idx[:, None], selected_channels]  # [N*L, K]
    snips = snips.astype(np.float32, copy=False).reshape(N, win_len, K).transpose(2,1,0)  # [K,L,N]

    return snips, valid_times


from scipy.optimize import lsq_linear

def fit_two_templates_bounded(y, TA, TB, lower=0.75, upper=1.25):

    X = np.vstack((TA, TB)).T  # shape (80, 2)
    y64 = y.astype(np.float64)
    result = lsq_linear(X, y64, bounds=(lower, upper), lsmr_tol='auto', verbose=0)
    alpha, beta = result.x
    rnorm = np.linalg.norm(y - X @ result.x)
    return alpha, beta, rnorm

def fit_two_templates(y, TA, TB_shift):
    X = np.vstack((TA, TB_shift)).T  # L×2 matrix: columns are templates
    coeffs, rnorm = nnls(X, y)
    return coeffs[0], coeffs[1], rnorm


import numpy as np
from typing import Tuple

def build_channel_weights_twoEI(
        snipsA_good: np.ndarray,    # [C, T, N_A]  core spikes of unit A
        snipsB_good: np.ndarray,    # [C, T, N_B]  core spikes of unit B
        eiA: np.ndarray,            # [C, T]       template A
        eiB: np.ndarray,            # [C, T]       template B
        p2p_thresh: float = 30.0,   # ADC threshold for “significant” channels
        rel_mask: float = 0.01      # |t| >= rel_mask * P2P keeps a sample
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    weights        : float32 [C_selected]  w_c = max(P2P_A,P2P_B) / σ_c^2
    selected_chans : int32   [C_selected]  channel indices kept
    """
    C, T = eiA.shape
    # ------------------------------------------------------------------ #
    # 1. identify channels to keep (either EI ≥ p2p_thresh)
    p2p_A = eiA.ptp(axis=1)
    p2p_B = eiB.ptp(axis=1)
    keep  = (p2p_A >= p2p_thresh) | (p2p_B >= p2p_thresh)
    sel   = np.where(keep)[0]
    if sel.size == 0:
        raise ValueError("No channels pass p2p_thresh")

    # ------------------------------------------------------------------ #
    # 2. build the *same* boolean mask the scorer will use
    #    use the template that dominates this channel (larger P2P)
    mask = np.zeros((sel.size, T), dtype=np.bool_)
    for k, c in enumerate(sel):
        if p2p_A[c] >= p2p_B[c]:
            tmpl = eiA[c]
            p2p  = p2p_A[c]
        else:
            tmpl = eiB[c]
            p2p  = p2p_B[c]
        mask[k] = np.abs(tmpl) >= rel_mask * p2p          # [T]

    # ------------------------------------------------------------------ #
    # 3. gather residuals for variance σ²  (no amplitude scaling)
    #    use spikes of the unit that dominates this channel
    sigma2 = np.empty(sel.size, dtype=np.float32)

    for k, c in enumerate(sel):
        if p2p_A[c] >= p2p_B[c]:
            # dominant unit A on this channel
            res = snipsA_good[c] - eiA[c][:, None]        # [T, N_A]
        else:
            res = snipsB_good[c] - eiB[c][:, None]        # [T, N_B]
        # apply same mask: only keep time-points that will matter later
        res_masked = res[mask[k]]
        # robust σ via MAD
        mad   = np.median(np.abs(res_masked))
        sigma = mad / 0.6745 + 1e-6                      # avoid /0
        sigma2[k] = sigma * sigma

    # ------------------------------------------------------------------ #
    # 4. final weights
    p2p_sel = np.maximum(p2p_A[sel], p2p_B[sel]).astype(np.float32)
    weights = p2p_sel / sigma2
    # normalise so mean weight = 1 (optional but convenient)
    weights /= weights.mean()

    return weights.astype(np.float32), sel.astype(np.int32)



from numba import njit, prange


# ------------------------------------------------------------------------
# helper: roll array by lag (positive => right shift), zero-pad the vacated end
@njit(fastmath=True, cache=True)
def _roll_pad(arr, lag):
    C, T = arr.shape
    out  = np.zeros_like(arr)
    if lag > 0:
        out[:, lag:] = arr[:, :T-lag]
    elif lag < 0:
        out[:, :T+lag] = arr[:, -lag:]
    else:
        out[:] = arr
    return out
# ------------------------------------------------------------------------

from numba import njit, prange
import numpy as np

AMP_FACTOR = 2.0     # skip channel if spike P2P > 3× template P2P
MIN_SAMP   = 3       # need ≥3 masked samples on a channel to keep it


@njit(parallel=True, fastmath=True, cache=True)
def _compute_rms(snips, rolled_templates, rolled_masks,
                 weights, p2p_template,p2p_global,
                 residuals, bestlags):
    """
    Parameters (same order/shape you already call with)
    ---------------------------------------------------
    snips            : float32 [C, T, N]
    rolled_templates : float32 [nT, nL, C, T]
    rolled_masks     : bool    [nT, nL, C, T]
    weights          : float32 [C]            (mean≈1)
    p2p_template     : float32 [nT, C]        max-abs of each template
    residuals        : float32 [nT, N]        (output)
    bestlags         : int16   [nT, N]        (lag index; caller converts)
    """
    nT, nL, C, T = rolled_templates.shape
    N            = snips.shape[2]

    for t in prange(nT):
        p2p_c = p2p_template[t]                 # view [C]

        for s in range(N):
            best_r  = 1e20
            best_li = 0

            for li in range(nL):
                tmpl = rolled_templates[t, li]
                mask = rolled_masks[t, li]

                err_tot = 0.0     # weighted sum of channel variances
                w_sum   = 0.0     # sum of weights actually used

                # ------------ per channel -----------------------------------
                for c in range(C):
                    max_abs = 0.0
                    sum_e   = 0.0
                    sum_e2  = 0.0
                    n_mask  = 0

                    for k in range(T):
                        if mask[c, k]:
                            val     = snips[c, k, s]
                            diff    = val - tmpl[c, k]
                            sum_e  += diff
                            sum_e2 += diff * diff
                            n_mask += 1
                            if abs(val) > max_abs:
                                max_abs = abs(val)

                    # channel–level guards
                    if n_mask < MIN_SAMP:
                        continue
                    if max_abs > AMP_FACTOR * p2p_global[c]:
                        continue

                    var_c = (sum_e2 - (sum_e * sum_e) / n_mask) / n_mask
                    if var_c < 0.0:          # numerical round-off
                        var_c = 0.0

                    w      = weights[c]
                    err_tot += w * var_c
                    w_sum  += w

                if w_sum == 0.0:
                    continue

                rms = np.sqrt(err_tot / w_sum)
                if rms < best_r:
                    best_r  = rms
                    best_li = li

            residuals[t, s] = best_r
            bestlags[t, s]  = best_li


def compute_residuals_with_lag(snips, templates, weights, max_lag=13, rel_thresh=0.01):
    """
    RMS residual between every spike and every template, allowing ±max_lag
    alignment.  Masks shift together with the template to avoid artefacts.
    """
    snips = snips.astype(np.float32, copy=False)
    temps = np.asarray(templates, dtype=np.float32)
    n_temp, C, T = temps.shape
    p2p_template = np.max(np.abs(temps), axis=2).astype(np.float32)  # [n_temp, C]

    _, _, N      = snips.shape

    lags      = np.arange(-max_lag, max_lag + 1, dtype=np.int16)
    n_lags    = lags.size

    p2p_global = np.max(p2p_template, axis=0).astype(np.float32)  # [C]

    # single unshifted mask (True where any template is “alive”)
    abs_max_over_templates = np.max(np.abs(templates), axis=0)        # [C,T]
    master_mask0 = abs_max_over_templates > (rel_thresh * p2p_global[:, None] + 1e-6)

    rolled_templates = np.empty((n_temp, n_lags, C, T), np.float32)
    rolled_masks     = np.empty((n_temp, n_lags, C, T), np.bool_)

    # for li, L in enumerate(lags):
    #     rolled_masks[li] = _roll_pad(master_mask0.astype(np.float32), L) > 0.5

    for t in range(n_temp):
        tmpl = templates[t]
        for li, L in enumerate(lags):
            rolled_templates[t, li] = _roll_pad(tmpl, L)
            rolled_masks[t, li]     = _roll_pad(master_mask0.astype(np.float32), L) > 0.5
    # # --------------------------------------------------------------------
    # # build rolled versions of template **and** its mask for every lag
    # # --------------------------------------------------------------------
    # rolled_templates = np.zeros((n_temp, n_lags, C, T), dtype=np.float32)
    # rolled_masks     = np.zeros((n_temp, n_lags, C, T), dtype=np.bool_)

    # for t in range(n_temp):
    #     tmpl = temps[t]
    #     p2p  = tmpl.ptp(axis=1)                    # [C]
    #     mask0 = np.abs(tmpl) > (rel_thresh * p2p[:, None] + 1e-6)

    #     for li, lag in enumerate(lags):
    #         rolled_templates[t, li] = _roll_pad(tmpl, lag)
    #         rolled_masks[t, li]     = _roll_pad(mask0.astype(np.float32), lag) > 0.5

    # outputs
    residuals = np.empty((n_temp, N), dtype=np.float32)
    bestlags  = np.empty((n_temp, N), dtype=np.int16)




    # heavy lifting
    _compute_rms(snips, rolled_templates, rolled_masks, weights, p2p_template, p2p_global, residuals, bestlags)

    # translate lag indices back to signed lag values
    bestlags[:] = lags[bestlags]
    return residuals, bestlags



import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d

def analyze_single_unit(spike_samples,
                        sampling_rate,
                        triggers_sec,
                        ei,
                        ei_positions,
                        lut=None,
                        sta_depth=20,
                        sta_offset=0,
                        sta_chunk_size=1000,
                        sta_refresh=2,
                        ei_scale=3,
                        ei_cutoff=0.05,
                        isi_max_ms=200,
                        sigma_ms=2500,
                        dt_ms=1000):


    if lut is None:
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

    spikes_sec = spike_samples / sampling_rate

    fig = plt.figure(figsize=(8, 10))
    gs = gridspec.GridSpec(5, 1, figure=fig)

    # Plot EI
    from plot_ei_python import plot_ei_python  # if not already imported
    ax_ei = fig.add_subplot(gs[0, 0])
    ax_ei.set_title(f"EI ({len(spike_samples)} spikes)", fontsize=10)
    plot_ei_python(ei, ei_positions, scale=ei_scale, cutoff=ei_cutoff,
                   pos_color='black', neg_color='red', ax=ax_ei, alpha=1)

    # ISI
    ax_isi = fig.add_subplot(gs[1, 0])
    if len(spikes_sec) > 1:
        isi = np.diff(spikes_sec)
        isi_max_s = isi_max_ms / 1000
        bins = np.arange(0, isi_max_s + 0.0005, 0.0005)
        hist, _ = np.histogram(isi, bins=bins)
        fractions = hist / hist.sum() if hist.sum() > 0 else np.zeros_like(hist)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax_isi.plot(bin_centers, fractions, color='blue')
        ax_isi.set_xlim(0, isi_max_s)
        ax_isi.set_ylim(0, np.max(fractions) * 1.1)
    else:
        ax_isi.text(0.5, 0.5, "Not enough spikes", ha='center', va='center')
        ax_isi.set_xlim(0, 0.2)
        ax_isi.set_ylim(0, 1)
    ax_isi.set_title("ISI (s)", fontsize=10)

    # Smoothed firing rate
    ax_rate = fig.add_subplot(gs[2, 0])
    dt = dt_ms / 1000
    sigma_samples = sigma_ms / dt_ms
    total_duration = spikes_sec.max() + 0.1 if len(spikes_sec) > 0 else 1.0
    time_vector = np.arange(0, total_duration, dt)
    counts, _ = np.histogram(spikes_sec, bins=np.append(time_vector, total_duration))
    rate = gaussian_filter1d(counts / dt, sigma=sigma_samples)
    ax_rate.plot(time_vector, rate, color='darkorange')
    ax_rate.set_title("Smoothed Firing Rate", fontsize=10)
    ax_rate.set_xlim(0, total_duration)
    ax_rate.set_ylabel("Hz")

    # STA
    from compute_sta_from_spikes import compute_sta_chunked
    sta = compute_sta_chunked(
        spikes_sec=spikes_sec,
        triggers_sec=triggers_sec,
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

    ax_tc = fig.add_subplot(gs[3, 0])
    ax_tc.plot(red_tc, color='red', label='R')
    ax_tc.plot(green_tc, color='green', label='G')
    ax_tc.plot(blue_tc, color='blue', label='B')
    ax_tc.set_title("STA Time Course (reversed)", fontsize=10)
    ax_tc.set_xlim(0, sta_depth - 1)
    ax_tc.set_xticks([0, sta_depth - 1])
    ax_tc.set_ylabel("Intensity")

    # STA frame at peak
    peak_frame = max_idx[3]
    rgb = sta[:, :, :, peak_frame]
    vmax = np.max(np.abs(sta)) * 2
    norm_rgb = rgb / vmax + 0.5
    norm_rgb = np.clip(norm_rgb, 0, 1)

    ax_sta = fig.add_subplot(gs[4, 0])
    ax_sta.imshow(norm_rgb.transpose(1, 0, 2), origin='upper')
    ax_sta.axis('off')
    ax_sta.set_title(f"STA Frame {peak_frame + 1}", fontsize=10)

    plt.tight_layout()
    plt.show()



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

    if len(all_residuals) == 0:
        return {'mean_residual': np.nan,
                'max_abs_residual': np.nan,
                'good_channels': [],
                'per_channel_residuals': [],
                'per_channel_cosine_sim': per_channel_cosine_sim,
                'fractional_shift': fractional_shift,
                'p2p_a': p2p_a}
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

from itertools import combinations

def cluster_separation_score(pcs, labels):
    unique_labels = np.unique(labels)
    scores = []

    for A, B in combinations(unique_labels, 2):
        pcs_A = pcs[labels == A]
        pcs_B = pcs[labels == B]

        mu_A = pcs_A.mean(axis=0)
        mu_B = pcs_B.mean(axis=0)

        d_AB = np.linalg.norm(mu_A - mu_B)

        std_A = pcs_A.std(axis=0).mean()
        std_B = pcs_B.std(axis=0).mean()
        spread = (std_A + std_B) / 2

        score = d_AB / (spread + 1e-8)
        scores.append({
            'pair': (A, B),
            'separation_score': score
        })

    return scores



def merge_similar_clusters_extra(
    snips,
    labels,
    max_lag      = 3,
    p2p_thresh   = 30.0,
    amp_thresh   = -20,
    cos_thresh   = 0.8,
    pcs2         = None,   # ← NEW  (N × 2) PC-space features, same order as labels
    sep_thresh   = 3.0     # ← NEW  veto if separation > this
):
    """
    Merge clusters whose EIs are highly similar, **unless** they are clearly
    separated in low-dimensional PC space (pcs2).

    pcs2 : array of shape (N_spikes, 2 or 3).  Usually the first two PCs.
            If None, no PC-space veto is applied.
    sep_thresh : clusters with separation_score > sep_thresh are *not* merged
                 even if EI cosine etc. pass.
    """

    # ---- basic bookkeeping ---------------------------------------------------
    cluster_ids          = sorted(np.unique(labels))
    cluster_spike_idx    = {k: np.where(labels == k)[0] for k in cluster_ids}
    n_clusters           = len(cluster_ids)
    id2idx               = {cid: i for i, cid in enumerate(cluster_ids)}

    # ---- pre-compute EI templates & per-channel variance ---------------------
    cluster_eis   = []
    cluster_vars  = []
    for k in cluster_ids:
        inds = cluster_spike_idx[k]
        ei_k = snips[:, :, inds].mean(axis=2)
        cluster_eis.append(ei_k)

        peak_idx  = np.argmin(ei_k, axis=1)
        var_ch    = np.array([
            np.var(snips[ch, max(0,i-1):i+2, inds]) if 1 <= i < ei_k.shape[1]-1 else 0.0
            for ch, i in enumerate(peak_idx)
        ])
        cluster_vars.append(var_ch)

    # ---- EI-similarity and bad-channel matrices (ASYMMETRIC) --------------------
    sim = np.eye(n_clusters)
    n_bad_ch = np.zeros((n_clusters, n_clusters), dtype=int)

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
                neg_inds = np.where(res_subset < amp_thresh)[0]

                sim[i, j] = weighted_cos_sim
                n_bad_ch[i, j] = len(neg_inds)

    # ---- optional PC-space separation matrix ---------------------------------
    if pcs2 is not None:
        sep = np.full((n_clusters, n_clusters), np.inf)
        pcs2 = np.asarray(pcs2)
        for i in range(n_clusters):
            inds_i = cluster_spike_idx[cluster_ids[i]]
            pcs_i  = pcs2[inds_i]
            mu_i   = pcs_i.mean(0)
            std_i  = pcs_i.std(0).mean()
            for j in range(i+1, n_clusters):
                inds_j = cluster_spike_idx[cluster_ids[j]]
                pcs_j  = pcs2[inds_j]
                mu_j   = pcs_j.mean(0)
                std_j  = pcs_j.std(0).mean()
                sep[i, j] = sep[j, i] = np.linalg.norm(mu_i - mu_j) / ((std_i + std_j)/2 + 1e-8)
    else:
        sep = None

    # ---- star-style merge with veto ------------------------------------------
    cluster_sizes  = {cid: len(cluster_spike_idx[cid]) for cid in cluster_ids}
    sorted_ids     = sorted(cluster_ids, key=lambda c: cluster_sizes[c], reverse=True)


    assigned       = set()
    merged_clusters = []

    for cid in sorted_ids:
        if cid in assigned:
            continue

        group = [cid]               # ← initialise
        assigned.add(cid)           # ← mark it used

        changed = True
        while changed:
            changed = False
            for other in sorted_ids:
                if other in assigned:
                    continue
                accept = False
                for existing in group:
                    i, j = id2idx[existing], id2idx[other]

                    # EI / bad-channel tests
                    sim_ok  = (
                        (sim[i,j] >= 0.95 and n_bad_ch[i,j] <= 6) or
                        (sim[i,j] >= 0.90 and n_bad_ch[i,j] <= 4) or
                        (sim[i,j] >= 0.80 and n_bad_ch[i,j] == 2) or
                        (sim[i,j] >= cos_thresh and n_bad_ch[i,j] == 0)
                    )

                    # NEW: PC-space veto
                    sep_ok  = (sep is None) or (sep[i,j] <= sep_thresh)

                    if sim_ok and sep_ok:
                        accept = True
                        break   # one existing member is enough

                if accept:
                    group.append(other)
                    assigned.add(other)
                    changed = True

        merged_spikes = np.concatenate([cluster_spike_idx[c] for c in group])
        merged_clusters.append(np.sort(merged_spikes))

    return merged_clusters, sim, n_bad_ch



def score_spikes_against_template(
        snips,            # [C, T, N] float32
        template,         # [C, T]     float32
        max_lag=3,
        p2p_thresh=30.0,
        amp_thresh=-20.0
):
    """
    Return per-spike similarity metrics w.r.t. a fixed EI template.
    """
    import numpy as np

    C, T, N = snips.shape
    p2p     = template.ptp(axis=1)               # [C]
    core_ch = np.where(p2p >= p2p_thresh)[0]     # indices
    # core_ch = core_ch[core_ch != ref_channel]
    w_ch    = p2p[core_ch]                       # weights

    # -------- pre-normalise template on core channels ----------
    T_core  = template[core_ch]                  # [Ccore, T]
    T_core  = T_core - T_core.mean(axis=1, keepdims=True)
    norm_T  = np.linalg.norm(T_core, axis=1) + 1e-12  # per-channel L2

    scores  = np.empty(N, dtype=[('score','f4'),
                                 ('n_bad','i4'),
                                 ('lag', 'i2')])

    # ------- brute-force lag search (N small so OK) -------------
    lags = np.arange(-max_lag, max_lag+1, dtype=int)
    for n in range(N):
        best_s, best_bad, best_lag = -np.inf, 0, 0
        X = snips[:, :, n]

        for lag in lags:
            if lag < 0:
                Xseg = X[core_ch, :lag]               # [Ccore, T+lag]
                Tseg = T_core[:, -lag:]
            elif lag > 0:
                Xseg = X[core_ch, lag:]
                Tseg = T_core[:, :-lag]
            else:
                Xseg = X[core_ch]
                Tseg = T_core

            # zero-mean the snippet on each channel
            Xseg = Xseg - Xseg.mean(axis=1, keepdims=True)

            # cosine per channel
            dot     = (Xseg * Tseg).sum(axis=1)
            norm_X  = np.linalg.norm(Xseg, axis=1) + 1e-12
            cos_ch  = dot / (norm_T * norm_X)

            # residual at template peak position
            peak_idx = np.argmin(Tseg, axis=1)
            resid    = Xseg[np.arange(len(core_ch)), peak_idx] - \
                       Tseg[np.arange(len(core_ch)), peak_idx]
            n_bad    = (resid < amp_thresh).sum()

            # weighted average cosine
            w_cos    = np.average(cos_ch, weights=w_ch)
            if w_cos > best_s:
                best_s, best_bad, best_lag = w_cos, int(n_bad), int(lag)

        scores['score'][n] = best_s
        scores['n_bad'][n] = best_bad
        scores['lag'][n]   = best_lag

    return scores        # dtype [('score','f4'), ('n_bad','i4'), ('lag','i2')]


def per_channel_cosine_scores(
        snips,            # [C, T, N]  float32
        template,         # [C, T]      float32
        max_lag=3,
        p2p_thresh=30.0
):
    """
    Return per-channel cosine similarities to 'template' for every spike.

    Parameters
    ----------
    snips      : waveform snippets [channels, samples, spikes]
    template   : EI template of the unit [channels, samples]
    max_lag    : ± samples to search for best alignment (int)
    p2p_thresh : keep only channels whose template P2P ≥ this (µV)

    Returns
    -------
    cosines : array  [C_core, N]   un-weighted cosine per channel & spike
    best_lag: array  [N]           chosen lag for each spike (−max_lag…+max_lag)
    core_idx: array  [C_core]      channel indices kept
    """
    C, T, N = snips.shape

    # --- core channels -------------------------------------------------------
    p2p       = template.ptp(axis=1)                      # [C]
    core_idx  = np.where(p2p >= p2p_thresh)[0]
    if core_idx.size == 0:
        raise ValueError("No channels pass p2p_thresh")

    T_core    = template[core_idx]                       # [Ccore, T]
    T_core    = T_core - T_core.mean(axis=1, keepdims=True)
    norm_T    = np.linalg.norm(T_core, axis=1) + 1e-12   # [Ccore]

    # --- allocate output -----------------------------------------------------
    cosines   = np.empty((core_idx.size, N), dtype=np.float32)
    best_lag  = np.empty(N, dtype=np.int16)

    lags = np.arange(-max_lag, max_lag + 1, dtype=int)

    # --- per-spike loop (vectorise later if needed) --------------------------
    for n in range(N):
        X = snips[:, :, n]

        best_mean_cos = -np.inf
        best_cos      = None
        best_l        = 0

        for l in lags:
            if l < 0:
                Xseg = X[core_idx, :l]
                Tseg = T_core[:, -l:]
            elif l > 0:
                Xseg = X[core_idx, l:]
                Tseg = T_core[:, :-l]
            else:
                Xseg = X[core_idx]
                Tseg = T_core

            Xseg = Xseg - Xseg.mean(axis=1, keepdims=True)
            norm_X = np.linalg.norm(Xseg, axis=1) + 1e-12
            cos_ch = np.sum(Xseg * Tseg, axis=1) / (norm_T * norm_X)

            mean_cos = cos_ch.mean()
            if mean_cos > best_mean_cos:
                best_mean_cos = mean_cos
                best_cos      = cos_ch
                best_l        = l

        cosines[:, n] = best_cos
        best_lag[n]   = best_l

    return cosines, best_lag, core_idx

import numpy as np

def score_shape_amp_resid(
        snips,            # [C, T, N]   float32
        template,         # [C, T]      float32
        max_lag=3,        # ± samples searched for best alignment
        p2p_thresh=30.0   # µV threshold for “core” channels
):
    """
    Compare each spike to a fixed template channel-by-channel.

    Returns
    -------
    cosines : [Ccore, N]   (float32)  plain cosine similarity
    alpha   : [Ccore, N]   (float32)  LS gain  α = <x,t>/<t,t>
    resid   : [Ccore, N]   (float32)  RMS of (x − αt)
    bestlag : [N]          (int16)    lag (–max_lag…+max_lag) chosen for each spike
    coreidx : [Ccore]      (int16)    indices of core channels kept
    """
    C, T, N = snips.shape

    # -------- core channels --------
    p2p       = template.ptp(axis=1)
    coreidx   = np.where(p2p >= p2p_thresh)[0]
    if coreidx.size == 0:
        raise ValueError("No channels above p2p_thresh")

    T_core    = template[coreidx].copy()
    T_core   -= T_core.mean(axis=1, keepdims=True)          # zero-mean per ch
    norm_T2   = np.sum(T_core**2, axis=1) + 1e-12           # [Ccore]

    # -------- outputs --------------
    Ccore     = coreidx.size
    cosines   = np.empty((Ccore, N), dtype=np.float32)
    alpha     = np.empty_like(cosines)
    resid     = np.empty_like(cosines)
    bestlag   = np.empty(N, dtype=np.int16)

    lags = np.arange(-max_lag, max_lag + 1, dtype=int)

    # -------- per-spike loop -------
    for n in range(N):
        X = snips[:, :, n]                       # [C, T]

        best_mean_cos = -np.inf
        best_a   = best_r = best_c = None
        best_l   = 0

        for l in lags:
            if l < 0:
                Xseg = X[coreidx, :l]            # [Ccore, T+l]
                Tseg = T_core[:, -l:]
            elif l > 0:
                Xseg = X[coreidx, l:]
                Tseg = T_core[:, :-l]
            else:
                Xseg = X[coreidx]
                Tseg = T_core

            Xseg  = Xseg - Xseg.mean(axis=1, keepdims=True)

            # α per channel
            dot_xt   = np.sum(Xseg * Tseg, axis=1)
            a_ch     = dot_xt / norm_T2
            # residual RMS per channel
            res_ch   = Xseg - a_ch[:, None] * Tseg
            rms_ch   = np.sqrt(np.mean(res_ch**2, axis=1))

            # cosine per channel
            norm_X2  = np.sum(Xseg**2, axis=1) + 1e-12
            cos_ch   = dot_xt / np.sqrt(norm_T2 * norm_X2)

            mean_cos = cos_ch.mean()
            if mean_cos > best_mean_cos:
                best_mean_cos = mean_cos
                best_a   = a_ch
                best_r   = rms_ch
                best_c   = cos_ch
                best_l   = l

        cosines[:, n] = best_c
        alpha[:, n]   = best_a
        resid[:, n]   = best_r
        bestlag[n]    = best_l

    return cosines, alpha, resid, bestlag, coreidx




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
                    idx_e = id_to_index[existing]   # translate ID → row/col index
                    idx_j = id_to_index[j]

                    sim_ij  = sim[idx_e, idx_j]
                    n_bad   = n_bad_channels[idx_e, idx_j]
                    # sim_ij = sim[existing, j]
                    # n_bad = n_bad_channels[existing, j]

                    accept = False
                    if sim_ij >= cos_thresh+0.15 and n_bad <= 6:
                        accept = True
                    elif sim_ij >= cos_thresh+0.1 and n_bad <= 4:
                        accept = True
                    elif sim_ij >= cos_thresh+0.05 and n_bad == 2:
                        accept = True
                    elif sim_ij >= cos_thresh and n_bad == 0:
                        accept = True

                    if accept:
                        base_group.append(j)
                        assigned.add(j)
                        added = True
                        break  # Stop checking other members once one match is found

        # Merge all spikes from group
        merged_spikes = np.concatenate([cluster_spike_indices[k] for k in base_group])
        merged_clusters.append(np.sort(merged_spikes))

    return merged_clusters, sim, n_bad_channels


def cluster_spike_waveforms_multi_kmeans(
    snips: np.ndarray,
    ei: np.ndarray,
    k_start: int = 3,          # ← kept for API parity, not used in prune loop
    p2p_threshold: float = 15,
    min_chan: int = 30,
    max_chan: int = 80,
    sim_threshold: float = 0.9,
    merge: bool = True,
    return_debug: bool = False,
    plot_diagnostic: bool = False
):
    """
    Cluster spike waveforms with a two–stage strategy:
      1. iterative k-means (k=2) pruning until separation ≤ 5
      2. valley-split / EI-merge on surviving spikes.
    Discarded clusters are returned as independent units
    so downstream code still sees every original spike.
    """

    # ───────────────────────────── 1. channel & snippet prep ────────────────────
    ei_p2p = ei.max(axis=1) - ei.min(axis=1)
    selected_channels = np.where(ei_p2p > p2p_threshold)[0]
    if len(selected_channels) > max_chan:
        selected_channels = np.argsort(ei_p2p)[-max_chan:]
    elif len(selected_channels) < min_chan:
        selected_channels = np.argsort(ei_p2p)[-min_chan:]
    selected_channels = np.sort(selected_channels)

    main_chan = int(np.argmax(ei_p2p))          # channel to quantify p2p in loop
    snips_sel = snips[selected_channels, :, :]  # [C,T,N]
    C, T, N = snips_sel.shape

    # flat view for PCA convenience
    def pca_augmented(mask: np.ndarray):
        """Return PCA-augmented feature matrix for spikes in mask."""
        # ------------- flattened EI -------------
        sn_flat = snips_sel[:, :, mask].transpose(2, 0, 1).reshape(mask.sum(), -1)
        pcs = PCA(n_components=10).fit_transform(sn_flat)
        pcs = StandardScaler().fit_transform(pcs)

        # ------------- main-channel PC1 -------------
        spike_zone = slice(20, 80)
        main_snips = snips[main_chan, spike_zone, :][:, mask].T
        main_pc1 = PCA(n_components=1).fit_transform(main_snips).flatten()
        main_pc1 = StandardScaler().fit_transform(main_pc1[:, None]).flatten()
        return np.hstack((pcs, main_pc1[:, None]))

    # keep_mask = np.ones(N, dtype=bool)      # spikes still in play
    labels_global = -np.ones(N, dtype=int)  # final labels, filled gradually
    next_discard_label = 0                  # 0,1,2,… for discarded clusters

    # ───────── 1. initial PCA/augment (full-set) ─────────
    full_mask = np.ones(N, bool)
    sn_flat = snips_sel.transpose(2, 0, 1).reshape(full_mask.sum(), -1)
    pcs_aug_full = PCA(n_components=2).fit_transform(sn_flat)

    # keep_mask will be mutated from here on
    keep_mask = full_mask.copy()

    # ───────────────────────────── 2. prune-until-mixed loop ───────────────────
    while True:
        if keep_mask.sum() < 40:            # too small to split meaningfully
            labels_global[keep_mask] = next_discard_label
            next_discard_label += 1
            break

        pcs_aug = pca_augmented(keep_mask)

        # k=2 split
        sub_labels = KMeans(n_clusters=2, n_init=10, random_state=42) \
                     .fit_predict(pcs_aug)
        pcs_1d = pcs_aug[:, :1]          # after building pcs_aug
        sep = cluster_separation_score(pcs_1d, sub_labels)[0]['separation_score']

        # import matplotlib.pyplot as plt

        # plt.figure(figsize=(6, 5))
        # for i in np.unique(sub_labels):
        #     cluster_points = pcs_aug[sub_labels == i]
        #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i+1}", s=10)
        # plt.xlabel("PC1")
        # plt.ylabel("PC2")
        # plt.title(f"PCA on spike waveforms!, {sep}")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

        if sep <= 5:                         # stop pruning
            labels_global[keep_mask] = sub_labels + next_discard_label
            break

        # decide which half to keep (higher p2p on main_chan)
        inds_keepmask = np.flatnonzero(keep_mask)
        cl0 = inds_keepmask[sub_labels == 0]
        cl1 = inds_keepmask[sub_labels == 1]
        if len(cl0) == 0 or len(cl1) == 0:
            labels_global[keep_mask] = next_discard_label
            next_discard_label += 1
            break

        # --- pick higher-p2p cluster ---
        w0 = snips[main_chan, :, cl0].mean(axis=1).ptp()
        w1 = snips[main_chan, :, cl1].mean(axis=1).ptp()
        keep_inds, discard_inds = (cl0, cl1) if w0 >= w1 else (cl1, cl0)


        # guard against tiny survivor
        if keep_inds.size < 20:
            # keep whole thing – treat as one label
            labels_global[keep_mask] = next_discard_label
            next_discard_label += 1
            break

        # assign label to discarded cluster and remove it from further splits
        labels_global[discard_inds] = next_discard_label
        next_discard_label += 1
        keep_mask[discard_inds] = False

    n_discarded = next_discard_label            # offset for surviving labels
    labels_pruned = labels_global.copy()        # snapshot before valley splits

    # ───────────────────────────── 3. valley split of survivors ────────────────
    # re-index survivors to consecutive ints starting at n_discarded
    survive_inds = np.flatnonzero(labels_pruned == -1)
    if survive_inds.size:                      # may be empty if everything pruned
        pcs_aug_survive = pca_augmented(labels_pruned == -1)
        kmeans_start = KMeans(n_clusters=k_start,
                              n_init=10, random_state=42).fit_predict(pcs_aug_survive)
        labels_pruned[survive_inds] = kmeans_start + n_discarded

        # iterative valley splits on pcs_aug_survive
        labels_updated = labels_pruned.copy()
        to_check = list(np.unique(labels_pruned[survive_inds]))
        next_label = labels_pruned.max() + 1

        while to_check:
            cl = to_check.pop(0)
            mask = labels_updated == cl
            pc_vals = pcs_aug_survive[labels_pruned[survive_inds] == cl, :2]

            if len(pc_vals) < 20:
                continue
            split = check_2d_gap_peaks_valley(pc_vals, 10, 0.25)
            if split is None:
                continue
            g1, g2 = split
            if g1.sum() < 20 or g2.sum() < 20:
                continue

            cluster_all_inds = survive_inds[mask]
            labels_updated[cluster_all_inds[g2]] = next_label
            to_check += [cl, next_label]
            next_label += 1

        labels_pruned = labels_updated

    # ───────────────────────────── 4. EI-similarity merge (kept only) ──────────
    kept_mask_final = labels_pruned >= n_discarded
    merged_clusters_kept, sim, n_bad_channels = merge_similar_clusters(
        snips[:, :, kept_mask_final], labels_pruned[kept_mask_final],
        max_lag=3, p2p_thresh=30.0, amp_thresh=-20, cos_thresh=0.75
    )

    # append discards as singleton groups
    merged_clusters = merged_clusters_kept + [
        np.flatnonzero(labels_pruned == lbl) for lbl in range(n_discarded)
    ]

    # ───────────────────────────── 5. build outputs ────────────────────────────
    output = []
    for inds in merged_clusters:
        ei_cluster = snips[:, :, inds].mean(axis=2)
        output.append({'inds': inds, 'ei': ei_cluster, 'channels': selected_channels})

    if return_debug:
        # (unchanged – adjust if you need extra debugging material)
        cluster_spike_indices = {k: np.where(labels_pruned == k)[0]
                                 for k in np.unique(labels_pruned)}
        cluster_eis = [snips[:, :, v].mean(axis=2)
                       for k, v in sorted(cluster_spike_indices.items())]
        cluster_to_merged_group = {}
        for orig_id, orig_inds in cluster_spike_indices.items():
            for g, merged_inds in enumerate(merged_clusters):
                if set(orig_inds).issubset(merged_inds):
                    cluster_to_merged_group[orig_id] = g
                    break
        return (output, pcs_aug_full if 'pcs_aug_full' in locals() else None,
                labels_pruned, sim, n_bad_channels,
                cluster_eis, cluster_to_merged_group)
    else:
        return output



def cluster_spike_waveforms(
    snips: np.ndarray,
    ei: np.ndarray,
    k_start: int = 3,
    p2p_threshold: float = 15,
    min_chan: int = 30,
    max_chan: int = 80,
    sim_threshold: float = 0.9,
    merge: bool = True,
    return_debug: bool = False, 
    plot_diagnostic: bool = False,
    print_diagnostic: bool = False
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
    spike_zone = slice(20, 80)  # or whatever range captures the peak

    main_snips = snips[main_chan, spike_zone, :].T  # shape: (N, T_spike)
    main_pc1 = PCA(n_components=1).fit_transform(main_snips).flatten()

    # --- Append to flattened snippets before clustering ---
    pcs = PCA(n_components=2).fit_transform(snips_flat)

    pcs_z = StandardScaler().fit_transform(pcs)
    main_pc1_z = StandardScaler().fit_transform(main_pc1[:, None]).flatten()

    # pcs_aug = np.hstack((pcs_z, main_pc1_z[:, None]))
    pcs_aug = np.hstack((pcs, main_pc1[:, None]))

    kmeans = KMeans(n_clusters=k_start, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pcs_aug)


    # scores = cluster_separation_score(pcs[:, :2], labels)
    # val = scores[0]['separation_score']
    # print(val)

    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(6, 5))
    # for i in np.unique(labels):
    #     cluster_points = pcs_aug[labels == i]
    #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}", s=10)
    # plt.xlabel("PC1")
    # plt.ylabel("PC2")
    # plt.title(f"PCA on spike waveforms, {val}")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


    if len(np.unique(labels))<8:

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

                if print_diagnostic and (n1 < 20 or n2 < 20):
                    print(f"  Cluster {cl} split discarded: would create small cluster (group1={n1}, group2={n2})")
                    continue

                # Assign new label to group2 (or vice versa)
                labels_updated[cluster_indices[group2_mask]] = next_label
                if print_diagnostic:
                    print(f"  Cluster {cl} split into {cl} and {next_label}")

                # Add both parts back to check if large enough
                if np.sum(group1_mask) >= 20:
                    to_check.append(cl)
                if np.sum(group2_mask) >= 20:
                    to_check.append(next_label)

                next_label += 1

        labels = labels_updated

    # # check separation

    # scores = cluster_separation_score(pcs[:, :2], labels)
    # # Convert list of dicts to matrix
    # unique_labels = np.unique(labels)
    # n_clusters = len(unique_labels)
    # score_matrix = np.full((n_clusters, n_clusters), np.nan)

    # # Fill upper triangle with scores
    # for s in scores:
    #     A, B = s['pair']
    #     i = np.where(unique_labels == A)[0][0]
    #     j = np.where(unique_labels == B)[0][0]
    #     score_matrix[i, j] = s['separation_score']

    # # Now print nicely
    # print("Separation score matrix (upper triangle):")
    # for i in range(n_clusters):
    #     row = ''
    #     for j in range(n_clusters):
    #         if i >= j:
    #             row += "   -"
    #         else:
    #             row += f"{score_matrix[i, j]:6.2f}"
    #     print(row)

    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(6, 5))  
    # for l in np.unique(labels):
    #     mask = labels == l
    #     plt.scatter(pcs[mask, 0], pcs[mask, 1], s=10, label=f"Cluster {l}", alpha=0.7)
    # plt.xlabel("PC1")
    # plt.ylabel("PC2")
    # plt.title("PC1 vs PC2 scatter with cluster labels")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


    # pca = PCA(n_components=10)
    # pcs = pca.fit_transform(snips_flat)

    # kmeans = KMeans(n_clusters=k_start, n_init=10, random_state=42)
    # labels = kmeans.fit_predict(pcs)

    if plot_diagnostic == True:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5, 3))
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


    # merged_clusters, sim, n_bad_channels = merge_similar_clusters(snips, labels, max_lag=3, p2p_thresh=30.0, amp_thresh=-20, cos_thresh=0.75)

    merged_clusters, sim, n_bad_channels = merge_similar_clusters_extra(
        snips,
        labels,                    # whatever variable you pass today
        max_lag   = 3,
        p2p_thresh= 30.0,
        amp_thresh= -20,
        cos_thresh= 0.75,
        pcs2      = pcs[:, :2],   # ← NEW
        sep_thresh= 8.0                                   # ← tune if needed
    )

    output = []
    for inds in merged_clusters:
        ei_cluster = np.mean(snips[:, :, inds], axis=2)
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


from scipy.stats import median_abs_deviation as mad

# This is a high-level scaffold. It assumes external access to the following utils:
# - extract_snippets_fast_ram
# - score_spikes_against_template
# - build_channel_weights_twoEI
# - compute_residuals_with_lag

def collision_model_two_unit(
    raw_data,
    spike_times,
    snips_baselined,
    clusters,
    cluster_ids,
    window=(-20, 60),
    p2p_thresh=30.0,
    score_z_thresh=1.0,
    max_lag=5,
    rel_mask=0.1,
    rel_thresh=0.1,
    plotting=False
):
    """
    Parameters:
    ----------
    raw_data     : np.ndarray, shape [C, T_total]
    spike_times  : np.ndarray, shape [N_total], all spike times in absolute samples
    clusters     : dict of cluster_id -> list of spike times (subset of spike_times)
    cluster_ids  : tuple (A_id, B_id)

    Returns:
    -------
    best_template_idx : [N_tested] int
    lag_matrix        : [N_templates, N_tested]
    residual_matrix   : [N_templates, N_tested]
    """

    A_id, B_id = cluster_ids
    spikesA = spike_times[clusters[A_id]['inds']]
    spikesB = spike_times[clusters[B_id]['inds']]
    print(len(spikesA))
    print(len(spikesB))

    n_channels = raw_data.shape[1]


    # --- Step 1: Extract snippets ---
    # snipsA, _ = extract_snippets_fast_ram(raw_data, spikesA, window, np.arange(n_channels))
    # snipsB, _ = extract_snippets_fast_ram(raw_data, spikesB, window, np.arange(n_channels))
    indsA = clusters[A_id]['inds']
    snipsA = snips_baselined[:, :, indsA]
    indsB = clusters[B_id]['inds']
    snipsB = snips_baselined[:, :, indsB]

    print("finished extracting snippets")


    # --- Step 2: Score spikes against their own template ---
    eiA = np.mean(snipsA, axis=2)
    eiB = np.mean(snipsB, axis=2)

    scoreA = score_spikes_against_template(snipsA, eiA)['score']
    scoreB = score_spikes_against_template(snipsB, eiB)['score']

    print("finished scoring against template")

    zA = (scoreA - np.median(scoreA)) / (mad(scoreA) + 1e-6)
    zB = (scoreB - np.median(scoreB)) / (mad(scoreB) + 1e-6)

    goodA = np.where(zA > 2)[0]
    badA  = np.where(zA <= -score_z_thresh)[0]
    goodB = np.where(zB > 2)[0]
    badB  = np.where(zB <= -score_z_thresh)[0]

    # goodA / goodB / badB are *positions inside* indsA / indsB
    snipsA_good = snips_baselined[:, :, indsA[goodA]]    # C × T × N_goodA
    snipsB_good = snips_baselined[:, :, indsB[goodB]]    # C × T × N_goodB
    snips_test   = snips_baselined[:, :, indsA[badA]]    # C × T × N_badB  (collision candidates)
    test_inds = indsA[badA]
    print(len(badA))
    print(len(goodA))

    eiA = np.mean(snipsA_good, axis=2)
    eiB = np.mean(snipsB_good, axis=2)

    # --- Step 3: Construct collision templates ---
    collision_templates = [eiA.copy(), eiB.copy(), eiA + eiB]
    template_labels = ["A", "B", "A+B (lag 0)"]

    T = eiA.shape[1]
    for lag in range(1, 25):
        t1 = eiA.copy()
        t1[:, lag:] += eiB[:, :T - lag]
        collision_templates.append(t1)
        template_labels.append(f"A + B (lag {lag})")

        t2 = eiB.copy()
        t2[:, lag:] += eiA[:, :T - lag]
        collision_templates.append(t2)
        template_labels.append(f"B + A (lag {lag})")

    collision_templates = np.array(collision_templates, dtype=np.float32)  # [n_templates, C, T]
    print("finished template building")


    # --- Step 4: Select channels and compute weights ---
    weights, selected_chans = build_channel_weights_twoEI(
        snipsA_good, snipsB_good, eiA, eiB,
        p2p_thresh=p2p_thresh,
        rel_mask=rel_mask
    )
    print("finished weight")
    templates_sel = collision_templates[:, selected_chans, :]  # [n_templates, C_sel, T]
    snips_sel     = snips_test[selected_chans, :, :].copy()

    # --- Step 5: Score all bad spikes against template dictionary ---
    residual_matrix, lag_matrix = compute_residuals_with_lag(
        snips_sel,
        templates_sel,
        weights=weights,
        max_lag=max_lag,
        rel_thresh=rel_thresh
    )
    print("finished scoring")

    best_template_idx = np.argmin(residual_matrix, axis=0)  # [N_spikes]

    for i, t_idx in enumerate(best_template_idx):
        label = template_labels[t_idx]
        resid = residual_matrix[t_idx, i]
        print(f"Spike {i:4d}: best match = {label:<8} | residual = {resid:.4f}")

    if plotting:
        import matplotlib.pyplot as plt
        i = 9 # example spike index to plot
        channels_to_plot = [395,403,402,394]#selected_chans[:min(8, len(selected_chans))]  # pick a few channels for visualization
        extra_templates = [0, 1]  # just show solo A and B for context

        x = snips_sel[:, :, i]  # [C, T]
        t_idx = best_template_idx[i]
        best_template = templates_sel[t_idx]  # restrict to selected channels
        best_label = template_labels[t_idx]
        lag = lag_matrix[t_idx, i]

        C, T = x.shape
        if lag < 0:
            lag_abs = abs(lag)
            templ_shifted = best_template[:, lag_abs:]
            spike_segment = x[:, :T - lag_abs]
        elif lag > 0:
            templ_shifted = best_template[:, :T - lag]
            spike_segment = x[:, lag:]
        else:
            templ_shifted = best_template
            spike_segment = x

        min_len = min(templ_shifted.shape[1], spike_segment.shape[1])
        templ_shifted = templ_shifted[:, :min_len]
        spike_segment = spike_segment[:, :min_len]


        assert templ_shifted.shape == spike_segment.shape, "Shape mismatch after alignment truncation"

        # 🟢 Then continue with the original plotting block:
        # Plotting
        n_channels = len(channels_to_plot)
        cols = min(n_channels, 4)
        rows = int(np.ceil(n_channels / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
        axs = axs.flatten()

        for idx, ch in enumerate(channels_to_plot):
            if ch not in selected_chans:
                print(f"Channel {ch} not in selected_chans — skipping")
                continue

            local_idx = np.where(selected_chans == ch)[0][0]  # convert global → local index

            axs[idx].plot(spike_segment[local_idx], color='black', linewidth=1, label='Spike')
            axs[idx].plot(templ_shifted[local_idx], color='red', linewidth=1.2, label=f'{best_label} (lag {lag})')

            for tidx in extra_templates:
                t_extra = templates_sel[tidx]  # already channel-selected
                if lag < 0:
                    t_extra_shifted = t_extra[:, abs(lag):]
                elif lag > 0:
                    t_extra_shifted = t_extra[:, :T - lag]
                else:
                    t_extra_shifted = t_extra

                t_extra_shifted = t_extra_shifted[:, :min_len]
                axs[idx].plot(t_extra_shifted[local_idx], linestyle='--', linewidth=1, label=f'{template_labels[tidx]}')


        for j in range(idx + 1, len(axs)):
            axs[j].axis('off')

        fig.suptitle(f"Spike {i} assigned to '{best_label}'", fontsize=10)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

    return (
        best_template_idx,
        lag_matrix,
        residual_matrix,
        template_labels,
        test_inds,
        templates_sel,      # [n_templates × C_sel × T]
        weights,            # [C_sel]
        selected_chans      # [C_sel] — global channel indices
    )


import numpy as np
from typing import Tuple, List
from numba import njit, prange
from scipy.signal import correlate

# --------------------------------------------------------------------
# ❶  helper: roll array by lag (positive → right shift), zero-pad vacated end
@njit(fastmath=True, cache=True)
def _roll_pad(arr, lag):
    C, T = arr.shape
    out  = np.zeros_like(arr)
    if lag > 0:
        out[:, lag:] = arr[:, :T-lag]
    elif lag < 0:
        out[:, :T+lag] = arr[:, -lag:]
    else:
        out[:] = arr
    return out
# --------------------------------------------------------------------


@njit(parallel=True, fastmath=True, cache=True)
def _compute_rms(snips, rolled_templates, rolled_masks,
                 weights, p2p_template, p2p_global,
                 residuals, bestlags):
    """
    Heavy inner loop: identical to the one already in axolotl_utils_ram.py,
    but reused here for clarity.
    """
    nT, nL, C, T = rolled_templates.shape
    N            = snips.shape[2]

    for t in prange(nT):
        p2p_c = p2p_template[t]

        for s in range(N):
            best_r  = 1e20
            best_li = 0

            for li in range(nL):
                tmpl = rolled_templates[t, li]
                mask = rolled_masks[t, li]

                err_tot = 0.0
                w_sum   = 0.0

                # per-channel RMS
                for c in range(C):
                    max_abs = 0.0
                    sum_e   = 0.0
                    sum_e2  = 0.0
                    n_mask  = 0

                    for k in range(T):
                        if mask[c, k]:
                            val     = snips[c, k, s]
                            diff    = val - tmpl[c, k]
                            sum_e  += diff
                            sum_e2 += diff * diff
                            n_mask += 1
                            if abs(val) > max_abs:
                                max_abs = abs(val)

                    if n_mask < 3:
                        continue
                    if max_abs > 2.0 * p2p_global[c]:   # AMP_FACTOR = 2
                        continue

                    var_c = (sum_e2 - (sum_e * sum_e) / n_mask) / n_mask
                    if var_c < 0.0:
                        var_c = 0.0

                    w      = weights[c]
                    err_tot += w * var_c
                    w_sum  += w

                if w_sum == 0.0:
                    continue

                rms = np.sqrt(err_tot / w_sum)
                if rms < best_r:
                    best_r  = rms
                    best_li = li

            residuals[t, s] = best_r
            bestlags[t, s]  = best_li
# --------------------------------------------------------------------


def collision_model_two_templates(
    snips: np.ndarray,                 # [C, T, N]  float32   – snippets (baseline-subtracted)
    eiA:   np.ndarray,                 # [C, T]      float32   – template A
    eiB:   np.ndarray,                 # [C, T]      float32   – template B
    *,
    max_lag:       int   = 25,         # build combos up to this lag (samples, ≥1)
    p2p_thresh:    float = 30.0,       # keep channels where max(P2P_A, P2P_B) ≥ thresh
    rel_mask:      float = 0.01,       # |t| ≥ rel_mask·P2P keeps a sample
    rel_thresh:    float = 0.10,        # threshold inside compute_residuals_with_lag
    plotting: False
) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, List[str],
        np.ndarray, np.ndarray, np.ndarray
     ]:
    """
    Score EVERY spike against {A, B, A+B_lagged, B+A_lagged} template dictionary.

    Returns
    -------
    best_template_idx : [N]      int      – index in template_labels
    lag_matrix        : [M, N]   int16    – chosen lag for each template/spike
    residual_matrix   : [M, N]   float32  – RMS residuals
    template_labels   : list[str]         – human-readable labels
    templates_sel     : [M, Csel, T]      – rolled templates (selected channels)
    weights           : [Csel]   float32  – per-channel weights (mean = 1)
    selected_chans    : [Csel]   int32    – global channel indices kept
    """

    # ------------------------------------------------------------
    # 1. channel selection & simple weights
    # ------------------------------------------------------------
    C, T = eiA.shape
    p2p_A = eiA.ptp(axis=1)
    p2p_B = eiB.ptp(axis=1)
    keep  = np.where(np.maximum(p2p_A, p2p_B) >= p2p_thresh)[0]
    if keep.size == 0:
        raise ValueError("No channels pass p2p_thresh")

    selected_chans = keep.astype(np.int32)
    weights        = np.maximum(p2p_A[keep], p2p_B[keep]).astype(np.float32)
    weights       /= weights.mean()                      # normalise

    # weights, selected_chans = build_channel_weights_twoEI(
    #     snips, snips, eiA, eiB,
    #     p2p_thresh=p2p_thresh,
    #     rel_mask=rel_mask
    # )

    # ------------------------------------------------------------
    # 2. build template dictionary
    # ------------------------------------------------------------
    templates = [eiA, eiB, eiA + eiB]          # lag 0 combos
    labels    = ["A", "B", "A+B (lag 0)"]

    for lag in range(1, max_lag + 1):
        # A then B
        t_AB = np.zeros_like(eiA)
        if lag < T:
            t_AB[:, :-lag] = eiA[:, :-lag] + eiB[:, lag:]
        templates.append(t_AB)
        labels.append(f"A+B (lag {lag})")

        # B then A
        t_BA = np.zeros_like(eiA)
        if lag < T:
            t_BA[:, :-lag] = eiB[:, :-lag] + eiA[:, lag:]
        templates.append(t_BA)
        labels.append(f"B+A (lag {lag})")

    templates = np.array(templates, dtype=np.float32)    # [M, C, T]

    # ------------------------------------------------------------
    # 3. restrict to selected channels
    # ------------------------------------------------------------
    templates_sel = templates[:, selected_chans, :]      # [M, Csel, T]
    snips_sel     = snips[selected_chans, :, :].copy()   # [Csel, T, N]

    # ------------------------------------------------------------
    # 4. build masks & rolled versions for lag search
    # ------------------------------------------------------------
    n_temp  = templates_sel.shape[0]
    lags    = np.arange(-max_lag, max_lag + 1, dtype=np.int16)
    n_lags  = lags.size
    Csel    = templates_sel.shape[1]

    # master mask: union of |template| > rel_mask·p2p over all templates
    p2p_global = np.max(np.abs(templates_sel), axis=(0,2)).astype(np.float32)   # [Csel]
    master_mask0 = np.max(np.abs(templates_sel), axis=0) > (rel_mask * p2p_global[:,None] + 1e-6)

    rolled_templates = np.empty((n_temp, n_lags, Csel, T), np.float32)
    rolled_masks     = np.empty_like(rolled_templates, dtype=np.bool_)

    for t in range(n_temp):
        tmpl = templates_sel[t]
        for li, L in enumerate(lags):
            rolled_templates[t, li] = _roll_pad(tmpl, L)
            rolled_masks[t, li]     = _roll_pad(master_mask0.astype(np.float32), L) > 0.5

    p2p_template = np.max(np.abs(templates_sel), axis=2).astype(np.float32)   # [M, Csel]

    # ------------------------------------------------------------
    # 5. residual computation (uses the numba kernel above)
    # ------------------------------------------------------------
    N = snips_sel.shape[2]
    residuals = np.empty((n_temp, N), dtype=np.float32)
    bestlags  = np.empty((n_temp, N), dtype=np.int16)

    _compute_rms(snips_sel, rolled_templates, rolled_masks,
                 weights, p2p_template, p2p_global,
                 residuals, bestlags)

    # translate lag indices → signed lag values
    bestlags[:] = lags[bestlags]

    best_template_idx = np.argmin(residuals, axis=0)      # [N]

    if plotting:
        import matplotlib.pyplot as plt

        spikes_to_plot = [0,12,  296]  # replace with any list of spike indices
        channels_to_plot = [ 49,  52,  57, 209, 210]  # global channel IDs
        extra_templates = [0, 1]  # indices of templates to overlay

        for i in spikes_to_plot:
            x = snips_sel[:, :, i]  # [Csel, T]
            t_idx = best_template_idx[i]
            best_template = templates_sel[t_idx]
            best_label = labels[t_idx]
            lag = bestlags[t_idx, i]

            C, T = x.shape
            if lag < 0:
                lag_abs = abs(lag)
                templ_shifted = best_template[:, lag_abs:]
                spike_segment = x[:, :T - lag_abs]
            elif lag > 0:
                templ_shifted = best_template[:, :T - lag]
                spike_segment = x[:, lag:]
            else:
                templ_shifted = best_template
                spike_segment = x

            min_len = min(templ_shifted.shape[1], spike_segment.shape[1])
            templ_shifted = templ_shifted[:, :min_len]
            spike_segment = spike_segment[:, :min_len]

            assert templ_shifted.shape == spike_segment.shape, "Shape mismatch after alignment truncation"

            # Plotting
            n_channels = len(channels_to_plot)
            cols = min(n_channels, 6)
            rows = int(np.ceil(n_channels / cols))
            fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
            axs = axs.flatten()

            for idx, ch in enumerate(channels_to_plot):
                if ch not in selected_chans:
                    print(f"Channel {ch} not in selected_chans — skipping")
                    continue

                local_idx = np.where(selected_chans == ch)[0][0]  # convert global → local index

                axs[idx].plot(spike_segment[local_idx], color='black', linewidth=1, label='Spike')
                axs[idx].plot(templ_shifted[local_idx], color='red', linewidth=1.2,
                            label=f'{best_label} (lag {lag})')

                for tidx in extra_templates:
                    t_extra = templates_sel[tidx]
                    if lag < 0:
                        t_extra_shifted = t_extra[:, abs(lag):]
                    elif lag > 0:
                        t_extra_shifted = t_extra[:, :T - lag]
                    else:
                        t_extra_shifted = t_extra

                    t_extra_shifted = t_extra_shifted[:, :min_len]
                    axs[idx].plot(t_extra_shifted[local_idx], linestyle='--', linewidth=1, label=f'{labels[tidx]}')

                axs[idx].set_title(f"Channel {ch}", fontsize=9)
                axs[idx].legend(fontsize=7)

            for j in range(idx + 1, len(axs)):
                axs[j].axis('off')

            fig.suptitle(f"Spike {i} assigned to '{best_label}'", fontsize=12)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()


    return (best_template_idx, bestlags, residuals,
            labels, templates_sel, weights, selected_chans)



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
        # p2p = ei[ref_channel, :].max() - ei[ref_channel, :].min()
        p2p = ei[ref_channel, :].min()
        amplitudes.append(p2p)
    print(amplitudes)

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

    if pcs is not None:
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
