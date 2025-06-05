import numpy as np
from scipy.signal import correlate
from joblib import Parallel, delayed

def compute_channel_windows(ei, selected_channels, window):
    """
    Extract per-channel EI waveform windows centered on the absolute peak.
    Returns: ei_windows [C x W], p2p [C], relative_indices [C]
    """
    C = len(selected_channels)
    win_len = window[1] - window[0] + 1
    ei_windows = np.zeros((C, win_len), dtype=np.float32)
    p2p = np.zeros(C, dtype=np.float32)
    rel_peaks = np.zeros(C, dtype=np.int32)

    for i, ch in enumerate(selected_channels):
        waveform = ei[ch]
        peak_idx = np.argmin(waveform)
        print(f"Channel {ch}: EI peak at sample {peak_idx}")  # Debug print
        rel_peaks[i] = peak_idx
        win_start = peak_idx + window[0]
        win_end = peak_idx + window[1] + 1
        if win_start < 0 or win_end > waveform.shape[0]:
            raise ValueError(f"Window [{win_start}, {win_end}) out of bounds for channel {ch}")
        ei_windows[i] = waveform[win_start:win_end]
        p2p[i] = waveform.max() - waveform.min()

    return ei_windows, p2p, rel_peaks

def estimate_single_lag(snippet, ei_windows, p2p_weights, rel_peaks, window, max_lag, snippet_index, selected_channels):
    """
    snippet: [C x T]
    ei_windows: [C x W]
    """
    C, T = snippet.shape
    lags = np.zeros(C, dtype=np.int32)

    for c in range(C):
        peak_idx = rel_peaks[c]
        win_start = peak_idx + window[0]
        win_end = peak_idx + window[1] + 1
        snip_win = snippet[c, win_start:win_end]

        xcorr = correlate(snip_win, ei_windows[c], mode='full')
        mid = len(xcorr) // 2
        search_range = xcorr[mid - max_lag: mid + max_lag + 1]
        lag = np.argmax(search_range) - max_lag
        lags[c] = lag

        if selected_channels[c] == 39 and 460 <= snippet_index <= 465:
            print(f"Spike {snippet_index}, Channel 39 lag = {lag}")

    weighted_lag = np.average(lags, weights=p2p_weights)
    return int(np.floor(weighted_lag))

def estimate_spike_lags(snippets, ei, selected_channels, window=(-5, 10), max_lag=3, n_jobs=16):
    """
    snippets: [N x C x T]
    ei: [512 x T]
    selected_channels: [C]
    Returns: [N] integer lags for each spike
    """
    N, C, T = snippets.shape
    ei_windows, p2p_weights, rel_peaks = compute_channel_windows(ei, selected_channels, window)

    def process_one(i):
        return estimate_single_lag(snippets[i], ei_windows, p2p_weights, rel_peaks, window, max_lag, i, selected_channels)

    lags = Parallel(n_jobs=n_jobs)(delayed(process_one)(i) for i in range(N))
    return np.array(lags, dtype=np.int32)
