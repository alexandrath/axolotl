from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

def build_templates(snips, align_channel, x_shifts, window=(10, 30), n_bins=5, peak_indices=None):
    """
    Aligns, vertically centers, and bins snippets on a target channel to build subtraction templates.
    Includes debug plots for visual inspection.

    Parameters:
        snips         : [C x T x N] array of spike snippets
        align_channel : int, channel to subtract (apply x_shifts here)
        x_shifts      : array of length N, sample shifts to apply to each snippet
        window        : tuple (start, end) indices for y-centering and peak search
        n_bins        : int, number of bins per dimension (default: 5)
        peak_indices  : optional (idx1, idx2) to override automatic RMS peak detection

    Returns:
        aligned_traces : [T x N] float32, time- and y-aligned waveforms
        templates       : 2D list of [T] arrays, size [n_bins x n_bins] (or 1x1 fallback)
        template_ids    : array of length N, (label1, label2) tuples per spike
        mean_template   : [T] array, overall average template
        peak_indices    : (idx1, idx2) used (or None if fallback)
    """
    C, T, N = snips.shape
    aligned_traces = np.zeros((T, N), dtype=np.float32)

    # Step 1: Apply x-shifts to align_channel
    for i in range(N):
        trace = snips[align_channel, :, i]
        aligned_traces[:, i] = np.roll(trace, -x_shifts[i])

    # Step 2: Y-centering per spike
    y_start, y_end = window
    for i in range(N):
        offset = np.mean(aligned_traces[y_start:y_end + 1, i])
        aligned_traces[:, i] -= offset

    # Step 3: Mean template and residuals
    mean_template = np.mean(aligned_traces, axis=1)
    residuals_mean = aligned_traces - mean_template[:, None]
    rms_mean = np.sqrt(np.mean(residuals_mean**2, axis=1))

    # Step 4: Get or detect peak_indices
    if peak_indices is not None:
        idx1, idx2 = peak_indices
    else:
        peak_range = np.arange(y_start, y_end + 1)
        rms_window = rms_mean[y_start:y_end + 1]
        peaks, props = find_peaks(rms_window, prominence=5)

        if len(peaks) >= 2:
            sorted_peaks = np.argsort(props['prominences'])[::-1][:2]
            local_idx1, local_idx2 = peaks[sorted_peaks[0]], peaks[sorted_peaks[1]]
            idx1 = peak_range[local_idx1]
            idx2 = peak_range[local_idx2]
        else:
            idx1 = idx2 = None

    # Step 5: Build templates
    if idx1 is not None and idx2 is not None:
        vals1 = aligned_traces[idx1, :]
        vals2 = aligned_traces[idx2, :]

        edges1 = np.percentile(vals1, np.linspace(0, 100, n_bins + 1))
        edges2 = np.percentile(vals2, np.linspace(0, 100, n_bins + 1))

        labels1 = np.digitize(vals1, bins=edges1[1:-1])
        labels2 = np.digitize(vals2, bins=edges2[1:-1])

        templates = [[None for _ in range(n_bins)] for _ in range(n_bins)]
        template_ids = []

        for i in range(n_bins):
            for j in range(n_bins):
                idx = np.where((labels1 == i) & (labels2 == j))[0]
                if len(idx) > 0:
                    templates[i][j] = np.mean(aligned_traces[:, idx], axis=1)
                else:
                    templates[i][j] = np.zeros(T, dtype=np.float32)

        residuals_fuzzy = np.zeros_like(aligned_traces)
        for i in range(N):
            ti, tj = labels1[i], labels2[i]
            template = templates[ti][tj]
            residuals_fuzzy[:, i] = aligned_traces[:, i] - template

        template_ids = [(labels1[i], labels2[i]) for i in range(N)]
        used_peak_indices = (idx1, idx2)

    else:
        # Fallback: single template
        templates = [[mean_template]]
        template_ids = [(0, 0)] * N
        residuals_fuzzy = residuals_mean
        used_peak_indices = None

    # --- Debug plots ---
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Mean template
    axs[0].plot(mean_template, color='black')
    axs[0].set_title('Mean Aligned Trace (Template)')
    axs[0].set_xlabel('Sample')
    axs[0].grid(True)

    # Plot 2: RMS comparison
    axs[1].plot(rms_mean, label='Mean Template RMS', linestyle='--')
    axs[1].plot(np.sqrt(np.mean(residuals_fuzzy**2, axis=1)), color='green', label='Fuzzy Template RMS')
    if used_peak_indices:
        axs[1].axvline(idx1, color='red', linestyle='--', label='Peak 1')
        axs[1].axvline(idx2, color='blue', linestyle='--', label='Peak 2')
    axs[1].set_title('RMS Residuals')
    axs[1].legend()
    axs[1].grid(True)

    # Plot 3: Mean residual
    axs[2].plot(np.mean(residuals_fuzzy, axis=1), color='black')
    axs[2].axhline(0, color='gray', linestyle='--')
    axs[2].set_title('Mean Residual (Fuzzy Subtraction)')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    return aligned_traces, templates, template_ids, mean_template, used_peak_indices
