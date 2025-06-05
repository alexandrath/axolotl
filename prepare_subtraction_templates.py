import numpy as np
from scipy.signal import find_peaks, correlate
from collections import defaultdict
import os
import matplotlib.pyplot as plt

def prepare_subtraction_templates(dat_path,
                                  spike_times,
                                  EI_threshold=10,
                                  window=(-20, 60),
                                  subtraction_window=(10, 50),
                                  max_lag=3,
                                  n_bins=5,
                                  n_channels=512,
                                  dtype=np.int16,
                                  save_dir=None,
                                  ks_id=None):
    """
    Prepares aligned templates and template IDs for all EI-relevant channels for a given unit.

    Parameters:
        dat_path     : path to original .dat file
        spike_times  : global spike sample indices for the unit
        EI_threshold : min p2p amplitude (in raw units) to include a channel
        window       : tuple (pre, post) window for snippets
        subtraction_window : tuple (start, end) for alignment/RMS
        max_lag      : max alignment lag (samples) for x-shift computation
        n_bins       : number of bins per peak dimension for fuzzy templates
        n_channels   : number of total channels in the recording
        dtype        : typically np.int16

    Returns:
        ei_template       : [n_channels x T] array
        ei_channels       : list of channel indices with p2p > threshold
        x_shifts          : [N] array of per-spike time shifts
        templates_by_chan : dict[channel] = 2D list of templates
        template_ids_by_chan : dict[channel] = list of template ID tuples per spike
        y_shifts_by_chan  : dict[channel] = list of per-spike y-shifts
    """

    # --- Step 1: Extract all snippets for all channels
    snip_len = window[1] - window[0] + 1
    spike_count = len(spike_times)
    snips = np.zeros((n_channels, snip_len, spike_count), dtype=np.float32)

    with open(dat_path, 'rb') as f:
        f.seek(0, 2)
        file_len_bytes = f.tell()
        total_samples = file_len_bytes // (np.dtype(dtype).itemsize * n_channels)

        for i, center in enumerate(spike_times):
            t_start = center + window[0]
            t_end = center + window[1]
            if t_start < 0 or t_end >= total_samples:
                continue

            offset = t_start * n_channels * np.dtype(dtype).itemsize
            f.seek(offset, 0)
            raw = np.fromfile(f, dtype=dtype, count=n_channels * snip_len)
            snips[:, :, i] = raw.reshape((snip_len, n_channels)).T

    # --- Step 2: Compute EI and determine channels to include
    ei_template = np.mean(snips, axis=2)
    ei_ptp = ei_template.max(axis=1) - ei_template.min(axis=1)
    ref_channel = np.argmax(ei_ptp)
    ei_channels = np.where(ei_ptp > EI_threshold)[0]

    # --- Step 3: Compute x-shifts using ref_channel and best matching spike
    ref_traces = snips[ref_channel, :, :]
    ref_template = np.mean(ref_traces, axis=1)
    dists = np.sum((ref_traces.T - ref_template) ** 2, axis=1)
    spike_idx = np.argmin(dists)

    x_shifts = []
    ref_trace = snips[ref_channel, :, spike_idx]
    for i in range(snips.shape[2]):
        trace = snips[ref_channel, :, i]
        xc = correlate(trace, ref_trace, mode='full')
        center = len(xc) // 2
        win = xc[center - max_lag:center + max_lag + 1]
        shift = np.argmax(win) - max_lag
        x_shifts.append(shift)
    x_shifts = np.array(x_shifts)

    # --- Step 4: Compute templates and IDs for each EI-included channel
    templates_by_chan = {}
    template_ids_by_chan = {}
    y_shifts_by_chan = {}
    y_start, y_end = subtraction_window

    for chan in ei_channels:
        # Align and track y-shifts
        aligned = np.zeros((snips.shape[1], snips.shape[2]), dtype=np.float32)
        for i, s in enumerate(x_shifts):
            aligned[:, i] = np.roll(snips[chan, :, i], -s)

        y_offsets = []
        ref_trace = aligned[:, spike_idx]
        ref_window = ref_trace[y_start:y_end+1]
        ref_centered = ref_window - np.mean(ref_window)

        for i in range(aligned.shape[1]):
            trace_window = aligned[y_start:y_end+1, i]
            offset = np.mean(trace_window - ref_centered)
            aligned[:, i] -= offset
            y_offsets.append(offset)

        y_shifts_by_chan[chan] = np.array(y_offsets)

        # Template and residual RMS in subtraction window only
        mean_template = np.mean(aligned, axis=1)
        residuals = aligned - mean_template[:, None]
        rms = np.sqrt(np.mean(residuals[y_start:y_end+1, :] ** 2, axis=1))

        peaks, props = find_peaks(rms, prominence=5)

        if len(peaks) >= 2:
            sorted_peaks = np.argsort(props['prominences'])[::-1][:2]

            idx1 = peaks[sorted_peaks[0]] + y_start
            idx2 = peaks[sorted_peaks[1]] + y_start


            vals1 = aligned[idx1, :]
            vals2 = aligned[idx2, :]

            edges1 = np.percentile(vals1, np.linspace(0, 100, n_bins + 1))
            edges2 = np.percentile(vals2, np.linspace(0, 100, n_bins + 1))
            labels1 = np.digitize(vals1, bins=edges1[1:-1])
            labels2 = np.digitize(vals2, bins=edges2[1:-1])
            labels1 = np.clip(labels1, 0, n_bins - 1)
            labels2 = np.clip(labels2, 0, n_bins - 1)

            templates = [[None for _ in range(n_bins)] for _ in range(n_bins)]
            for i_bin in range(n_bins):
                for j_bin in range(n_bins):
                    idx = np.where((labels1 == i_bin) & (labels2 == j_bin))[0]
                    if len(idx) > 0:
                        tpl = np.mean(aligned[:, idx], axis=1)
                        tpl[y_start:y_end+1] -= tpl[y_start]  # zero baseline
                        templates[i_bin][j_bin] = tpl
                    else:
                        templates[i_bin][j_bin] = np.zeros(aligned.shape[0])

            ids = [(labels1[i], labels2[i]) for i in range(len(labels1))]

            if 0:
                ids = []
                for i in range(aligned.shape[1]):
                    trace = aligned[y_start:y_end+1, i]
                    best_score = np.inf
                    best_id = (0, 0)
                    for a in range(n_bins):
                        for b in range(n_bins):
                            tpl = templates[a][b]
                            if tpl is not None:
                                tpl_segment = tpl[y_start:y_end+1]
                                score = np.sum((trace - tpl_segment) ** 2)
                                if score < best_score:
                                    best_score = score
                                    best_id = (a, b)
                    ids.append(best_id)

        else:
            mean_template[y_start:y_end+1] -= mean_template[y_start]
            templates = [[mean_template]]
            ids = [(0, 0)] * snips.shape[2]
            idx1 = 0
            idx2 = 0

        if chan==ref_channel:
            if len(peaks) >= 2:
                print(idx1, idx2)
                fig, axs = plt.subplots(n_bins, n_bins, figsize=(3*n_bins, 3*n_bins), sharex=True, sharey=True)

                for i_bin in range(n_bins):
                    for j_bin in range(n_bins):
                        ax = axs[i_bin, j_bin]
                        idx = np.where((labels1 == i_bin) & (labels2 == j_bin))[0]
                        if len(idx) > 0:
                            traces = aligned[:, idx]
                            for k in range(traces.shape[1]):
                                ax.plot(traces[:, k], color='gray', alpha=0.5, linewidth=0.5)
                            mean_trace = np.mean(traces, axis=1)
                            ax.plot(mean_trace, color='red', linewidth=1.5, label='Mean')
                        ax.set_title(f"Bin ({i_bin},{j_bin}) - N={len(idx)}")
                        ax.grid(True)

                if save_dir is not None:
                    template_plot_path = os.path.join(save_dir, f'unit_{ks_id}_templates.png')
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave room for title
                    plt.suptitle(f"Unit {ks_id} | Template bins (idx1={idx1}, idx2={idx2}) | channel {ref_channel+1}", fontsize=16)
                    plt.savefig(template_plot_path, dpi=150)
                    plt.close()
            elif len(peaks) < 2:
                fig, ax = plt.subplots(figsize=(6, 4))
                for k in range(aligned.shape[1]):
                    ax.plot(aligned[:, k], color='gray', alpha=0.5, linewidth=0.5)
                mean_trace = np.mean(aligned, axis=1)
                ax.plot(mean_trace, color='red', linewidth=1.5)
                ax.set_title(f"Unit {ks_id} | Single Template | channel {ref_channel+1}")
                ax.grid(True)
                if save_dir is not None:
                    fallback_path = os.path.join(save_dir, f'unit_{ks_id}_single_template.png')
                    plt.tight_layout()
                    plt.savefig(fallback_path, dpi=150)
                    plt.close()







        templates_by_chan[chan] = templates
        template_ids_by_chan[chan] = ids

    return ei_template, ei_channels, x_shifts, templates_by_chan, template_ids_by_chan, y_shifts_by_chan
