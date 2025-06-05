import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from plot_subtraction_overlay import plot_subtraction_overlay

def subtract_and_plot_single_channel(dat_path,
                                     channel,
                                     final_spike_times,
                                     start_sample,
                                     segment_length,
                                     n_channels=512,
                                     dtype=np.int16,
                                     window=(-20, 60),
                                     subtraction_window=(10, 30),
                                     template_center=20,
                                     max_lag=3):
    """
    Subtract spike templates from a single-channel segment in a .dat file and plot overlay.

    Parameters:
        dat_path           : path to the .dat file (will be modified in place)
        channel            : integer channel index (0-based)
        final_spike_times  : array of global spike sample indices
        start_sample       : starting global sample of the segment
        segment_length     : number of samples to process
        All other params follow usual MEA data conventions.
    """

    # --- Step 1: Read unmodified segment ---
    segment_offset = start_sample * n_channels
    segment_shape = (segment_length, n_channels)
    raw_data = np.memmap(dat_path, dtype=dtype, mode='r', offset=segment_offset, shape=segment_shape)
    raw_channel_segment = raw_data[:, channel].astype(np.float32).copy()  # preserve original

    # --- Step 2: Extract spike-aligned snippets for this channel ---
    win_start, win_end = window
    snip_len = win_end - win_start + 1
    spikes_in_segment = final_spike_times[
        (final_spike_times >= start_sample - win_start) &
        (final_spike_times + win_end < start_sample + segment_length)
    ]

    if len(spikes_in_segment) == 0:
        print("No spikes fall fully within the specified segment.")
        return

    snippets = []
    for spike_time in spikes_in_segment:
        local_start = spike_time + win_start - start_sample
        local_end = local_start + snip_len
        snippet = raw_data[local_start:local_end, channel].astype(np.float32)
        snippets.append(snippet)

    snippets = np.stack(snippets, axis=1)  # shape: [T, N]

    # --- Step 3: Compute mean template and align all snippets ---
    ref_trace = np.mean(snippets, axis=1)
    x_shifts = []
    aligned_snippets = np.zeros_like(snippets)

    for i in range(snippets.shape[1]):
        trace = snippets[:, i]
        xc = correlate(trace, ref_trace, mode='full')
        center = len(xc) // 2
        lag_window = xc[center - max_lag:center + max_lag + 1]
        shift = np.argmax(lag_window) - max_lag
        aligned_snippets[:, i] = np.roll(trace, -shift)
        x_shifts.append(shift)

    mean_template = np.mean(aligned_snippets, axis=1)

    # --- Step 4: Write modified trace to file (in-place) ---
    dat = np.memmap(dat_path, dtype=dtype, mode='r+', shape=(-1, n_channels), order='C')
    sub_start, sub_end = subtraction_window
    snip_sub_len = sub_end - sub_start + 1

    for i, spike_time in enumerate(spikes_in_segment):
        aligned_time = spike_time + x_shifts[i]
        t0 = aligned_time + sub_start
        t1 = t0 + snip_sub_len
        if t0 < start_sample or t1 >= start_sample + segment_length:
            continue  # skip boundary violations

        local_t0 = t0 - start_sample
        local_t1 = t1 - start_sample
        cut_start = template_center + sub_start
        cut_end = cut_start + snip_sub_len
        template_cut = mean_template[cut_start:cut_end]

        trace = dat[local_t0:local_t1, channel].astype(np.float32)
        trace -= template_cut
        dat[local_t0:local_t1, channel] = np.clip(trace, -32768, 32767).astype(np.int16)

    del dat  # ensure write is flushed

    # --- Step 5: Read modified trace for plotting ---
    dat_post = np.memmap(dat_path, dtype=dtype, mode='r', offset=segment_offset, shape=segment_shape)
    raw_channel_modified = dat_post[:, channel].astype(np.float32)

    # --- Step 6: Plot overlay ---
    plot_subtraction_overlay(raw_channel_segment, raw_channel_modified,
                              start_sample_plot=0,
                              end_sample_plot=segment_length,
                              channel_label=f"Channel {channel}")
