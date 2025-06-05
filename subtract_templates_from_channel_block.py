import numpy as np
import matplotlib.pyplot as plt

def subtract_templates_from_channel_block(raw_block,
                                          spike_times_block,
                                          x_shifts,
                                          templates,
                                          template_ids,
                                          start_sample,
                                          template_center=20,
                                          subtraction_window=(10, 30)):
    """
    Subtracts templates from a single-channel raw_block (1D), in-place.

    Parameters:
        raw_block         : [T] float32 array, voltage from one channel
        spike_times_block : array of global spike sample indices
        x_shifts          : array of per-spike x-shifts (same length as spike_times_block)
        templates         : list or 2D list of [T] templates
        template_ids      : list of template ID tuples (e.g. (i,j)) per spike
        start_sample      : global starting sample index of raw_block
        template_center   : int, index of spike peak within full template (default 20)
        subtraction_window: tuple (pre, post) relative to spike peak (default (10, 30))

    Returns:
        raw_block (modified in-place)
    """
    T = raw_block.shape[0]
    win_start, win_end = subtraction_window
    snippet_len = win_end - win_start + 1

    for i in range(len(spike_times_block)):
        global_time = spike_times_block[i]
        shift = x_shifts[i]
        template_id = template_ids[i]

        t_aligned = global_time + shift 
        t0 = t_aligned + win_start - start_sample
        t1 = t0 + snippet_len

        if t0 < 0 or t1 > T:
            continue  # skip spikes near edge

        if isinstance(template_id, tuple):
            full_template = templates[template_id[0]][template_id[1]]
        else:
            full_template = templates[template_id]

        cut_start = template_center + win_start
        cut_end = template_center + win_end + 1
        template_cut = full_template[cut_start:cut_end]

        if i < 0:
            t_range = np.arange(t0, t1)
            trace_before = raw_block[t0:t1].copy()

            plt.figure(figsize=(8, 3))
            plt.plot(t_range, trace_before, label='Raw Trace (Before)', color='gray')
            plt.plot(t_range, trace_before - template_cut, label='After Subtraction', color='black')
            plt.plot(t_range, template_cut, label='Template', color='red', linestyle='--')
            plt.title(f'Spike {i}: Template Subtraction Preview')
            plt.xlabel('Sample (local to block)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


        raw_block[t0:t1] -= template_cut

    return raw_block
