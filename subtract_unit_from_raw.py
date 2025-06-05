import numpy as np
import h5py

def subtract_unit_from_raw(dat_path,
                           spike_times,
                           x_shifts,
                           templates,
                           template_ids,
                           ei_waveform,
                           unit_id,
                           target_channels,
                           start_sample,
                           total_samples,
                           template_center=20,
                           subtraction_window=(-10,30),
                           n_channels=512,
                           dtype=np.int16,
                           y_shifts=None): # optional dictionary: y_shifts[channel][i]
    """
    Subtract templates for a given unit from selected channels of a raw .dat file in-place.

    Parameters:
        dat_path            : path to modified .dat file (memmapped and overwritten)
        spike_times         : global spike times for this unit [N]
        x_shifts            : per-spike x-align shifts [N]
        templates           : [C][T x N_templates] or 2D list per channel
        template_ids        : list of template ID tuples per spike [(i,j), ...]
        ei_waveform         : [512 x T] EI waveform of the unit
        unit_id             : unit ID (for labeling)
        target_channels     : list of channels to subtract from
        start_sample        : sample offset of the raw block (usually 0)
        total_samples       : total number of samples in the file (to avoid overflow)
        template_center     : center index in the template (default: 20)
        subtraction_window  : tuple (pre, post), e.g., (10, 50)
        n_channels          : number of channels in the raw data file
        dtype               : data type, usually np.int16
    """
    win_start, win_end = subtraction_window
    snip_len = win_end - win_start + 1
    dat = np.memmap(dat_path, dtype=dtype, mode='r+', order='C')

    for chan in target_channels:
        #print(f"Subtracting unit {unit_id} from channel {chan}...")

        # Step 1: Get channel-specific template grid
        channel_templates = templates[chan]

        for i in range(len(spike_times)):
            global_time = spike_times[i]
            shift = x_shifts[i]
            template_id = template_ids[chan][i]

            # Step 2: Align spike to reference
            t_aligned = global_time + shift
            t0 = t_aligned + win_start
            t1 = t0 + snip_len

            # Step 3: Skip spikes that fall outside data bounds
            if t0 < 0 or t1 >= total_samples:
                continue

            # Step 4: Get channel-specific template
            if isinstance(template_id, tuple):
                full_template = channel_templates[template_id[0]][template_id[1]].copy()
            else:
                full_template = channel_templates[template_id].copy()
            if i <0 and chan == 125:
                import matplotlib.pyplot as plt

                t_full = np.arange(len(full_template))

                plt.figure(figsize=(6, 3))
                plt.plot(t_full, full_template, color='purple')
                plt.title(f"Full Template (before cut), Channel {chan}, Spike {i}")
                plt.xlabel("Time (samples)")
                plt.ylabel("Amplitude")
                plt.grid(True)
                plt.tight_layout()
                plt.show()


            cut_start = template_center + win_start
            cut_end = template_center + win_end + 1
            template_cut = full_template[cut_start:cut_end].copy()

            y_offset = 0
            if y_shifts is not None:
                y_offset = y_shifts[chan][i]
                template_cut += y_offset



            # Step 5: Compute indices for raw data modification
            sample_indices = np.arange(t0, t1)
            indices = sample_indices * n_channels + chan

            if i <0 and chan == 125:

                import matplotlib.pyplot as plt

                # Make a copy to avoid mutation
                raw_segment_copy = dat[indices].astype(np.float32)
                template_pre_yshift = full_template[cut_start:cut_end].copy()
                template_post_yshift = template_pre_yshift + y_offset
                trace_after = raw_segment_copy - template_post_yshift

                t = np.arange(len(template_pre_yshift))

                fig, axs = plt.subplots(1, 2, figsize=(10, 4))

                # Left: raw vs subtracted trace
                axs[0].plot(t, raw_segment_copy, label='Raw', color='black')
                axs[0].plot(t, trace_after, label='Subtracted', color='red')
                axs[0].set_title("Raw vs Subtracted")
                axs[0].legend()
                axs[0].grid(True)

                # Right: template before and after y-shift
                axs[1].plot(t, template_pre_yshift, label='Template (pre)', color='gray')
                axs[1].plot(t, template_post_yshift, label='Template (+y_shift)', color='blue')
                axs[1].set_title(f"Template Shifted, y_offset = {y_offset:.2f}")
                axs[1].legend()
                axs[1].grid(True)

                plt.tight_layout()
                plt.show()

            # Step 6: Subtract template from raw data
            raw_segment = dat[indices].astype(np.float32)
            raw_segment -= template_cut
            dat[indices] = np.clip(raw_segment, -32768, 32767).astype(np.int16)




    print(f"Unit {unit_id} subtraction complete.")
