import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import os


def run_template_subtraction_on_channel(
    channel_idx,
    final_spike_times,
    ei_waveform,
    dat_path,
    start_sample,
    segment_length,
    dtype,
    n_channels,
    window,
    fit_offsets,
    fallback_fit_params=None,
    diagnostic_plot_spikes=None
):
    """
    Subtracts a warped EI template from a raw data segment on a specific channel.

    Parameters:
    - channel_idx: Index of the channel to process
    - final_spike_times: Array of all spike times (in samples)
    - ei_waveform: 1D array, the EI template for this channel
    - dat_path: Path to raw binary data file
    - start_sample: Start sample of the segment to process
    - segment_length: Number of samples to process
    - dtype: Data type of raw file (e.g. np.int16)
    - n_channels: Total number of channels
    - window: Tuple of (pre, post) samples around spike for fitting
    - fit_offsets: Tuple for fit range relative to peak
    - fallback_fit_params: dict {spike_time: [A, w, delta]} to use if RMS improvement is too low
    - diagnostic_plot_spikes: list of spike times to plot fits for

    Returns:
    - A list of fit parameters [A, w, delta] per spike (same order as valid_spikes)
    """

    pre, post = window
    snip_len = post - pre + 1

    ei_trace = ei_waveform.copy()
    ei_trace -= np.mean(ei_trace[:5])
    t_template = np.arange(len(ei_trace))
    t_peak_template = np.argmin(ei_trace)

    # Read segment
    segment_offset_bytes = start_sample * n_channels * np.dtype(dtype).itemsize
    segment_shape = (segment_length, n_channels)

    with open(dat_path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        file_len_bytes = f.tell()
    n_total_samples = file_len_bytes // (np.dtype(dtype).itemsize * n_channels)

    assert start_sample + segment_length <= n_total_samples, "Segment overflows file."

    # Step 1: Extract valid spikes
    valid_spikes = [s for s in final_spike_times
                    if (s + pre >= start_sample) and (s + post < start_sample + segment_length)]

    # Step 2: Open file in-place for modification
    raw_edit = np.memmap(dat_path, dtype=dtype, mode='r+', shape=(n_total_samples, n_channels))

    fit_params_per_spike = []

    for s in valid_spikes:
        t_start = s + pre
        t_end = s + post
        if t_start < 0 or t_end >= n_total_samples:
            continue

        snippet = raw_edit[t_start:t_end + 1, channel_idx].astype(np.float32)
        snippet -= np.mean(snippet[:5])
        t = np.arange(snip_len)

        t0 = max(t_peak_template + fit_offsets[0], 0)
        t1 = min(t_peak_template + fit_offsets[1], snip_len - 1)

        # Baseline amplitude-only fit
        A_only = np.dot(snippet, ei_trace) / np.dot(ei_trace, ei_trace)
        template_amp_only = A_only * ei_trace
        rms_raw = np.sqrt(np.mean((snippet[t0:t1 + 1] - template_amp_only[t0:t1 + 1]) ** 2))

        # Define fit error
        def fit_error(params):
            A, w, b, delta = params
            t_shifted = (t - t_peak_template - delta) / w + t_peak_template
            t_shifted = np.clip(t_shifted, 0, len(ei_trace) - 1)
            warped = interp1d(t_template, ei_trace, kind='cubic', bounds_error=False)(t_shifted)
            return np.sum((snippet[t0:t1 + 1] - (A * warped[t0:t1 + 1] + b)) ** 2)

        # Fit optimization
        res = minimize(fit_error, x0=[1.0, 1.0, 0.0, 0.0],
                       bounds=[(0.75, 1.25), (0.9, 1.1), (-500, 500), (-1.0, 1.0)])
        A_fit, w_fit, b_fit, delta_fit = res.x
        rms_fit = np.sqrt(res.fun / (t1 - t0 + 1))

        rms_improvement = (rms_raw - rms_fit) / rms_raw

        # Fallback if poor improvement
        if rms_improvement < 0.1 and fallback_fit_params is not None:
            if s in fallback_fit_params:
                A_fit, w_fit, delta_fit = fallback_fit_params[s]
            else:
                A_fit, w_fit, delta_fit = A_only, 1.0, 0.0
            template_fit = A_fit * ei_trace
        else:
            t_shifted = (t - t_peak_template - delta_fit) / w_fit + t_peak_template
            t_shifted = np.clip(t_shifted, 0, len(ei_trace) - 1)
            warped = interp1d(t_template, ei_trace, kind='cubic', bounds_error=False)(t_shifted)
            template_fit = A_fit * warped  # b_fit not used in subtraction

        # Diagnostic plot
        if diagnostic_plot_spikes is not None and s in set(diagnostic_plot_spikes):
            raw_snip = raw_edit[t_start:t_end + 1, channel_idx].astype(np.float32)
            residual = raw_snip - template_fit
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

        # Subtract
        raw_snip = raw_edit[t_start:t_end + 1, channel_idx].astype(np.float32)
        #raw_edit[t_start:t_end + 1, channel_idx] = np.clip(raw_snip - template_fit, -32768, 32767).astype(dtype)

        fit_params_per_spike.append([A_fit, w_fit, delta_fit])

    return fit_params_per_spike
