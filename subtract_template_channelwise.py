from scipy.optimize import minimize
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

def subtract_template_channelwise(channel_idx,
                                   fallback_params_per_spike,
                                   ei_trace,
                                   final_spike_times_cleaned,
                                   raw_data,
                                   t_template,
                                   t_peak_template,
                                   window=(-20, 60),
                                   fit_offsets=(-5, 10),
                                   plot_spikes=[],
                                   dtype=np.int16):


    n_total_samples, n_channels = raw_data.shape
    A_vals, w_vals, delta_vals, rms_vals = [], [], [], []
    fallback_used_flags = []

        # --- Filter spikes to be within segment ---
    valid_spikes = []
    for s in final_spike_times_cleaned:
        if (s + window[0] >= start_sample) and (s + window[1] < start_sample + segment_length):
            valid_spikes.append(s)
    valid_spikes = np.array(valid_spikes)


    for s in valid_spikes:
        t_start = s + window[0]
        t_end = s + window[1]
        snip_len = t_end - t_start + 1

        if t_start < 0 or t_end >= n_total_samples:
            A_vals.append(np.nan)
            w_vals.append(np.nan)
            delta_vals.append(np.nan)
            fallback_used_flags.append(True)
            continue

        snippet = raw_data[t_start:t_end + 1, channel_idx].astype(np.float32)
        snippet -= np.mean(snippet[:5])
        t = np.arange(snip_len)

        t0 = max(t_peak_template + fit_offsets[0], 0)
        t1 = min(t_peak_template + fit_offsets[1], snip_len - 1)

        # --- Amplitude-only match
        A_only = np.dot(snippet, ei_trace) / np.dot(ei_trace, ei_trace)
        template_amp_only = A_only * ei_trace
        rms_raw = np.sqrt(np.mean((snippet[t0:t1 + 1] - template_amp_only[t0:t1 + 1]) ** 2))

        def fit_error(params):
            A, w, b, delta = params
            t_shifted = (t - t_peak_template - delta) / w + t_peak_template
            t_shifted = np.clip(t_shifted, 0, len(ei_trace) - 1)
            warped = interp1d(t_template, ei_trace, kind='cubic', bounds_error=False)(t_shifted)
            return np.sum((snippet[t0:t1 + 1] - (A * warped[t0:t1 + 1] + b)) ** 2)

        try:
            res = minimize(fit_error, x0=[1.0, 1.0, 0.0, 0.0],
                           bounds=[(0.75, 1.25), (0.9, 1.1), (-500, 500), (-1.0, 1.0)],
                           method='L-BFGS-B')
            A_fit, w_fit, b_fit, delta_fit = res.x
            rms_fit = np.sqrt(res.fun / (t1 - t0 + 1))
            rms_improvement = (rms_raw - rms_fit) / rms_raw
        except Exception:
            rms_improvement = -np.inf

        if rms_improvement < 0.1:
            # --- Use fallback parameters
            if fallback_params_per_spike is not None:
                A_fit, w_fit, delta_fit = fallback_params_per_spike[s]
            else:
                A_fit = A_only
                w_fit = 1.0
                delta_fit = 0.0
            fallback_used = True
        else:
            fallback_used = False

        t_shifted = (t - t_peak_template - delta_fit) / w_fit + t_peak_template
        t_shifted = np.clip(t_shifted, 0, len(ei_trace) - 1)
        warped = interp1d(t_template, ei_trace, kind='cubic', bounds_error=False)(t_shifted)
        template_fit = A_fit * warped

        # --- Subtract from raw
        raw_snip = raw_data[t_start:t_end + 1, channel_idx].astype(np.float32)
        raw_data[t_start:t_end + 1, channel_idx] = np.clip(
            raw_snip - template_fit, -32768, 32767
        ).astype(dtype)

        # --- Plot diagnostic
        if s in plot_spikes:
            plt.figure(figsize=(10, 3))
            plt.plot(raw_snip, label='Original', color='black')
            plt.plot(template_fit, label='Template Fit', color='red')
            plt.plot(raw_snip - template_fit, label='Residual', color='green')
            plt.axvline(t_peak_template, linestyle='--', color='gray')
            plt.title(f"Channel {channel_idx}, Spike @ {s}, Fallback: {fallback_used}")
            plt.legend()
            plt.tight_layout()
            plt.show()

        A_vals.append(A_fit)
        w_vals.append(w_fit)
        delta_vals.append(delta_fit)
        fallback_used_flags.append(fallback_used)
        rms_vals.append(rms_raw)

    return {
        'A': np.array(A_vals),
        'w': np.array(w_vals),
        'delta': np.array(delta_vals),
        'rms_raw': np.array(rms_vals),
        'used_fallback': np.array(fallback_used_flags)
    }
