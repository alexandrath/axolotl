import matplotlib.pyplot as plt
import numpy as np

def plot_subtraction_overlay(raw_block, raw_block_subtracted,
                              start_sample_plot=0, end_sample_plot=1000,
                              channel_label="Channel"):
    """
    Overlays original and subtracted raw traces for a given sample range.

    Parameters:
        raw_block             : [T] float32, original trace
        raw_block_subtracted : [T] float32, subtracted trace
        start_sample_plot     : int, start index (inclusive)
        end_sample_plot       : int, end index (exclusive)
        channel_label         : optional string for plot title
    """
    t = np.arange(start_sample_plot, end_sample_plot)

    trace_orig = raw_block[start_sample_plot:end_sample_plot]
    trace_sub  = raw_block_subtracted[start_sample_plot:end_sample_plot]

    plt.figure(figsize=(18, 3))
    plt.plot(t, trace_orig, color='gray', linewidth=1, label='Original')
    plt.plot(t, trace_sub,  color='red', linewidth=1.0, label='Subtracted')
    plt.title(f"{channel_label} – Overlay of Original and Subtracted Traces")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
