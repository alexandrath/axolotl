# Re-import after code state reset
import numpy as np
import matplotlib.pyplot as plt

def plot_ei_waveforms(ei, positions, ref_channel=None, scale=1.0, ax=None,
                      colors='black', alpha=1.0, linewidth=0.5,
                      box_height=1.0, box_width=1.0):
    """
    Plot one or more EI waveforms overlaid at their spatial locations.

    Parameters:
    - ei: [512, T] array or list of such arrays
    - positions: [512, 2] array of electrode (x, y) positions
    - ref_channel: channel to highlight
    - scale: vertical scaling factor relative to box height (1.0 = max waveform fills box_height)
    - ax: matplotlib Axes (optional)
    - colors: single color or list of colors for each EI
    - alpha: line transparency
    - linewidth: width of waveform trace
    - box_height: vertical size of virtual box hosting the waveform
    - box_width: horizontal width of waveform (same for all electrodes)
    """

    ax = ax or plt.gca()
    ax.set_aspect('equal')
    ax.axis('off')

    # Handle single EI case by wrapping it in a list
    if isinstance(ei, np.ndarray):
        eis = [ei]
    else:
        eis = ei

    if isinstance(colors, str):
        colors = [colors] * len(eis)

    # Normalize all EIs by the max amplitude across all
    global_max = max(np.max(np.abs(e)) for e in eis)
    if global_max == 0:
        return

    t = np.linspace(-0.5, 0.5, eis[0].shape[1]) * box_width

    for ei_array, color in zip(eis, colors):
        norm_ei = (ei_array / global_max) * scale * box_height
        # Compute normalized P2P amplitudes per channel
        p2ps = norm_ei.max(axis=1) - norm_ei.min(axis=1)
        max_p2p = p2ps.max()
        p2p_thresh = 0.05 * max_p2p  # 10% of largest channel amplitude

        for i in range(ei_array.shape[0]):
            x_offset, y_offset = positions[i]
            y = norm_ei[i]
            if p2ps[i] < p2p_thresh:
                this_alpha = 0.4
                this_lw = 0.4
            else:
                this_alpha = alpha
                this_lw = linewidth
            if isinstance(ref_channel, int) and i == ref_channel:
                this_alpha = 1
                this_lw = linewidth*2

            ax.plot(t + x_offset, y + y_offset, color=color, alpha=this_alpha, linewidth=this_lw)



    pad_x, pad_y = box_width, box_height
    min_x, max_x = positions[:, 0].min(), positions[:, 0].max()
    min_y, max_y = positions[:, 1].min(), positions[:, 1].max()
    extra_y_pad = box_height * scale
    ax.set_ylim(min_y - pad_y - extra_y_pad, max_y + pad_y + extra_y_pad)

    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    #ax.set_ylim(min_y - pad_y, max_y + pad_y)
