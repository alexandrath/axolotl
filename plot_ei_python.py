# Re-import necessary modules after code execution state reset
import numpy as np
import matplotlib.pyplot as plt

def plot_ei_python(ei, positions, frame_number=0, cutoff=0.03, scale=1.0, alpha=0.5,
                   neg_color='blue', pos_color='blue', label=None, ax=None,
                   scale_ref_electrode=None):
    """
    Plot an electrical image (EI) as a spatial map of electrode amplitudes.

    Parameters:
    - ei: [512, T] array of spike waveforms
    - positions: [512, 2] array of electrode (x, y) positions
    - frame_number: 0 for max projection, or integer frame index
    - cutoff: minimum absolute amplitude to plot
    - scale: multiplier relative to reference electrode (default = 1.0)
    - alpha: transparency of disks
    - neg_color: color for negative deflections
    - pos_color: color for positive deflections
    - label: list/array of electrode indices to label, or 'all', or None
    - ax: matplotlib Axes to draw into (optional)
    - scale_ref_electrode: index to use as reference for scaling; if None, uses max
    """

    n_elec, T = ei.shape
    assert positions.shape == (n_elec, 2)

    if frame_number == 0:
        ei_frame = ei[np.arange(n_elec), np.argmax(np.abs(ei), axis=1)]
    else:
        ei_frame = ei[:, frame_number]

    # Determine reference amplitude
    if scale_ref_electrode is not None:
        ref_amp = np.abs(ei_frame[scale_ref_electrode])
    else:
        ref_amp = np.max(np.abs(ei_frame))

    if ref_amp == 0:
        radii = np.zeros_like(ei_frame)
    else:
        raw_radii = scale * np.abs(ei_frame) / ref_amp
        radii = np.minimum(raw_radii, 1.0)
        radii[raw_radii < cutoff] = 0


    # Apply cutoff after normalization
    #radii[np.abs(ei_frame) < cutoff] = 0

    radii = radii*30

    ax = ax or plt.gca()
    ax.set_aspect('equal')
    ax.axis('off')

    for i in range(n_elec):
        if radii[i] == 0:
            continue
        color = neg_color if ei_frame[i] < 0 else pos_color
        circle = plt.Circle(positions[i], radii[i], color=color, alpha=alpha, edgecolor='none')
        ax.add_patch(circle)

        if label is not None:
            if isinstance(label, str) and label.lower() == 'all':
                ax.text(*positions[i], str(i+1), fontsize=6, ha='center', va='center')
            elif isinstance(label, (list, np.ndarray)) and i in label:
                ax.text(*positions[i], str(i+1), fontsize=6, ha='center', va='center')

    padding = np.max(radii) * 1.5 if np.any(radii) else 1
    min_x, max_x = positions[:, 0].min(), positions[:, 0].max()
    min_y, max_y = positions[:, 1].min(), positions[:, 1].max()
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
