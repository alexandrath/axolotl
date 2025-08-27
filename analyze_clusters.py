import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
from scipy.io import loadmat
from extract_data_snippets import extract_snippets
from plot_ei_python import plot_ei_python
from compute_sta_from_spikes import compute_sta_chunked
from benchmark_c_rgb_generation import RGBFrameGenerator
from compare_eis import compare_eis
from scipy.ndimage import gaussian_filter1d


def analyze_clusters(
        spike_times,
        sampling_rate,
        dat_path,
        h5_path,
        triggers_mat_path,
        clusters=None,
        *,  # force keyword‑only after this
        cluster_ids=None,
        lut=None,
        sta_depth=20,
        sta_offset=0,
        sta_chunk_size=1000,
        sta_refresh=2,
        ei_scale=3,
        ei_cutoff=0.05,
        isi_max_ms=200,
        template_ei=None,
        sigma_ms=2500,
        dt_ms=1000,
        anchor_frame=None):
    """Analyse clusters (or the full spike set) and visualise EI, ISI, rate and STA.

    Parameters
    ----------
    anchor_frame : int | None, optional
        If given, **use this STA depth index (0‑based) as the reference**:
        • Pixel chosen for the RGB time‑courses is the max‑|STA| pixel *in that frame*.
        • Displayed STA image is that frame.
        When *None* (default) the global max‑|STA| across all frames is used (old behaviour).
    Other parameters
    ----------------
    (See original docstring inside the canvas — unchanged except that *clusters*
    became optional in the previous revision.)
    """

    # ---------------- sanity checks & defaults ------------------------
    if spike_times is None or len(spike_times) == 0:
        raise ValueError("spike_times cannot be None/empty")

    if clusters is None:
        clusters = [{'inds': np.arange(len(spike_times))}]

    if lut is None:
        lut = np.array([
            [255, 255, 255],
            [255, 255,   0],
            [255,   0, 255],
            [255,   0,   0],
            [0,   255, 255],
            [0,   255,   0],
            [0,     0, 255],
            [0,     0,   0]
        ], dtype=np.uint8).flatten()

    # ---------------- load ancillary data ----------------------------
    with h5py.File(h5_path, 'r') as f:
        ei_positions = f['/ei_positions'][:].T  # [512,2]

    triggers_sec = loadmat(triggers_mat_path)['triggers'].flatten()

    generator = RGBFrameGenerator('/Volumes/Lab/Users/alexth/axolotl/sta/libdraw_rgb.so')
    generator.configure(width=20, height=40, lut=lut, noise_type=1, n_bits=3)

    # ---------------- cluster ordering -------------------------------
    if cluster_ids is None:
        sorted_clusters = sorted(enumerate(clusters), key=lambda x: len(x[1]['inds']), reverse=True)
    else:
        sorted_clusters = [(i, clusters[i]) for i in cluster_ids]

    if not sorted_clusters:
        raise ValueError("No clusters to analyse.")

    n_clusters = len(sorted_clusters)

    # ---------------- prepare figure ----------------------------------
    fig_height = 5
    fig = plt.figure(figsize=(4 * n_clusters, 8))
    gs = gridspec.GridSpec(fig_height, n_clusters, figure=fig)

    for col, (idx, cluster) in enumerate(sorted_clusters):
        inds = cluster['inds']
        if len(inds) == 0:
            print(f"Cluster {idx} empty — skipping.")
            continue

        spike_samples = spike_times[inds]
        spikes_sec = spike_samples / sampling_rate

        # ---------- EI --------------------------------------------------
        if isinstance(dat_path, np.ndarray):
            snips = dat_path[:, :, inds].copy()
        elif isinstance(dat_path, str):
            snips = extract_snippets(dat_path, spike_samples, window=(-20, 60), n_channels=512, dtype='int16')
        else:
            raise TypeError(f"Unsupported dat_path type: {type(dat_path)}")

        ei = np.mean(snips, axis=2)
        ei -= np.mean(ei[:, :5], axis=1, keepdims=True)

        # ---------- similarity to template -----------------------------
        sim_str = ""
        if template_ei is not None:
            T = min(ei.shape[1], template_ei.shape[1])
            sim = compare_eis([ei[:, :T]], template_ei[:, :T])[0, 0]
            sim_str = f"\nSim: {sim:.2f}"

        # ---------- EI plot -------------------------------------------
        ax_ei = fig.add_subplot(gs[0, col])
        ax_ei.set_title(f"Cluster {idx}\n({len(inds)} spikes){sim_str}", fontsize=10)

        if template_ei is not None:
            plot_ei_python(template_ei, ei_positions, scale=ei_scale, cutoff=ei_cutoff,
                           pos_color='blue', neg_color='blue', ax=ax_ei, alpha=0.5)
            ei_alpha = 0.5
        else:
            ei_alpha = 1
        plot_ei_python(ei, ei_positions, scale=ei_scale, cutoff=ei_cutoff,
                       pos_color='black', neg_color='red', ax=ax_ei, alpha=ei_alpha)

        # ---------- ISI histogram -------------------------------------
        ax_isi = fig.add_subplot(gs[1, col])
        if len(spikes_sec) > 1:
            isi = np.diff(spikes_sec)
            isi_max_s = isi_max_ms / 1000
            bins = np.arange(0, isi_max_s + 0.0005, 0.0005)
            hist, _ = np.histogram(isi, bins=bins)
            fractions = hist / hist.sum() if hist.sum() else np.zeros_like(hist)
            ax_isi.plot((bins[:-1] + bins[1:]) / 2, fractions, color='blue')
            ax_isi.set_xlim(0, isi_max_s)
            ax_isi.set_ylim(0, fractions.max() * 1.1)
        else:
            ax_isi.text(0.5, 0.5, "Not enough spikes", ha='center', va='center')
            ax_isi.set_xlim(0, 0.2)
            ax_isi.set_ylim(0, 1)
        ax_isi.set_title("ISI (s)", fontsize=10)

        # ---------- smoothed firing rate ------------------------------
        ax_rate = fig.add_subplot(gs[2, col])
        dt = dt_ms / 1000
        sigma_samples = sigma_ms / dt_ms
        total_duration = spikes_sec.max() + 0.1 if len(spikes_sec) else 1.0
        time_vector = np.arange(0, total_duration, dt)
        counts, _ = np.histogram(spikes_sec, bins=np.append(time_vector, total_duration))
        rate = gaussian_filter1d(counts / dt, sigma=sigma_samples)
        ax_rate.plot(time_vector, rate, color='darkorange')
        ax_rate.set_title("Smoothed Firing Rate", fontsize=10)
        ax_rate.set_xlim(0, total_duration)
        if col == 0:
            ax_rate.set_ylabel("Hz")

        # ---------- STA ----------------------------------------------
        sta = compute_sta_chunked(spikes_sec=spikes_sec,
                                  triggers_sec=triggers_sec,
                                  generator=generator,
                                  seed=11111,
                                  depth=sta_depth,
                                  offset=sta_offset,
                                  chunk_size=sta_chunk_size,
                                  refresh=sta_refresh)

        # pick reference frame & pixel
        if anchor_frame is not None:
            if not (0 <= anchor_frame < sta_depth):
                raise ValueError(f"anchor_frame {anchor_frame} out of range 0–{sta_depth-1}")
            frame_idx = anchor_frame
            frame = sta[:, :, :, frame_idx]
            # use max‑abs across channels as criterion
            abs_max = np.abs(frame).max(axis=2)
            y, x = np.unravel_index(abs_max.argmax(), abs_max.shape)
        else:
            max_idx = np.unravel_index(np.abs(sta).argmax(), sta.shape)
            y, x, _, frame_idx = max_idx

        # time courses (reverse so 0 is most recent)
        red_tc = sta[y, x, 0, :][::-1]
        green_tc = sta[y, x, 1, :][::-1]
        blue_tc = sta[y, x, 2, :][::-1]

        ax_tc = fig.add_subplot(gs[3, col])
        ax_tc.plot(red_tc, color='red', label='R')
        ax_tc.plot(green_tc, color='green', label='G')
        ax_tc.plot(blue_tc, color='blue', label='B')
        ax_tc.set_title("STA Time Course (reversed)", fontsize=10)
        ax_tc.set_xlim(0, sta_depth - 1)
        ax_tc.set_xticks([0, sta_depth - 1])
        if col == 0:
            ax_tc.set_ylabel("Intensity")

        # display chosen frame
        rgb = sta[:, :, :, frame_idx]
        vmax = np.abs(sta).max() * 2
        norm_rgb = np.clip(rgb / vmax + 0.5, 0, 1)

        ax_sta = fig.add_subplot(gs[4, col])
        ax_sta.imshow(norm_rgb.transpose(1, 0, 2), origin='upper')
        ax_sta.axis('off')
        ax_sta.set_title(f"STA Frame {frame_idx + 1}", fontsize=10)

    plt.tight_layout()
    plt.show()
