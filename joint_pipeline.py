# %% [markdown]
# # Joint pipeline development

# %% [markdown]
# ## Load data

# %%
import numpy as np
import os
import h5py
import json
from compare_eis import compare_eis
from scipy.io import loadmat

# --- Path and recording setup ---
dat_path = "/Volumes/Lab/Users/alexth/axolotl/201703151_data001.dat"
n_channels = 512
dtype = np.int16

# --- Get total number of samples ---
file_size_bytes = os.path.getsize(dat_path)
total_samples = file_size_bytes // (np.dtype(dtype).itemsize * n_channels)

# --- Load entire file into RAM as int16 ---
raw_data = np.fromfile(dat_path, dtype=dtype, count=total_samples * n_channels)
raw_data = raw_data.reshape((total_samples, n_channels))  # shape: [T, C]


# --- Parameters ---
n_channels = 512
dtype = 'int16'
max_units = 1500
amplitude_threshold = 15
window = (-20, 60)
peak_window = 30
total_samples=36_000_000
fit_offsets = (-5, 10)

do_pursuit = 0


h5_in_path = '/Volumes/Lab/Users/alexth/axolotl/201703151_kilosort_data001_spike_times.h5'  # from MATLAB export, to get EI positions


with h5py.File(h5_in_path, 'r') as f:
    # Load electrode positions
    ei_positions = f['/ei_positions'][:].T  # shape becomes [512 x 2]
    ks_vision_ids = f['/vision_ids'][:]  # shape: (N_units,)

import axolotl_utils_ram
import importlib
importlib.reload(axolotl_utils_ram)

baseline_path = "/Volumes/Lab/Users/alexth/axolotl/201703151_data001_baseline_derivative_20k.json"

segment_len = 20_000
if os.path.exists(baseline_path):
    print(f"Loading baselines")
    with open(baseline_path, 'r') as f:
        data = json.load(f)
    baselines = np.array(data['baselines'], dtype=np.float32)
else:
    print(f"Computing baselines")
    baselines = axolotl_utils_ram.compute_baselines_int16_deriv_robust(raw_data, segment_len=segment_len, diff_thresh=10, trim_fraction=0.15) # shape (512, 360)

    with open(baseline_path, 'w') as f:
        json.dump({
            'baselines': baselines.tolist(),
        }, f)

print("subtracting baselines")

axolotl_utils_ram.subtract_segment_baselines_int16(raw_data=raw_data,
                                     baselines_f32=baselines,
                                     segment_len=segment_len) 

triggers_mat_path='/Volumes/Lab/Users/alexth/axolotl/trigger_in_samples_201703151.mat'
triggers_sec = loadmat(triggers_mat_path)['triggers'].flatten()


# %% [markdown]
# ## Imports

# %%
import numpy as np
import matplotlib.pyplot as plt
import importlib

import axolotl_utils_ram
importlib.reload(axolotl_utils_ram)

import plot_ei_waveforms
importlib.reload(plot_ei_waveforms)


import collision_utils
importlib.reload(collision_utils)

import plot_ei_waveforms as pew

from axolotl_utils_ram import extract_snippets_fast_ram

from collision_utils import (
        select_template_channels,
        main_channel_and_neg_peak,
        compute_harm_map_noamp,
        plot_harm_heatmap,
        scan_continuous_harm_regions,
        median_ei_adaptive,
)

import joint_utils
importlib.reload(joint_utils)

from joint_utils import (
    split_first_good_channel_and_visualize,
    classify_two_cells_vs_ab_shard,
    plot_grouped_harm_maps_two_eis,
    check_bimodality_and_plot,
    recenter_ei_to_ref_trough,
    compute_ei_from_indices,
    assess_ei_drift,
    kmeans_split_diagnostics,
    detect_negative_peaks_on_channel,
    select_top_by_amplitude,
    extract_and_build_ei,
    harm_map_diagnostic,
    amplitude_gate_pruner,
    two_cells_or_ab_shard_pruner,
    harm_compliance_pruner,
    bimodality_split_pruner,
    make_mask_entry,
    mask_trace_with_template_bank,
    save_mask_bank_pickle,
    load_mask_bank_pickle,
    find_ks_ax_matches,
    best_crossmatches
    
)

from matplotlib.backends.backend_pdf import PdfPages

import contextlib, gc, warnings

import io
import textwrap


# %% [markdown]
# ### Context - save figures to PDF

# %%
# === FigureCollector: capture ALL Matplotlib figures created inside a block ===
class FigureCollector:
    """
    Capture all Matplotlib figures created while active, even if utils reuse numbers
    or import pyplot functions (figure/subplots/close) by name.

    Strategy:
      1) Hard-reset any preexisting figs (using original close).
      2) Make plt.show/plt.close no-ops during capture.
      3) Intercept matplotlib._pylab_helpers.Gcf.destroy -> no-op during capture.
      4) Wrap plt.figure/plt.subplots to record new figure births.
    """
    def __init__(self):
        self.figures = []
        self._pre_ids = None
        self._created = None
        self._orig_show = None
        self._orig_close = None
        self._orig_figure = None
        self._orig_subplots = None
        self._orig_gcf_destroy = None

    def __enter__(self):
        import matplotlib.pyplot as plt
        from matplotlib import _pylab_helpers as pylab_helpers

        plt.ioff()

        # Save originals
        self._orig_show = plt.show
        self._orig_close = plt.close
        self._orig_figure = plt.figure
        self._orig_subplots = plt.subplots
        self._orig_gcf_destroy = pylab_helpers.Gcf.destroy

        # Hard reset any lingering figs BEFORE we override destroy
        try:
            self._orig_close('all')
        except Exception:
            pass

        # Snapshot after reset
        self._pre_ids = set(plt.get_fignums())

        # No-op show/close during capture window
        def _noop(*args, **kwargs):
            return None
        plt.show = _noop
        plt.close = _noop

        # Block all destruction, even via "from matplotlib.pyplot import close"
        pylab_helpers.Gcf.destroy = staticmethod(lambda *args, **kwargs: None)

        # Record births
        self._created = set()

        def _wrapped_figure(*args, **kwargs):
            fig = self._orig_figure(*args, **kwargs)
            try:
                self._created.add(fig.number)
            except Exception:
                pass
            return fig

        def _wrapped_subplots(*args, **kwargs):
            fig, axes = self._orig_subplots(*args, **kwargs)
            try:
                self._created.add(fig.number)
            except Exception:
                pass
            return fig, axes

        plt.figure = _wrapped_figure
        plt.subplots = _wrapped_subplots

        return self

    def __exit__(self, exc_type, exc, tb):
        import matplotlib.pyplot as plt
        from matplotlib import _pylab_helpers as pylab_helpers

        # Gather open figs and also any we know were birthed
        post_ids = set(plt.get_fignums())
        new_ids = sorted((post_ids - self._pre_ids) | self._created)
        self.figures = [plt.figure(fid) for fid in new_ids]

        # Restore pyplot and core destroy
        plt.figure = self._orig_figure
        plt.subplots = self._orig_subplots
        plt.show = self._orig_show
        plt.close = self._orig_close
        pylab_helpers.Gcf.destroy = self._orig_gcf_destroy

        # do not suppress exceptions
        return False


# === LogCollector: capture prints, stderr, and warnings during a run ===
class LogCollector:
    """
    Context manager that captures stdout (print), optionally stderr, and warnings.
    The captured text is available as `.text` on exit.
    """
    def __init__(self, capture_stderr=True, capture_warnings=True):
        self.capture_stderr = capture_stderr
        self.capture_warnings = capture_warnings
        self._buf = io.StringIO()
        self._redir_out = contextlib.redirect_stdout(self._buf)
        self._redir_err = contextlib.redirect_stderr(self._buf) if capture_stderr else contextlib.nullcontext()
        self._warn_cm = None
        self._warn_records = None
        self.text = ""

    def __enter__(self):
        self._redir_out.__enter__()
        self._redir_err.__enter__()
        if self.capture_warnings:
            self._warn_cm = warnings.catch_warnings(record=True)
            self._warn_records = self._warn_cm.__enter__()
            warnings.simplefilter("always")
        return self

    def __exit__(self, exc_type, exc, tb):
        # Append warnings to the buffer (after the run)
        if self.capture_warnings and self._warn_records is not None:
            for w in self._warn_records:
                try:
                    self._buf.write(f"[warning] {w.category.__name__}: {w.message}\n")
                except Exception:
                    pass
            self._warn_cm.__exit__(exc_type, exc, tb)
        # Restore streams
        self._redir_err.__exit__(exc_type, exc, tb)
        self._redir_out.__exit__(exc_type, exc, tb)
        self.text = self._buf.getvalue()
        return False


# %% [markdown]
# ### LOOP OVER CHANNELS

# %%
def proto_to_kmeans_for_channel(ch, params=None, rng=None):
    """
    Run the full proto→k-means sequence for a single reference channel `ch`.

    Returns:
        result : dict with a few summary fields (safe-to-miss if not set)
        figs   : list[matplotlib.figure.Figure] created during the run
    """
    if params is None:
        params = {}

    # Mild defaults; adjust later in your batch driver if you want
    MIN_PEAKS = int(params.get("min_peaks", 30))

    # Figure capture: grab ALL plots spawned anywhere inside this block
    with FigureCollector() as _fc:
        with LogCollector(capture_stderr=True, capture_warnings=True) as _lc:
            # ------------------------------------------------------------------
            # BEGIN: paste your existing code starting at "### Find proto template"
            #        and ending after "### Check if template explained more than one cell (k-means)"
            #


            # ---------- Config ----------
            thr         = -200.0       # negative threshold (same units as raw)
            max_events  = 400          # first N events in time
            min_gap     = 40           # enforce ≥40-sample separation
            pre, post   = 20, 40       # window relative to the negative peak (inclusive)
            to_select   = 100           # events to select for template

            # ---- Config for harm map ----
            p2p_thr      = 30.0     # channel selection threshold by p2p
            max_channels = 80       # hard cap
            min_channels = 10       # ensure at least this many
            lag_radius   = 3        # ± samples for lag scan
            win          = (-40, 80)

            # =========================
            # Config for harm map
            # =========================
            MEAN_THR  = -2.0    # mean ΔRMS across selected channels must be <= this
            CHAN_THR  = 15.0    # any channel ΔRMS > this -> reject spike
            REF_THR   = -5.0    # ref-channel ΔRMS must be < this
            LAG_RADIUS = 0      # no micro-shift; keep it literal
            MAX_ITERS  = 10     # safety stop



            # ---------- Source trace (EDIT THIS LINE to your array name) ----------
            trace = raw_data[:,ch].astype(np.float32).T  # raw: shape (n_channels, n_samples)

            # ---------- Find negative-peak events crossing threshold ----------
            x = trace
            # local minima below threshold
            cand = np.where((x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:]) & (x[1:-1] <= thr))[0] + 1
            # enforce min_gap and window boundaries, keep first max_events in time
            picked = []
            last = -10**9
            for i in cand:
                if i - last >= min_gap and (i - pre) >= 0 and (i + post) < x.size:
                    picked.append(i)
                    last = i
                    if len(picked) >= max_events:
                        break
            picked = np.array(picked, dtype=int)

            if picked.size == 0:
                print("No events found—check threshold/sign/channel.")
            else:
                # ---------- Extract waveforms ----------
                wf = np.stack([x[i - pre : i + post + 1] for i in picked], axis=0)  # shape (N, pre+post+1)
                t = np.arange(-pre, post + 1)

                # ---------- Select top-N by amplitude on ref channel (prescreen) ----------
                # Use negative-peak magnitude within the local window on channel `ch`
                amp_all = -wf.min(axis=1)  # shape (N,)

                k = min(to_select, wf.shape[0])
                sel_idx = np.argsort(-amp_all)[:k]   # indices of top-N by amplitude

                wf_sel  = wf[sel_idx]
                med_sel = np.median(wf_sel, axis=0)


                # ---------- Plots ----------
                fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)

                ax = axes[0]
                ax.plot(t, wf.T, linewidth=0.5, alpha=0.4)
                ax.set_title(f"All waveforms (N={wf.shape[0]})  | ch {ch}")
                ax.set_xlabel("samples from peak"); ax.set_ylabel("ADC")

                ax = axes[1]
                ax.plot(t, wf_sel.T, linewidth=0.5, alpha=0.55, label=f"selected {k}")
                ax.plot(t, med_sel, linewidth=2.5, label="median(selected)")
                ax.set_title("Selected top-N by amplitude + median")
                ax.set_xlabel("samples from peak")

                plt.tight_layout()
                # plt.show()

                # ===== EI from selected spikes (full array) =====

                # Use your selected spike times on ch
                times_sel = picked[sel_idx].astype(np.int64)

                # EI window (use your standard full EI window)
                window_ei = (-40, 80)   # center = -window_ei[0]
                C_full    = raw_data.shape[1]
                all_ch    = np.arange(C_full, dtype=int)

                # Extract snippets on ALL channels at those times
                snips_ei, valid_times = extract_snippets_fast_ram(raw_data, times_sel, window_ei, all_ch)
                if snips_ei.shape[2] == 0:
                    raise RuntimeError("No valid snippets for EI (check edges / times).")

                # EI = mean across selected events
                ei_sel_full = median_ei_adaptive(snips_ei)   # [C_full, L_ei]


                # Identify main channel by the most-negative trough on the current EI
                try:
                    ch0, t_neg0 = main_channel_and_neg_peak(ei_sel_full)
                except Exception:
                    # fallback: same logic inline
                    mins = ei_sel_full.min(axis=1)
                    ch0  = int(np.argmin(mins))
                    t_neg0 = int(np.argmin(ei_sel_full[ch0]))

                print(f"Main channel: {ch0}")
                amps = (-snips_ei[ch0, t_neg0, :]).astype(np.float32)   # positive trough magnitude per spike
                if amps.size:
                    TOPK = 25
                    FRAC = 0.75
                    k = min(TOPK, amps.size)
                    mu_top = float(np.mean(np.sort(amps)[-k:]))           # mean of top-k trough magnitudes
                    thr_amp = FRAC * mu_top
                    keep_amp = amps >= thr_amp
                    n_drop = int((~keep_amp).sum())
                    if n_drop > 0:
                        print(f"amplitude gate on main ch {ch0} @t={t_neg0}: "
                                f"drop {n_drop}/{amps.size} (μ_top{k}={mu_top:.1f} → thr={thr_amp:.1f})")
                        snips_ei = snips_ei[:, :, keep_amp]
                        if snips_ei.shape[2] == 0:
                            print("amplitude gate removed all spikes; stopping.")
                        else: # Rebuild EI on the reduced set to keep harm-map inputs consistent
                            ei_sel_full = median_ei_adaptive(snips_ei)

                fig, axes = plt.subplots(1, 3, figsize=(15, 3), sharey=True)

                ax = axes[0]
                ax.plot(snips_ei[ch0], linewidth=0.5, alpha=0.4)
                ax.set_title(f"All waveforms on main ch {ch0}")
                ax.set_xlabel("samples from peak"); ax.set_ylabel("ADC")

                ax = axes[1]
                ax.plot(snips_ei[ch], linewidth=0.5, alpha=0.4)
                ax.set_title(f"All waveforms on ref ch {ch}")
                ax.set_xlabel("samples from peak"); ax.set_ylabel("ADC")

                ax = axes[2]
                extra_ch_to_plot = 330
                ax.plot(snips_ei[extra_ch_to_plot], linewidth=0.5, alpha=0.4)
                ax.set_title(f"All waveforms on extra ch {extra_ch_to_plot}")
                ax.set_xlabel("samples from peak"); ax.set_ylabel("ADC")

                plt.tight_layout()
                # plt.show()


                # Plot the EI (overlay-ready if you add more later)
                fig, ax = plt.subplots(1, 1, figsize=(10,12))
                plot_ei_waveforms.plot_ei_waveforms(
                    ei_sel_full, ei_positions, ref_channel=int(ch), ax=ax,
                    colors='C2', scale=70.0, box_height=1.0, box_width=50.0
                )
                ax.set_title(f"EI from selected spikes on ch {int(ch)} (k={len(sel_idx)})")
                plt.tight_layout(); 
                # plt.show()




                # 1) Preselect channels from EI (same rule the harm-map uses)
                chans_pre, ptp = select_template_channels(
                    ei_sel_full, p2p_thr=p2p_thr, max_n=max_channels, min_n=min_channels, force_include_main=True
                )

                # Make sure the main (most-negative) channel is present
                ch_main, _ = main_channel_and_neg_peak(ei_sel_full)
                if ch_main not in chans_pre:
                    # replace weakest with main, re-sort by p2p desc, keep unique
                    pool = np.concatenate([chans_pre[:-1], [ch_main]])
                    chans_pre = np.array(sorted(set(pool), key=lambda c: ptp[c], reverse=True), dtype=int)

                print(f"channels used: {len(chans_pre)} (main={int(ch_main)})")

                # 3) Harm-map on the reduced channel set
                #    Lock selection inside harm-map to use *exactly* these channels by setting p2p_thr very low and min=max=len(chans_pre).
                res = compute_harm_map_noamp(
                    ei                   = ei_sel_full[chans_pre],   # [K, L]
                    snips                = snips_ei[chans_pre],            # [K, L, N]
                    p2p_thr              = -1e9,                 # include all provided chans
                    max_channels         = len(chans_pre),
                    min_channels         = len(chans_pre),
                    lag_radius           = lag_radius,
                    weight_by_p2p        = True,
                    weight_beta          = 0.7,
                    force_include_main   = True,
                )

                # 4) Plot
                plot_harm_heatmap(res, field="harm_matrix",
                                title=f"Harm map ΔRMS | ch {int(ch)} | N={valid_times.size}")


                # Work on a copy of your current snippets
                snips_cur = snips_ei.copy()
                C_avail, L, N0 = snips_cur.shape

                n_init = int(snips_cur.shape[2])
                if n_init < MIN_PEAKS:
                    return {"status":"skip_insufficient_peaks","ch":int(ch),
                            "n_init":n_init, "n_final":0}, []


                # === Split & visualize based on first good channel (post-explain) ===
                res_split = None
                res_split = split_first_good_channel_and_visualize(
                        snips_cur,                 # 512 x 121 x 661 (explained spikes, all channels)
                        ei_sel_full,      # 512 x 121 (mean over explained spikes), centered
                        ei_positions,
                        rms_thr=10.0,
                        dprime_thr=5.0,
                        min_per_cluster=10,
                        n_init=8, max_iter=60,
                        lag_radius=0
                    )
                
                if res_split is not None:
                    metrics = classify_two_cells_vs_ab_shard(
                        res_split['EI0'], res_split['EI1'], snips_cur, res_split['idx0'], res_split['idx1'],
                        p2p_thr=30.0, max_channels=80, min_channels=10,
                        lag_radius=0, weight_by_p2p=True, weight_beta=0.7,
                        rms_thr_support=10.0,
                        asym_strong_z=2.0,  # tighten/loosen if needed
                        asym_pure_z=1.0
                    )

                    if metrics["label"] == "two cells":
                        if len(res_split['idx0'])>=len(res_split['idx1']):
                            snips_cur = snips_cur[:,:,res_split['idx0']]
                        else:
                            snips_cur = snips_cur[:,:,res_split['idx1']]

                C_avail, L, N0 = snips_cur.shape

                print(f"[start] snippets: C={C_avail}, L={L}, N={N0}")

                final_res = None
                final_ei_full = None
                bimodal_plotted = False
                bimodal_payload = None   # store first detected bimodal split to finalize at the end




                for it in range(1, MAX_ITERS + 1):
                    # --- EI from current spikes (no lags, straight mean) ---
                    ei_full = median_ei_adaptive(snips_cur)     # [C_avail, L]
                

                    # --- Harm map with fresh channel selection each round ---
                    res_it = compute_harm_map_noamp(
                        ei=ei_full, snips=snips_cur,
                        p2p_thr=50.0,               # your usual selector; adjust if needed
                        max_channels=C_avail,       # allow selector to choose from what's available
                        min_channels=10,
                        lag_radius=LAG_RADIUS,
                        weight_by_p2p=True, weight_beta=0.7,
                        force_include_main=True
                    )
                    HM    = np.asarray(res_it["harm_matrix"])                 # [K_sel, N_cur]
                    sel   = np.asarray(res_it["selected_channels"], int)      # [K_sel]
                    K_sel, N_cur = HM.shape

                    # --- Build the reject mask from your three rules ---
                    mean_d  = HM.mean(axis=0)          # [N_cur]
                    max_d   = HM.max(axis=0)           # [N_cur]

                    # ref channel row (if not selected, we can't check REF_THR; warn & skip that term)
                    ref_matches = np.where(sel == int(ch))[0]
                    if ref_matches.size == 0:
                        print(f"[iter {it}] WARNING: ref ch {int(ch)} not in selected_channels this round; skipping REF_THR check.")
                        ref_ok = np.ones(N_cur, dtype=bool)
                    else:
                        ref_row = int(ref_matches[0])
                        ref_d   = HM[ref_row]          # [N_cur]
                        ref_ok  = (ref_d < REF_THR)

                    keep = (mean_d <= MEAN_THR) & (max_d <= CHAN_THR) & ref_ok
                    n_bad = int((~keep).sum())
                    print(f"[iter {it}] N={N_cur} | reject={n_bad} "
                        f"(mean>{MEAN_THR}: {int((mean_d>MEAN_THR).sum())}, "
                        f"any>{CHAN_THR}: {int((max_d>CHAN_THR).sum())}, "
                        f"ref>={REF_THR}: {int((~ref_ok).sum())})")

                    # --- Debug: which channels cause the max-Δ failures? ---
                    if n_bad > 0:
                        from collections import Counter
                        sel_global = np.asarray(res_it["selected_channels"], dtype=int)   # rows → global channel ids
                        bad_spikes = np.where(max_d > CHAN_THR)[0]                        # failing spike indices
                        # argmax row for each failing spike (row is within selected_channels)
                        row_of_max = np.argmax(HM[:, bad_spikes], axis=0)
                        ch_of_max  = sel_global[row_of_max]                               # map to global channel ids
                        val_of_max = HM[row_of_max, bad_spikes]

                        # summary: top offending channels by count
                        counts = Counter(ch_of_max.tolist()).most_common(15)
                        print(f"[iter {it}] channels hitting max Δ>{CHAN_THR} (top 15):")
                        for ch_id, cnt in counts:
                            print(f"  ch {int(ch_id):4d}  →  {cnt} spikes")

                        # first few concrete examples (spike index, channel, Δ value)
                        head = min(10, bad_spikes.size)
                        for s_idx, ch_id, val in zip(bad_spikes[:head], ch_of_max[:head], val_of_max[:head]):
                            print(f"    spike {int(s_idx):5d}: ch {int(ch_id):4d}, Δ={float(val):.2f}")


                    if n_bad == 0:
                        # default: no split, use current survivors as final
                        final_res = res_it
                        final_ei_full = ei_full

                        # One more bimodality check if we never looked (e.g., no rejections happened)
                        if bimodal_payload is None:
                            tmp = check_bimodality_and_plot(
                                snips_cur, res_it, ei_positions, ref_ch=int(ch),
                                dprime_thr=5.0, min_per_cluster=5
                            )
                            if tmp and tmp.get("hit", False):
                                bimodal_payload = tmp

                        if bimodal_payload and bimodal_payload.get("hit", False):
                            # expose cohort EIs for inspection
                            ei_lo_bimodal = bimodal_payload["ei_lo"]
                            ei_hi_bimodal = bimodal_payload["ei_hi"]
                            idx_lo = bimodal_payload["idx_lo"]
                            idx_hi = bimodal_payload["idx_hi"]

                            # choose cohort by larger N
                            # amp_lo = float(-ei_lo_bimodal[int(ch)].min())
                            # amp_hi = float(-ei_hi_bimodal[int(ch)].min())
                            pick_hi = (len(idx_hi) >= len(idx_lo))
                            chosen_idx = idx_hi if pick_hi else idx_lo

                            # --- guard against stale/out-of-range indices ---
                            chosen_idx = np.asarray(chosen_idx, dtype=np.int64).ravel()
                            Ncur = snips_cur.shape[2]
                            bad = (chosen_idx < 0) | (chosen_idx >= Ncur)
                            if bad.any():
                                print(f"[bimodality] WARNING: {bad.sum()} invalid indices (min={chosen_idx.min()}, "
                                    f"max={chosen_idx.max()}, N={Ncur}). Filtering.")
                                chosen_idx = chosen_idx[~bad]

                            if chosen_idx.size == 0:
                                print("[bimodality] cohort empty after filtering; skipping split.")
                                final_res = res_it
                                final_ei_full = ei_full
                            else:
                                # restrict survivors to chosen cohort and recompute EI + harm-map
                                snips_fin = snips_cur[:, :, chosen_idx]
                                final_ei_full = median_ei_adaptive(snips_fin)

                            


                            final_res = compute_harm_map_noamp(
                                ei=final_ei_full, snips=snips_fin,
                                p2p_thr=50.0, max_channels=snips_fin.shape[0], min_channels=10,
                                lag_radius=LAG_RADIUS, weight_by_p2p=True, weight_beta=0.7,
                                force_include_main=True
                            )

                            # plots: new harm-map and the final EI
                            plot_harm_heatmap(
                                final_res, field="harm_matrix",
                                title=f"Final harm map (post-bimodal split; picked {'high' if pick_hi else 'low'} amp cohort)"
                            )
                            try:
                                fig, ax = plt.subplots(figsize=(20, 12))
                                pew.plot_ei_waveforms(
                                    final_ei_full, ei_positions,
                                    ref_channel=int(ch), scale=70.0, box_height=1.0, box_width=50.0, ax=ax
                                )
                                ax.set_title(f"Final EI | ch {int(ch)} | cohort={'high' if pick_hi else 'low'}")
                                # plt.show()
                            except Exception as e:
                                print(f"[final EI] plotting skipped: {e}")

                        print("[done] All events satisfy thresholds.")
                        break



                    # Kick failing spikes, recompute everything fresh next round
                    snips_cur = snips_cur[:, :, keep]
                    if snips_cur.shape[2] == 0:
                        print(f"[iter {it}] all spikes rejected; stopping.")
                        final_res = res_it
                        final_ei_full = ei_full
                        break

                    print(f"[iter {it}] kept {snips_cur.shape[2]}/{N_cur} after harm-map pruning")

                    # Bimodality check on the post-kick set (capture FIRST hit only)
                    if bimodal_payload is None:
                        # Recompute a quick harm-map on the trimmed set so channel selection is accurate
                        res_tmp = compute_harm_map_noamp(
                            ei=median_ei_adaptive(snips_cur),
                            snips=snips_cur,
                            p2p_thr=50.0, max_channels=snips_cur.shape[0], min_channels=10,
                            lag_radius=LAG_RADIUS, weight_by_p2p=True, weight_beta=0.7,
                            force_include_main=True
                        )
                        tmp = check_bimodality_and_plot(
                            snips_cur, res_tmp, ei_positions, ref_ch=int(ch),
                            dprime_thr=5.0, min_per_cluster=5
                        )
                        if tmp and tmp.get("hit", False):
                            bimodal_payload = tmp




                else:
                    # fell out by MAX_ITERS
                    final_res = res_it
                    final_ei_full = ei_full
                    print("[stop] Reached MAX_ITERS; some violations may remain.")

                # =========================
                # Plots: final harm map & EI
                # =========================
                if final_res is not None:
                    plot_harm_heatmap(final_res, field="harm_matrix", sort_by_ptp=False,
                                    title=f"Final harm map after pruning "
                                            f"(K={len(final_res['selected_channels'])}, N={final_res['harm_matrix'].shape[1]})")

                if final_ei_full is not None:
                    # If snips_cur covered the full array, this is full; otherwise it covers available channels only.
                    C_full = ei_positions.shape[0]
                    L_ei   = final_ei_full.shape[1]
                    ei_plot = (final_ei_full if final_ei_full.shape[0] == C_full
                            else np.pad(final_ei_full, ((0, C_full - final_ei_full.shape[0]), (0, 0)), mode='constant'))

                    fig, ax = plt.subplots(1, 1, figsize=(10,12))
                    pew.plot_ei_waveforms(ei_plot, ei_positions, ref_channel=int(ch), ax=ax,
                                        colors='C2', scale=70.0, box_height=1.0, box_width=50.0)
                    ax.set_title(f"Final EI after pruning (events kept = {final_res['harm_matrix'].shape[1] if final_res else 'n/a'})")
                    plt.tight_layout(); 
                    # plt.show()


            n_final = int(chosen_idx.size) if 'chosen_idx' in locals() else int(snips_cur.shape[2])

            results = scan_continuous_harm_regions(
                raw_data=raw_data.T,
                final_ei_full=final_ei_full,
                start_sample=0,
                stop_sample=2_000_000,   # exclusive
                mean_thr=-2.0,
                chan_thr=10.0,
                ref_thr=-5.0,
                center_index=40,
                region_max_len=20,
            )

            harm = results["harm_at_best"]
            n_sel, n_regions = harm.shape
            print(f"Spikes: {n_regions}, Channels: {n_sel}")



            # 1) Plot compact harm map (ΔRMS per selected channel at each region’s best sample)
            harm = results["harm_at_best"]             # (n_sel, n_regions)
            sel_ch = results["selected_channels"]      # (n_sel,)
            ref_ch = int(results["ref_channel"])      # integer channel id
            v = np.nanpercentile(np.abs(harm), 95) if harm.size else 1.0


            # harm: (n_channels, n_regions)  ->  X: (n_regions, n_channels)
            X = harm.T

            # Center features (channels). This “concatenates all channels” in the sense that
            # every region’s feature vector is the ΔRMS across *all* selected channels.
            Xc = X - X.mean(axis=0, keepdims=True)

            # Optional (uncomment to z-score channels if you want equal weighting):
            # Xc = Xc / (Xc.std(axis=0, ddof=1, keepdims=True) + 1e-12)

            # PCA via SVD
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            scores = U * S  # rows = regions, cols = PCs
            pc1, pc2 = scores[:, 0], scores[:, 1]
            evr = (S**2) / (S**2).sum()  # explained variance ratio

            # Plot PC1 vs PC2
            plt.figure(figsize=(4,4))
            plt.scatter(pc1, pc2, s=6, alpha=0.6)
            plt.axhline(0, linestyle='--', linewidth=0.8)
            plt.axvline(0, linestyle='--', linewidth=0.8)
            plt.xlabel(f"PC1  ({evr[0]*100:.1f}% var)")
            plt.ylabel(f"PC2  ({evr[1]*100:.1f}% var)")
            plt.title("PCA of harm map (regions as points; features = channels)")
            plt.tight_layout()
            # plt.show()


            # ΔRMS histograms per channel (grid of small subplots)

            # Threshold lines for reference (adjust if you used different gates)
            REF_THR  = -5.0
            CHAN_THR = 10.0

            if harm.size == 0:
                print("harm_at_best is empty; nothing to plot.")
            else:
                n_sel, n_regions = harm.shape

                # Robust global x-range so all histograms share axes
                all_vals = harm.ravel()
                x_min, x_max = np.nanpercentile(all_vals, [1, 99])
                if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
                    x_min, x_max = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
                    if x_min == x_max:
                        x_min, x_max = x_min - 1.0, x_max + 1.0

                bins = 100

                # Grid size ~ square
                cols = int(np.ceil(np.sqrt(n_sel)))
                rows = int(np.ceil(n_sel / cols))

                fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 1.8), sharex=True, sharey=True)
                axes = np.atleast_2d(axes)

                for i in range(rows * cols):
                    ax = axes.flat[i]
                    if i < n_sel:
                        ch_id = int(sel_ch[i])
                        x = harm[i, :]  # ΔRMS samples for this channel across regions

                        ax.hist(x, bins=bins, range=(x_min, x_max), histtype='stepfilled', alpha=0.7)
                        ax.axvline(0.0, color='black', linestyle='--', linewidth=0.8)
                        ax.axvline(CHAN_THR, color='red', linestyle=':', linewidth=0.8)

                        if ch_id == ref_ch:
                            ax.axvline(REF_THR, color='magenta', linestyle='-.', linewidth=0.8)
                            ax.set_title(f"{ch_id} (ref)", fontsize=9)
                        else:
                            ax.set_title(f"{ch_id}", fontsize=9)

                        ax.tick_params(labelsize=8)
                    else:
                        ax.axis('off')

                fig.suptitle("ΔRMS per channel (harm at best samples)", y=1.02)
                plt.tight_layout()
                # plt.show()



            plt.figure(figsize=(20, 6))
            im = plt.imshow(harm, aspect='auto', cmap='coolwarm', vmin=-v, vmax=v, interpolation='nearest')
            plt.colorbar(im, label='ΔRMS')
            plt.yticks(np.arange(len(sel_ch)), sel_ch)
            plt.xlabel('Region index')
            plt.ylabel('Channel (selected)')
            plt.title('Harm map at best samples (compact)')
            plt.tight_layout()
            # plt.show()

            # 2) Recompute EI from the best indices
            best_indices = np.array([r["best_idx"] for r in results["regions"]], dtype=np.int64)

            C, T = final_ei_full.shape
            ref_ch = results["ref_channel"]
            ci     = results["center_index"]
            pre  = -ci
            post = T - ci - 1
            selected_channels = np.arange(C, dtype=int)  # full EI

            # raw_data is (total_samples, 512) — matches extract_snippets_fast_ram
            snips, valid_times = extract_snippets_fast_ram(
                raw_data=raw_data,
                spike_times=np.asarray([r["best_idx"] for r in results["regions"]], dtype=np.int64),
                window=(pre, post),
                selected_channels=selected_channels,
            )
            # snips: (C, T, N). Average over spikes → EI
            ei_from_best = snips.mean(axis=2).astype(np.float32)

            # 3) Overlay recomputed EI with original (both recentred so trough aligns at sample 40)
            ei_orig_centered  = recenter_ei_to_ref_trough(final_ei_full, center_index=results["center_index"])
            ei_best_centered  = recenter_ei_to_ref_trough(ei_from_best,    center_index=results["center_index"])

            fig = plt.figure(figsize=(20, 12))
            ax = fig.add_subplot(111)
            pew.plot_ei_waveforms(
                [ei_orig_centered, ei_best_centered],
                positions=ei_positions,
                ref_channel=ref_ch,
                scale=70.0,
                ax=ax,
                colors=['green', 'red'],
                alpha=[0.9, 0.9],
                linewidth=[0.6, 0.6],
                box_height=1.0,
                box_width=50.0,
                aspect=1.0,
            )
            ax.set_title('EI overlay: original (green) vs recomputed from best indices (red)')
            plt.tight_layout()
            # plt.show()


            wfs = snips[ref_ch, :, :]           # (T, N)
            mean_wf = wfs.mean(axis=1)          # (T,)

            plt.figure(figsize=(20, 12))
            plt.plot(wfs, color='black', linewidth=0.25, alpha=0.08)  # ALL traces
            plt.plot(mean_wf, color='red', linewidth=2.5, alpha=0.95)
            plt.plot(ei_orig_centered[ref_ch], color='green', linewidth=2.5, alpha=0.95)
            plt.axvline(ci, color='k', linestyle='--', linewidth=1, alpha=0.6)

            plt.title(f"Ref channel {ref_ch}: all waveforms (N={wfs.shape[1]}) + new mean (red) + original mean (green)", fontsize=14)
            plt.xlabel("Sample (aligned; trough at center_index)", fontsize=12)
            plt.ylabel("μV", fontsize=12)
            plt.tight_layout()
            # plt.show()




            # ----- EI difference assessment (orig vs best) -----
            _ = assess_ei_drift(
                    ei_orig_centered,
                    ei_best_centered,     # or final_ei_full if that's your current EI
                    rms_thr=5.0,
                    channel_ids=np.arange(ei_orig_centered.shape[0], dtype=int),
                    title_prefix=f"ch {int(ch)}"
                )


            # === Mix check: k-means per-channel on explained spikes (post-explain, using ei_best_centered) ===

            # 1) Channels by RMS on ei_best_centered
            rms = np.sqrt(np.mean(ei_best_centered**2, axis=1)).astype(np.float32)  # [C]
            channels = np.flatnonzero(rms > 10.0).astype(int)
            print(f"[mix-check] channels passing RMS>10: {channels.size} → {channels[:20]}{' ...' if channels.size>20 else ''}")

            if channels.size == 0:
                print("[mix-check] No channels pass RMS>10; skipping k-means diagnostics.")
            else:
                # 2) Subset snips and EI to those channels (keep global labels in channel_ids)
                snips_sub = snips[channels, :, :]            # [C_sel, L, N]
                E_sub     = ei_best_centered[channels, :]    # [C_sel, L]

                # 3) Run diagnostics (k=2 on full waveform), 4 panels per row
                diag = kmeans_split_diagnostics(
                    snips_sub,
                    E_sub,
                    channel_ids=channels,   # label with GLOBAL ch ids
                    rms_thr=0.0,            # we've already filtered by RMS>10
                    n_init=8,
                    max_iter=60,
                    ncols=4,
                    title_prefix="post-explain (ei_best_centered)"
                )

            # === Split & visualize based on first good channel (post-explain) ===
            res_split = None

            res_split = split_first_good_channel_and_visualize(
                    snips,                 # 512 x 121 x 661 (explained spikes, all channels)
                    ei_best_centered,      # 512 x 121 (mean over explained spikes), centered
                    ei_positions,
                    rms_thr=10.0,
                    dprime_thr=5.0,
                    min_per_cluster=10,
                    n_init=8, max_iter=60,
                    lag_radius=0
                )

            if res_split is not None:
                _ = plot_grouped_harm_maps_two_eis(
                        res_split['EI0'], res_split['EI1'], snips, res_split['idx0'], res_split['idx1'],
                        p2p_thr=30.0,      # same family as elsewhere
                        max_channels=80,
                        min_channels=10,
                        lag_radius=0,
                        weight_by_p2p=True,
                        weight_beta=0.7,
                        title_prefix=f"ch {int(ch)}"
                    )

                metrics = classify_two_cells_vs_ab_shard(
                    res_split['EI0'], res_split['EI1'], snips, res_split['idx0'], res_split['idx1'],
                    p2p_thr=30.0, max_channels=80, min_channels=10,
                    lag_radius=0, weight_by_p2p=True, weight_beta=0.7,
                    rms_thr_support=10.0,
                    asym_strong_z=2.0,  # tighten/loosen if needed
                    asym_pure_z=1.0
                )


            # Edits to make INSIDE the pasted code (minimal, mechanical):
            #   (A) Make sure the channel variable equals `ch`. If your code uses
            #       a name like channel_of_interest / ref_channel / peak_channel,
            #       force-assign them to `ch` at the very top of the pasted block:
            #           channel_of_interest = ch
            #           ref_channel = ch
            #           peak_channel = ch
            #       (It’s fine if some of these names aren’t used; harmless.)
            #
            #   (B) After you build the FIRST snippet cube for this channel
            #       (the line that assigns to `snips_cur = ... extract_snippets_fast_ram(...)`),
            #       add the two lines below to set counts and an early skip:
            #           n_init = int(snips_cur.shape[2])
            #           if n_init < MIN_PEAKS:
            #               return {"status":"skip_insufficient_peaks","ch":int(ch),
            #                       "n_init":n_init, "n_final":0}, []
            #
            #   (C) Near the end, after you decide the cohort and sanitize indices
            #       (right where you have `chosen_idx = ...` and after filtering bad indices),
            #       set:
            #           n_final = int(chosen_idx.size) if 'chosen_idx' in locals() else int(snips_cur.shape[2])
            #       If you have an early branch where you skip the split (cohort empty),
            #       set `n_final = 0` in that branch.
            #
            #   (D) DO NOT call plt.show(). Keep all plotting calls unchanged.
            #
            #   (E) You already switched to median_ei_adaptive; keep that.
            #
            # ------------------------------------------------------------------

        # --- strict main-channel anchor gate (post-proto EI, pre-continuous-scan) ---
        try:
            # choose the final proto EI if available
            proto_ei = final_ei_full if 'final_ei_full' in locals() else ei_full

            # peak-to-peak per channel to find main
            p2p = proto_ei.max(axis=1) - proto_ei.min(axis=1)
            main_ch = int(np.argmax(p2p))

            if main_ch != int(ch):
                print(
                    f"[main-anchor] ABORT TEMPLATE: main={main_ch} "
                    f"amp_main={p2p[main_ch]:.1f} | ch={int(ch)} amp_ch={p2p[int(ch)]:.1f}"
                )

                # small visible page so the PDF records the decision
                _page = plt.figure(figsize=(7, 3))
                _page.suptitle(f"Ch {int(ch)} · NOT MAIN ANCHOR", y=0.98)
                _page.text(
                    0.02, 0.75,
                    f"main={main_ch}  amp_main={p2p[main_ch]:.1f}  "
                    f"amp_ch={p2p[int(ch)]:.1f}",
                    family="monospace"
                )

                # stash a result to return after we add the LOG page
                _anchor_skip__ = True
                _anchor_result__ = {
                    "status": "not_main_anchor",
                    "ch": int(ch),
                    "n_init": int(n_init) if 'n_init' in locals() else None,
                    "n_final": int(n_final) if 'n_final' in locals() else None,
                    "main_ch": main_ch,
                    "amp_main": float(p2p[main_ch]),
                    "amp_ch": float(p2p[int(ch)]),
                }
            else:
                _anchor_skip__ = False

        except Exception as _e_anchor:
            print(f"[main-anchor] ERROR: {type(_e_anchor).__name__}: {_e_anchor}")
            _anchor_skip__ = False

        # Build a concise title for the log pages
        label_parts = [f"Ch {ch}"]
        if "n_init" in locals() and "n_final" in locals():
            try:
                label_parts.append(f"spikes {int(n_init)}→{int(n_final)}")
            except Exception:
                pass
        if "cohort" in locals() and cohort:
            try:
                label_parts.append(f"cohort={cohort}")
            except Exception:
                pass
        if "dprime" in locals() and dprime is not None:
            try:
                label_parts.append(f"d′={float(dprime):.2f}")
            except Exception:
                pass
        log_title = " · ".join(label_parts) + " · LOG"

        # Create log pages WHILE FigureCollector is active
        _ = logs_to_figures(_lc.text, title=log_title)

        # return early if we aborted on anchor mismatch (figures already captured)
        if '_anchor_skip__' in locals() and _anchor_skip__:
            return _anchor_result__, _fc.figures


    # Build a resilient result dict; only populate fields that exist
    result = {"status": "ok", "ch": int(ch)}
    if "n_init" in locals():   result["n_init"]  = int(n_init)
    if "n_final" in locals():  result["n_final"] = int(n_final)
    if "dprime" in locals():   result["dprime"]  = float(dprime)
    if "cohort" in locals():   result["cohort"]  = str(cohort)

    return result, _fc.figures

# --- helper: save collected figs to a single PDF and close them ---
def save_figs_to_pdf(figs, pdf_path):
    """Append all figures to one PDF file, then close them to free memory."""
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(pdf_path) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

def logs_to_figures(text, title=None, max_chars=110, max_lines=55, figsize=(11, 8.5), fontsize=9):
    """
    Convert a long log string into one or more Matplotlib figures (pages).
    Each page uses a monospace block of wrapped text.
    """
    if text is None or not str(text).strip():
        text = "(no output)"
    s = str(text).replace("\r\n", "\n").replace("\r", "\n")
    # Wrap and paginate
    lines = []
    for raw in s.split("\n"):
        wrapped = textwrap.wrap(
            raw, width=max_chars,
            replace_whitespace=False, drop_whitespace=False
        ) or [""]
        lines.extend(wrapped)

    figs = []
    for i in range(0, len(lines), max_lines):
        page = "\n".join(lines[i:i+max_lines])
        fig = plt.figure(figsize=figsize)
        if title:
            try:
                fig.suptitle(title, y=0.98)
            except Exception:
                pass
        fig.text(0.02, 0.96, page, va="top", ha="left", family="monospace", fontsize=fontsize)
        figs.append(fig)
    return figs


# --- run a subset of channels and write a single multipage PDF + CSV ---
def run_subset_channels(channels, pdf_path, params=None):
    if params is None:
        params = {}

    from matplotlib.backends.backend_pdf import PdfPages
    import csv, traceback

    summary_rows = []

    with PdfPages(pdf_path) as pdf:
        for ch in channels:
            figs = []
            try:
                result, figs = proto_to_kmeans_for_channel(ch=int(ch), params=params)

                # Build a compact label for pages that don't already have a suptitle
                title = f"Ch {ch}"
                if result.get("n_init") is not None and result.get("n_final") is not None:
                    title += f" · spikes {result['n_init']}→{result['n_final']}"
                if result.get("cohort"):
                    title += f" · cohort={result['cohort']}"
                if result.get("dprime") is not None:
                    try:
                        title += f" · d′={float(result['dprime']):.2f}"
                    except Exception:
                        pass

                # If no figures were produced (e.g., early skip), add a placeholder page
                if not figs:
                    fig = plt.figure(figsize=(7, 4))
                    fig.suptitle(title, y=0.98)
                    fig.text(0.5, 0.5, "No figures produced (early exit?)",
                             ha="center", va="center")
                    figs = [fig]

                # Save all figures for this channel, lightly annotating where helpful
                for fig in figs:
                    try:
                        # Only add a label if the figure doesn't already have one (e.g., LOG pages do)
                        if not getattr(fig, "_suptitle", None):
                            fig.suptitle(title, y=0.98)
                    except Exception:
                        pass
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

                    # --- hard cleanup per channel (prevents any accumulation) ---
                    try:
                        plt.close('all')
                    except Exception:
                        pass
                    figs = []                 # drop references safely
                    gc.collect()   # force GC cycle for large arrays
                    # Optional sanity check (enable briefly if debugging):
                    # assert len(plt.get_fignums()) == 0, f"Leak: open figs after ch {ch}: {plt.get_fignums()}"


                summary_rows.append({
                    "ch": int(ch),
                    "status": result.get("status", "ok"),
                    "n_init": result.get("n_init"),
                    "n_final": result.get("n_final"),
                    "dprime": result.get("dprime"),
                    "cohort": result.get("cohort"),
                    "n_pages": len(figs),
                })

            except Exception:
                # If something blows up, we still put an ERROR page into the PDF
                tb = traceback.format_exc()
                fig = plt.figure(figsize=(11, 8.5))
                fig.suptitle(f"Ch {ch} · ERROR", y=0.98)
                fig.text(0.01, 0.95, tb, va="top", ha="left",
                        family="monospace", fontsize=9)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                figs = [fig]  # define figs for downstream cleanup/accounting

                summary_rows.append({
                    "ch": int(ch),
                    "status": "error",
                    "n_init": None,
                    "n_final": None,
                    "dprime": None,
                    "cohort": None,
                    "n_pages": 1,
                })


    # Write a tiny CSV summary next to the PDF
    csv_path = pdf_path.rsplit(".", 1)[0] + "_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ch", "status", "n_init", "n_final", "dprime", "cohort", "n_pages"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved PDF: {pdf_path}")
    print(f"Saved CSV: {csv_path}")


# %%
C = raw_data.shape[1]
chunk = 64  # 32–128 are all reasonable; 64 is a good start
params_all = {"min_peaks": 30}

for start in range(0, C, chunk):
    end = min(start + chunk, C)
    chs = list(range(start, end))
    out_pdf = f"/Volumes/Lab/Users/alexth/axolotl/joint_main/proto_ch{start:03d}-{end-1:03d}.pdf"
    run_subset_channels(chs, out_pdf, params=params_all)


# %%
# Example A — manual list (edit these)
channels = [29, 133,142,395]

# # Example B — random 10 channels with a fixed seed (reproducible)
# rng = np.random.default_rng(123)
# channels = sorted(rng.choice(raw_data.shape[1], size=10, replace=False).tolist())

# # Example C — strided sample (every 32nd channel)
# channels = list(range(0, raw_data.shape[1], 32))  # adjust stride as you like

# Run the subset
params_test = {"min_peaks": 30}  # or your usual knobs
out_pdf = "/Volumes/Lab/Users/alexth/axolotl/joint/proto_subset_test.pdf"  # change path if you prefer
run_subset_channels(channels, out_pdf, params=params_test)


# %%
# --- one-channel smoke test (saves a PDF for just this channel) ---
ch_test = 29  # ← pick a busy channel
params_test = {"min_peaks": 30}  # keep/adjust your usual knobs here

res, figs = proto_to_kmeans_for_channel(ch=ch_test, params=params_test)
print(res)
print(f"Captured {len(figs)} figure(s)")

# Optionally label pages with a suptitle before saving
try:
    n_init = res.get("n_init", None)
    n_final = res.get("n_final", None)
    dprime = res.get("dprime", None)
    cohort = res.get("cohort", None)
    label = f"Ch {ch_test}"
    if n_init is not None and n_final is not None:
        label += f" · spikes {n_init}→{n_final}"
    if cohort:
        label += f" · cohort={cohort}"
    if dprime is not None:
        label += f" · d′={dprime:.2f}"
    for fig in figs:
        try:
            fig.suptitle(label, y=0.99)
        except Exception:
            pass
except Exception:
    pass

# Save to the current working directory; use an absolute path if you prefer
pdf_path = "/Volumes/Lab/Users/alexth/axolotl/joint/proto_to_kmeans_ch29.pdf"
save_figs_to_pdf(figs, pdf_path)
print(f"Saved: {pdf_path}")



# %% [markdown]
# ## Single channel processing

# %% [markdown]
# ### Find proto template

# %%


# ---------- Config ----------
ch          = 3            # channel index
thr         = -200.0       # negative threshold (same units as raw)
max_events  = 400          # first N events in time
min_gap     = 40           # enforce ≥40-sample separation
pre, post   = 20, 40       # window relative to the negative peak (inclusive)
to_select   = 100           # events to select for template

# ---- Config for harm map ----
p2p_thr      = 30.0     # channel selection threshold by p2p
max_channels = 80       # hard cap
min_channels = 10       # ensure at least this many
lag_radius   = 3        # ± samples for lag scan
win          = (-40, 80)

# =========================
# Config for harm map
# =========================
MEAN_THR  = -2.0    # mean ΔRMS across selected channels must be <= this
CHAN_THR  = 15.0    # any channel ΔRMS > this -> reject spike
REF_THR   = -5.0    # ref-channel ΔRMS must be < this
LAG_RADIUS = 0      # no micro-shift; keep it literal
MAX_ITERS  = 10     # safety stop



# ---------- Source trace  ----------

trace = raw_data[:2_000_000, ch].astype(np.float32).ravel()

# trace = raw_data[:,ch].astype(np.float32).T  # raw: shape (n_channels, n_samples)

# # ---------- Find negative-peak events crossing threshold ----------
x = trace
# local minima below threshold
cand = np.where((x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:]) & (x[1:-1] <= thr))[0] + 1
# enforce min_gap and window boundaries, keep first max_events in time
picked = []
last = -10**9
for i in cand:
    if i - last >= min_gap and (i - pre) >= 0 and (i + post) < x.size:
        picked.append(i)
        last = i
        if len(picked) >= max_events:
            break
picked = np.array(picked, dtype=int)


# # ---------- Picked events with mask ----------
# if 'mask_bank' in globals():
#     # Mask segments belonging to any existing templates that should mute this channel
#     trace_masked, mask_info = joint_utils.mask_trace_with_template_bank(
#         trace, target_ch=int(ch), bank=mask_bank,
#         amp_thr_default=25.0, center_index_default=40,
#         thr=thr,        # your detection threshold (e.g., -200.0); ensures masked samples sit above it
#         fill_value=None # leave None → auto-select 0.0 or thr+1; or set explicitly
#     )

#     print(f"[mask] ch={ch} segments={mask_info['segments']} "
#         f"masked_samples={mask_info['masked_samples']} "
#         f"({100*mask_info['masked_fraction']:.3f}%)")
# else:
#     trace_masked = trace.copy()
#     print("No templates yet, skip masking")
# det   = joint_utils.detect_negative_peaks_on_channel(trace_masked, thr=thr, min_gap=min_gap, pre=pre, post=post, max_events=max_events)
# picked = det['picked']

# # ---------- Picked events manually inserted ----------
# picked = np.sort(picked_sel)


if picked.size == 0:
    print("No events found—check threshold/sign/channel.")
else:
    # ---------- Extract waveforms ----------
    wf = np.stack([x[i - pre : i + post + 1] for i in picked], axis=0)  # shape (N, pre+post+1)
    t = np.arange(-pre, post + 1)

    # ---------- Select top-N by amplitude on ref channel (prescreen) ----------
    # Use negative-peak magnitude within the local window on channel `ch`
    amp_all = -wf.min(axis=1)  # shape (N,)

    k = min(to_select, wf.shape[0])
    sel_idx = np.argsort(-amp_all)[:k]   # indices of top-N by amplitude

    wf_sel  = wf[sel_idx]
    med_sel = np.median(wf_sel, axis=0)


    # ---------- Plots ----------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)

    ax = axes[0]
    ax.plot(t, wf.T, linewidth=0.5, alpha=0.4)
    ax.set_title(f"All waveforms (N={wf.shape[0]})  | ch {ch}")
    ax.set_xlabel("samples from peak"); ax.set_ylabel("ADC")

    ax = axes[1]
    ax.plot(t, wf_sel.T, linewidth=0.5, alpha=0.55, label=f"selected {k}")
    ax.plot(t, med_sel, linewidth=2.5, label="median(selected)")
    ax.set_title("Selected top-N by amplitude + median")
    ax.set_xlabel("samples from peak")

    plt.tight_layout()
    plt.show()

    # ===== EI from selected spikes (full array) =====

    # Use your selected spike times on ch
    times_sel = picked[sel_idx].astype(np.int64)

    # EI window (use your standard full EI window)
    window_ei = (-40, 80)   # center = -window_ei[0]
    C_full    = raw_data.shape[1]
    all_ch    = np.arange(C_full, dtype=int)

    # Extract snippets on ALL channels at those times
    snips_ei, valid_times = extract_snippets_fast_ram(raw_data, times_sel, window_ei, all_ch)
    if snips_ei.shape[2] == 0:
        raise RuntimeError("No valid snippets for EI (check edges / times).")

    # EI = mean across selected events
    ei_sel_full = median_ei_adaptive(snips_ei)   # [C_full, L_ei]


    # Identify main channel by the most-negative trough on the current EI
    try:
        ch0, t_neg0 = main_channel_and_neg_peak(ei_sel_full)
    except Exception:
        # fallback: same logic inline
        mins = ei_sel_full.min(axis=1)
        ch0  = int(np.argmin(mins))
        t_neg0 = int(np.argmin(ei_sel_full[ch0]))

    print(f"Main channel: {ch0}")
    amps = (-snips_ei[ch0, t_neg0, :]).astype(np.float32)   # positive trough magnitude per spike
    if amps.size:
        TOPK = 25
        FRAC = 0.75
        k = min(TOPK, amps.size)
        mu_top = float(np.mean(np.sort(amps)[-k:]))           # mean of top-k trough magnitudes
        thr_amp = FRAC * mu_top
        keep_amp = amps >= thr_amp
        n_drop = int((~keep_amp).sum())
        if n_drop > 0:
            print(f"amplitude gate on main ch {ch0} @t={t_neg0}: "
                    f"drop {n_drop}/{amps.size} (μ_top{k}={mu_top:.1f} → thr={thr_amp:.1f})")
            # print(f"Dropped: {times_sel[~keep_amp]}")
            snips_ei = snips_ei[:, :, keep_amp]
            if snips_ei.shape[2] == 0:
                print("amplitude gate removed all spikes; stopping.")
            else: # Rebuild EI on the reduced set to keep harm-map inputs consistent
                ei_sel_full = median_ei_adaptive(snips_ei)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3), sharey=True)

    ax = axes[0]
    ax.plot(snips_ei[ch0], linewidth=0.5, alpha=0.4)
    ax.set_title(f"All waveforms on main ch {ch0}")
    ax.set_xlabel("samples from peak"); ax.set_ylabel("ADC")

    ax = axes[1]
    ax.plot(snips_ei[ch], linewidth=0.5, alpha=0.4)
    ax.set_title(f"All waveforms on ref ch {ch}")
    ax.set_xlabel("samples from peak"); ax.set_ylabel("ADC")

    ax = axes[2]
    extra_ch_to_plot = 330
    ax.plot(snips_ei[extra_ch_to_plot], linewidth=0.5, alpha=0.4)
    ax.set_title(f"All waveforms on extra ch {extra_ch_to_plot}")
    ax.set_xlabel("samples from peak"); ax.set_ylabel("ADC")

    plt.tight_layout()
    plt.show()


    # Plot the EI (overlay-ready if you add more later)
    fig, ax = plt.subplots(1, 1, figsize=(10,12))
    plot_ei_waveforms.plot_ei_waveforms(
        ei_sel_full, ei_positions, ref_channel=int(ch), ax=ax,
        colors='C2', scale=70.0, box_height=1.0, box_width=50.0
    )
    ax.set_title(f"EI from selected spikes on ch {int(ch)} (k={len(sel_idx)})")
    plt.tight_layout(); plt.show()




    # 1) Preselect channels from EI (same rule the harm-map uses)
    chans_pre, ptp = select_template_channels(
        ei_sel_full, p2p_thr=p2p_thr, max_n=max_channels, min_n=min_channels, force_include_main=True
    )

    # Make sure the main (most-negative) channel is present
    ch_main, _ = main_channel_and_neg_peak(ei_sel_full)
    if ch_main not in chans_pre:
        # replace weakest with main, re-sort by p2p desc, keep unique
        pool = np.concatenate([chans_pre[:-1], [ch_main]])
        chans_pre = np.array(sorted(set(pool), key=lambda c: ptp[c], reverse=True), dtype=int)

    print(f"channels used: {len(chans_pre)} (main={int(ch_main)})")

    # 3) Harm-map on the reduced channel set
    #    Lock selection inside harm-map to use *exactly* these channels by setting p2p_thr very low and min=max=len(chans_pre).
    res = compute_harm_map_noamp(
        ei                   = ei_sel_full[chans_pre],   # [K, L]
        snips                = snips_ei[chans_pre],            # [K, L, N]
        p2p_thr              = -1e9,                 # include all provided chans
        max_channels         = len(chans_pre),
        min_channels         = len(chans_pre),
        lag_radius           = lag_radius,
        weight_by_p2p        = True,
        weight_beta          = 0.7,
        force_include_main   = True,
    )

    # 4) Plot
    plot_harm_heatmap(res, field="harm_matrix",
                    title=f"Harm map ΔRMS | ch {int(ch)} | N={valid_times.size}")


    # Work on a copy of your current snippets
    snips_cur = snips_ei.copy()
    C_avail, L, N0 = snips_cur.shape


    # === Split & visualize based on first good channel (post-explain) ===
    res_split = None
    res_split = split_first_good_channel_and_visualize(
            snips_cur,                 # 512 x 121 x 661 (explained spikes, all channels)
            ei_sel_full,      # 512 x 121 (mean over explained spikes), centered
            ei_positions,
            rms_thr=10.0,
            dprime_thr=5.0,
            min_per_cluster=10,
            n_init=8, max_iter=60,
            lag_radius=0
        )
    
    if res_split is not None:
        metrics = classify_two_cells_vs_ab_shard(
            res_split['EI0'], res_split['EI1'], snips_cur, res_split['idx0'], res_split['idx1'],
            p2p_thr=30.0, max_channels=80, min_channels=10,
            lag_radius=0, weight_by_p2p=True, weight_beta=0.7,
            rms_thr_support=10.0,
            asym_strong_z=2.0,  # tighten/loosen if needed
            asym_pure_z=1.0
        )

        if metrics["label"] == "two cells":
            if len(res_split['idx0'])>=len(res_split['idx1']):
                snips_cur = snips_cur[:,:,res_split['idx0']]
            else:
                snips_cur = snips_cur[:,:,res_split['idx1']]

    C_avail, L, N0 = snips_cur.shape

    print(f"[start] snippets: C={C_avail}, L={L}, N={N0}")

    final_res = None
    final_ei_full = None
    bimodal_plotted = False
    bimodal_payload = None   # store first detected bimodal split to finalize at the end




    for it in range(1, MAX_ITERS + 1):
        # --- EI from current spikes (no lags, straight mean) ---
        ei_full = median_ei_adaptive(snips_cur)     # [C_avail, L]
     

        # --- Harm map with fresh channel selection each round ---
        res_it = compute_harm_map_noamp(
            ei=ei_full, snips=snips_cur,
            p2p_thr=50.0,               # your usual selector; adjust if needed
            max_channels=C_avail,       # allow selector to choose from what's available
            min_channels=10,
            lag_radius=LAG_RADIUS,
            weight_by_p2p=True, weight_beta=0.7,
            force_include_main=True
        )
        HM    = np.asarray(res_it["harm_matrix"])                 # [K_sel, N_cur]
        sel   = np.asarray(res_it["selected_channels"], int)      # [K_sel]
        K_sel, N_cur = HM.shape

        # --- Build the reject mask from your three rules ---
        mean_d  = HM.mean(axis=0)          # [N_cur]
        max_d   = HM.max(axis=0)           # [N_cur]

        # ref channel row (if not selected, we can't check REF_THR; warn & skip that term)
        ref_matches = np.where(sel == int(ch))[0]
        if ref_matches.size == 0:
            print(f"[iter {it}] WARNING: ref ch {int(ch)} not in selected_channels this round; skipping REF_THR check.")
            ref_ok = np.ones(N_cur, dtype=bool)
        else:
            ref_row = int(ref_matches[0])
            ref_d   = HM[ref_row]          # [N_cur]
            ref_ok  = (ref_d < REF_THR)

        keep = (mean_d <= MEAN_THR) & (max_d <= CHAN_THR) & ref_ok
        n_bad = int((~keep).sum())
        print(f"[iter {it}] N={N_cur} | reject={n_bad} "
            f"(mean>{MEAN_THR}: {int((mean_d>MEAN_THR).sum())}, "
            f"any>{CHAN_THR}: {int((max_d>CHAN_THR).sum())}, "
            f"ref>={REF_THR}: {int((~ref_ok).sum())})")

        # --- Debug: which channels cause the max-Δ failures? ---
        if n_bad > 0:
            from collections import Counter
            sel_global = np.asarray(res_it["selected_channels"], dtype=int)   # rows → global channel ids
            bad_spikes = np.where(max_d > CHAN_THR)[0]                        # failing spike indices
            # argmax row for each failing spike (row is within selected_channels)
            row_of_max = np.argmax(HM[:, bad_spikes], axis=0)
            ch_of_max  = sel_global[row_of_max]                               # map to global channel ids
            val_of_max = HM[row_of_max, bad_spikes]

            # summary: top offending channels by count
            counts = Counter(ch_of_max.tolist()).most_common(15)
            print(f"[iter {it}] channels hitting max Δ>{CHAN_THR} (top 15):")
            for ch_id, cnt in counts:
                print(f"  ch {int(ch_id):4d}  →  {cnt} spikes")

            # first few concrete examples (spike index, channel, Δ value)
            head = min(10, bad_spikes.size)
            for s_idx, ch_id, val in zip(bad_spikes[:head], ch_of_max[:head], val_of_max[:head]):
                print(f"    spike {int(s_idx):5d}: ch {int(ch_id):4d}, Δ={float(val):.2f}")


        if n_bad == 0:
            # default: no split, use current survivors as final
            final_res = res_it
            final_ei_full = ei_full

            # One more bimodality check if we never looked (e.g., no rejections happened)
            if bimodal_payload is None:
                tmp = check_bimodality_and_plot(
                    snips_cur, res_it, ei_positions, ref_ch=int(ch),
                    dprime_thr=5.0, min_per_cluster=5
                )
                if tmp and tmp.get("hit", False):
                    bimodal_payload = tmp

            if bimodal_payload and bimodal_payload.get("hit", False):
                # expose cohort EIs for inspection
                ei_lo_bimodal = bimodal_payload["ei_lo"]
                ei_hi_bimodal = bimodal_payload["ei_hi"]
                idx_lo = bimodal_payload["idx_lo"]
                idx_hi = bimodal_payload["idx_hi"]

                # choose cohort by larger N
                # amp_lo = float(-ei_lo_bimodal[int(ch)].min())
                # amp_hi = float(-ei_hi_bimodal[int(ch)].min())
                pick_hi = (len(idx_hi) >= len(idx_lo))
                chosen_idx = idx_hi if pick_hi else idx_lo

                # --- guard against stale/out-of-range indices ---
                chosen_idx = np.asarray(chosen_idx, dtype=np.int64).ravel()
                Ncur = snips_cur.shape[2]
                bad = (chosen_idx < 0) | (chosen_idx >= Ncur)
                if bad.any():
                    print(f"[bimodality] WARNING: {bad.sum()} invalid indices (min={chosen_idx.min()}, "
                        f"max={chosen_idx.max()}, N={Ncur}). Filtering.")
                    chosen_idx = chosen_idx[~bad]

                if chosen_idx.size == 0:
                    print("[bimodality] cohort empty after filtering; skipping split.")
                    final_res = res_it
                    final_ei_full = ei_full
                else:
                    # restrict survivors to chosen cohort and recompute EI + harm-map
                    snips_fin = snips_cur[:, :, chosen_idx]
                    final_ei_full = median_ei_adaptive(snips_fin)


                final_res = compute_harm_map_noamp(
                    ei=final_ei_full, snips=snips_fin,
                    p2p_thr=50.0, max_channels=snips_fin.shape[0], min_channels=10,
                    lag_radius=LAG_RADIUS, weight_by_p2p=True, weight_beta=0.7,
                    force_include_main=True
                )

                # plots: new harm-map and the final EI
                plot_harm_heatmap(
                    final_res, field="harm_matrix",
                    title=f"Final harm map (post-bimodal split; picked {'high' if pick_hi else 'low'} amp cohort)"
                )
                try:
                    fig, ax = plt.subplots(figsize=(20, 12))
                    pew.plot_ei_waveforms(
                        final_ei_full, ei_positions,
                        ref_channel=int(ch), scale=70.0, box_height=1.0, box_width=50.0, ax=ax
                    )
                    ax.set_title(f"Final EI | ch {int(ch)} | cohort={'high' if pick_hi else 'low'}")
                    plt.show()
                except Exception as e:
                    print(f"[final EI] plotting skipped: {e}")

            print("[done] All events satisfy thresholds.")
            break



        # Kick failing spikes, recompute everything fresh next round
        snips_cur = snips_cur[:, :, keep]
        if snips_cur.shape[2] == 0:
            print(f"[iter {it}] all spikes rejected; stopping.")
            final_res = res_it
            final_ei_full = ei_full
            break

        print(f"[iter {it}] kept {snips_cur.shape[2]}/{N_cur} after harm-map pruning")

        # Bimodality check on the post-kick set (capture FIRST hit only)
        if bimodal_payload is None:
            # Recompute a quick harm-map on the trimmed set so channel selection is accurate
            res_tmp = compute_harm_map_noamp(
                ei=median_ei_adaptive(snips_cur),
                snips=snips_cur,
                p2p_thr=50.0, max_channels=snips_cur.shape[0], min_channels=10,
                lag_radius=LAG_RADIUS, weight_by_p2p=True, weight_beta=0.7,
                force_include_main=True
            )
            tmp = check_bimodality_and_plot(
                snips_cur, res_tmp, ei_positions, ref_ch=int(ch),
                dprime_thr=5.0, min_per_cluster=5
            )
            if tmp and tmp.get("hit", False):
                bimodal_payload = tmp




    else:
        # fell out by MAX_ITERS
        final_res = res_it
        final_ei_full = ei_full
        print("[stop] Reached MAX_ITERS; some violations may remain.")

    # =========================
    # Plots: final harm map & EI
    # =========================
    if final_res is not None:
        plot_harm_heatmap(final_res, field="harm_matrix", sort_by_ptp=False,
                        title=f"Final harm map after pruning "
                                f"(K={len(final_res['selected_channels'])}, N={final_res['harm_matrix'].shape[1]})")

    if final_ei_full is not None:
        # If snips_cur covered the full array, this is full; otherwise it covers available channels only.
        C_full = ei_positions.shape[0]
        L_ei   = final_ei_full.shape[1]
        ei_plot = (final_ei_full if final_ei_full.shape[0] == C_full
                else np.pad(final_ei_full, ((0, C_full - final_ei_full.shape[0]), (0, 0)), mode='constant'))

        fig, ax = plt.subplots(1, 1, figsize=(10,12))
        pew.plot_ei_waveforms(ei_plot, ei_positions, ref_channel=int(ch), ax=ax,
                            colors='C2', scale=70.0, box_height=1.0, box_width=50.0)
        ax.set_title(f"Final EI after pruning (events kept = {final_res['harm_matrix'].shape[1] if final_res else 'n/a'})")
        plt.tight_layout(); plt.show()

# try:
proto_ei = final_ei_full if 'final_ei_full' in locals() else ei_full
# Peak-to-peak per channel
p2p = proto_ei.max(axis=1) - proto_ei.min(axis=1)
main_ch = int(np.argmax(p2p))
if main_ch != ch:
    print(f"ABORT TEMPLATE: MAIN CHANNEL {main_ch} with amp {p2p[main_ch]:0.1f}, current CH {ch} with amp {p2p[ch]:0.1f}")



# %% [markdown]
# ### Scan continuous data with proto template

# %%
results = scan_continuous_harm_regions(
    raw_data=raw_data.T,
    final_ei_full=final_ei_full,
    start_sample=0,
    stop_sample=2_000_000,   # exclusive
    mean_thr=-2.0,
    chan_thr=10.0,
    ref_thr=-5.0,
    center_index=40,
    region_max_len=20,
)

harm = results["harm_at_best"]
n_sel, n_regions = harm.shape
print(f"Spikes: {n_regions}, Channels: {n_sel}")


# %% [markdown]
# #### Plot results of continuous scan, assess EI drift

# %%

# 1) Plot compact harm map (ΔRMS per selected channel at each region’s best sample)
harm = results["harm_at_best"]             # (n_sel, n_regions)
sel_ch = results["selected_channels"]      # (n_sel,)
ref_ch = int(results["ref_channel"])      # integer channel id
v = np.nanpercentile(np.abs(harm), 95) if harm.size else 1.0


# harm: (n_channels, n_regions)  ->  X: (n_regions, n_channels)
X = harm.T

# Center features (channels). This “concatenates all channels” in the sense that
# every region’s feature vector is the ΔRMS across *all* selected channels.
Xc = X - X.mean(axis=0, keepdims=True)

# Optional (uncomment to z-score channels if you want equal weighting):
# Xc = Xc / (Xc.std(axis=0, ddof=1, keepdims=True) + 1e-12)

# PCA via SVD
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
scores = U * S  # rows = regions, cols = PCs
pc1, pc2 = scores[:, 0], scores[:, 1]
evr = (S**2) / (S**2).sum()  # explained variance ratio

# Plot PC1 vs PC2
plt.figure(figsize=(4,4))
plt.scatter(pc1, pc2, s=6, alpha=0.6)
plt.axhline(0, linestyle='--', linewidth=0.8)
plt.axvline(0, linestyle='--', linewidth=0.8)
plt.xlabel(f"PC1  ({evr[0]*100:.1f}% var)")
plt.ylabel(f"PC2  ({evr[1]*100:.1f}% var)")
plt.title("PCA of harm map (regions as points; features = channels)")
plt.tight_layout()
plt.show()


# ΔRMS histograms per channel (grid of small subplots)

# Threshold lines for reference (adjust if you used different gates)
REF_THR  = -5.0
CHAN_THR = 10.0

if harm.size == 0:
    print("harm_at_best is empty; nothing to plot.")
else:
    n_sel, n_regions = harm.shape

    # Robust global x-range so all histograms share axes
    all_vals = harm.ravel()
    x_min, x_max = np.nanpercentile(all_vals, [1, 99])
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        x_min, x_max = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        if x_min == x_max:
            x_min, x_max = x_min - 1.0, x_max + 1.0

    bins = 100

    # Grid size ~ square
    cols = int(np.ceil(np.sqrt(n_sel)))
    rows = int(np.ceil(n_sel / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 1.8), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for i in range(rows * cols):
        ax = axes.flat[i]
        if i < n_sel:
            ch_id = int(sel_ch[i])
            x = harm[i, :]  # ΔRMS samples for this channel across regions

            ax.hist(x, bins=bins, range=(x_min, x_max), histtype='stepfilled', alpha=0.7)
            ax.axvline(0.0, color='black', linestyle='--', linewidth=0.8)
            ax.axvline(CHAN_THR, color='red', linestyle=':', linewidth=0.8)

            if ch_id == ref_ch:
                ax.axvline(REF_THR, color='magenta', linestyle='-.', linewidth=0.8)
                ax.set_title(f"{ch_id} (ref)", fontsize=9)
            else:
                ax.set_title(f"{ch_id}", fontsize=9)

            ax.tick_params(labelsize=8)
        else:
            ax.axis('off')

    fig.suptitle("ΔRMS per channel (harm at best samples)", y=1.02)
    plt.tight_layout()
    plt.show()



plt.figure(figsize=(20, 6))
im = plt.imshow(harm, aspect='auto', cmap='coolwarm', vmin=-v, vmax=v, interpolation='nearest')
plt.colorbar(im, label='ΔRMS')
plt.yticks(np.arange(len(sel_ch)), sel_ch)
plt.xlabel('Region index')
plt.ylabel('Channel (selected)')
plt.title('Harm map at best samples (compact)')
plt.tight_layout()
plt.show()

# 2) Recompute EI from the best indices
best_indices = np.array([r["best_idx"] for r in results["regions"]], dtype=np.int64)

C, T = final_ei_full.shape
ref_ch = results["ref_channel"]
ci     = results["center_index"]
pre  = -ci
post = T - ci - 1
selected_channels = np.arange(C, dtype=int)  # full EI

spike_times=np.asarray([r["best_idx"] for r in results["regions"]], dtype=np.int64)

# raw_data is (total_samples, 512) — matches extract_snippets_fast_ram
snips, valid_times = extract_snippets_fast_ram(
    raw_data=raw_data,
    spike_times=spike_times,
    window=(pre, post),
    selected_channels=selected_channels,
)
# snips: (C, T, N). Average over spikes → EI
ei_from_best = snips.mean(axis=2).astype(np.float32)

# 3) Overlay recomputed EI with original (both recentred so trough aligns at sample 40)
ei_orig_centered  = recenter_ei_to_ref_trough(final_ei_full, center_index=results["center_index"])
ei_best_centered  = recenter_ei_to_ref_trough(ei_from_best,    center_index=results["center_index"])

fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(111)
pew.plot_ei_waveforms(
    [ei_orig_centered, ei_best_centered],
    positions=ei_positions,
    ref_channel=ref_ch,
    scale=70.0,
    ax=ax,
    colors=['green', 'red'],
    alpha=[0.9, 0.9],
    linewidth=[0.6, 0.6],
    box_height=1.0,
    box_width=50.0,
    aspect=1.0,
)
ax.set_title('EI overlay: original (green) vs recomputed from best indices (red)')
plt.tight_layout()
plt.show()


wfs = snips[ref_ch, :, :]           # (T, N)
mean_wf = wfs.mean(axis=1)          # (T,)

plt.figure(figsize=(20, 12))
plt.plot(wfs, color='black', linewidth=0.25, alpha=0.08)  # ALL traces
plt.plot(mean_wf, color='red', linewidth=2.5, alpha=0.95)
plt.plot(ei_orig_centered[ref_ch], color='green', linewidth=2.5, alpha=0.95)
plt.axvline(ci, color='k', linestyle='--', linewidth=1, alpha=0.6)

plt.title(f"Ref channel {ref_ch}: all waveforms (N={wfs.shape[1]}) + new mean (red) + original mean (green)", fontsize=14)
plt.xlabel("Sample (aligned; trough at center_index)", fontsize=12)
plt.ylabel("μV", fontsize=12)
plt.tight_layout()
plt.show()




# ----- EI difference assessment (orig vs best) -----
_ = assess_ei_drift(
        ei_orig_centered,
        ei_best_centered,     # or final_ei_full if that's your current EI
        rms_thr=5.0,
        channel_ids=np.arange(ei_orig_centered.shape[0], dtype=int),
        title_prefix=f"ch {int(ch)}"
    )


# %% [markdown]
# ### Check if template explained more than one cell (k-means)

# %%

# === Mix check: k-means per-channel on explained spikes (post-explain, using ei_best_centered) ===

# 1) Channels by RMS on ei_best_centered
rms = np.sqrt(np.mean(ei_best_centered**2, axis=1)).astype(np.float32)  # [C]
channels = np.flatnonzero(rms > 10.0).astype(int)
print(f"[mix-check] channels passing RMS>10: {channels.size} → {channels[:20]}{' ...' if channels.size>20 else ''}")

if channels.size == 0:
    print("[mix-check] No channels pass RMS>10; skipping k-means diagnostics.")
else:
    # 2) Subset snips and EI to those channels (keep global labels in channel_ids)
    snips_sub = snips[channels, :, :]            # [C_sel, L, N]
    E_sub     = ei_best_centered[channels, :]    # [C_sel, L]

    # 3) Run diagnostics (k=2 on full waveform), 4 panels per row
    diag = kmeans_split_diagnostics(
        snips_sub,
        E_sub,
        channel_ids=channels,   # label with GLOBAL ch ids
        rms_thr=0.0,            # we've already filtered by RMS>10
        n_init=8,
        max_iter=60,
        ncols=4,
        title_prefix="post-explain (ei_best_centered)"
    )

# === Split & visualize based on first good channel (post-explain) ===
res_split = None

res_split = split_first_good_channel_and_visualize(
        snips,                 # 512 x 121 x 661 (explained spikes, all channels)
        ei_best_centered,      # 512 x 121 (mean over explained spikes), centered
        ei_positions,
        rms_thr=10.0,
        dprime_thr=5.0,
        min_per_cluster=10,
        n_init=8, max_iter=60,
        lag_radius=0
    )

if res_split is not None:
    _ = plot_grouped_harm_maps_two_eis(
            res_split['EI0'], res_split['EI1'], snips, res_split['idx0'], res_split['idx1'],
            p2p_thr=30.0,      # same family as elsewhere
            max_channels=80,
            min_channels=10,
            lag_radius=0,
            weight_by_p2p=True,
            weight_beta=0.7,
            title_prefix=f"ch {int(ch)}"
        )

    metrics = classify_two_cells_vs_ab_shard(
        res_split['EI0'], res_split['EI1'], snips, res_split['idx0'], res_split['idx1'],
        p2p_thr=30.0, max_channels=80, min_channels=10,
        lag_radius=0, weight_by_p2p=True, weight_beta=0.7,
        rms_thr_support=10.0,
        asym_strong_z=2.0,  # tighten/loosen if needed
        asym_pure_z=1.0
    )

    fig, ax = plt.subplots(figsize=(15, 12))
    pew.plot_ei_waveforms(
        res_split['EI0'], ei_positions,
        ref_channel=int(ch), colors='black', scale=70.0, box_height=1.0, box_width=50.0, ax=ax
    )
    ax.set_title(f"Final EI0")
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 12))
    pew.plot_ei_waveforms(
        res_split['EI1'], ei_positions,
        ref_channel=int(ch), colors='black', scale=70.0, box_height=1.0, box_width=50.0, ax=ax
    )
    ax.set_title(f"Final EI1")
    plt.show()

    # # ----- EI difference assessment (orig vs best) -----
    # _ = assess_ei_drift(
    #         res_split['EI0'],
    #         res_split['EI1'],     # or final_ei_full if that's your current EI
    #         rms_thr=5.0,
    #         channel_ids=np.arange(ei_orig_centered.shape[0], dtype=int),
    #         title_prefix=f"EI0, EI1, ch {int(ch)}"
    #     )

    # # ----- EI difference assessment (orig vs best) -----
    # _ = assess_ei_drift(
    #         ei_orig_centered,
    #         res_split['EI0'],     # or final_ei_full if that's your current EI
    #         rms_thr=5.0,
    #         channel_ids=np.arange(ei_orig_centered.shape[0], dtype=int),
    #         title_prefix=f"Best, EI0, ch {int(ch)}"
    #     )

    # # ----- EI difference assessment (orig vs best) -----
    # _ = assess_ei_drift(
    #         ei_orig_centered,
    #         res_split['EI1'],     # or final_ei_full if that's your current EI
    #         rms_thr=5.0,
    #         channel_ids=np.arange(ei_orig_centered.shape[0], dtype=int),
    #         title_prefix=f"Best, EI1, ch {int(ch)}"
    #     )

# %% [markdown]
# ## Find more units

# %% [markdown]
# ### loop to look for units

# %%
del mask_bank

# %%
print(len(mask_bank))

# %%
# ========== Config ==========
ch          = 0
thr         = -75.0
max_events  = 400
min_gap     = 40
pre, post   = 20, 40
to_select   = 100
max_length  = 10_000_000

# Harm-map channel selection defaults
p2p_thr      = 30.0
max_channels = 80
min_channels = 10
lag_radius   = 3
window_ei    = (-40, 80)

# Compliance thresholds
MEAN_THR   = -1.0
CHAN_THR   = 15.0
REF_THR    = -3.0
MAX_ITERS  = 10

# ---- shared harm cfg (override channels later by setting 'channels_override')
harm_cfg = dict(
    p2p_thr=p2p_thr, max_channels=max_channels, min_channels=min_channels,
    lag_radius=lag_radius, weight_by_p2p=True, weight_beta=0.7, force_include_main=True,
    channels_override=None
)


for ch in range(0,512):

    flag = 1
    print(f"Current channel: {ch}")

    while flag:


        print(f"\n\n%%%%%%%% [CURRENTLY PROCESSING CHANNEL {ch}] %%%%%%%%")

        # ========== 1) Detection on ref channel ==========
        trace = raw_data[:max_length, ch].astype(np.float32).ravel()

        if 'mask_bank' in globals():
            # Mask segments belonging to any existing templates that should mute this channel
            trace_masked, mask_info = joint_utils.mask_trace_with_template_bank(
                trace, target_ch=int(ch), bank=mask_bank,
                amp_thr_default=25.0, center_index_default=40,
                thr=thr,        # your detection threshold (e.g., -200.0); ensures masked samples sit above it
                fill_value=None # leave None → auto-select 0.0 or thr+1; or set explicitly
            )

            print(f"[mask] ch={ch} segments={mask_info['segments']} "
                f"masked_samples={mask_info['masked_samples']} "
                f"({100*mask_info['masked_fraction']:.3f}%)")
        else:
            trace_masked = trace.copy()
            print("No templates yet, skip masking")



        det   = joint_utils.detect_negative_peaks_on_channel(trace_masked, thr=thr, min_gap=min_gap, pre=pre, post=post, max_events=max_events)


        if det["picked"].size < 10:
            # if det["picked"].size > 0:
            #     print(np.sort(det["picked"]))
            flag = 0
            print(f"Too few peaks ({det['picked'].size}), skipping")
            continue
            # raise RuntimeError("No events found—adjust thr/sign/channel.")

        sel   = joint_utils.select_top_by_amplitude(det["wf"], top_n=to_select)
        picked_sel = det["picked"][sel["sel_idx"]]


        # Optional quick plot (keeps steps visible, but minimal)
        # fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
        # axes[0].plot(det["t"], det["wf"].T, linewidth=0.5, alpha=0.4); axes[0].set_title(f"All waveforms (N={det['wf'].shape[0]}) | ch {ch}")
        # axes[1].plot(det["t"], sel["wf_sel"].T, linewidth=0.5, alpha=0.6); axes[1].plot(det["t"], sel["median_sel"], linewidth=2.0)
        # axes[1].set_title(f"Top-N by amplitude (N={min(to_select, det['wf'].shape[0])})")
        # plt.tight_layout(); plt.show()

        # ========== 2) Snippets + EI (full array) ==========
        state = joint_utils.extract_and_build_ei(raw_data, picked_sel, window_ei=window_ei, channels="all", reducer="median")
        state["ref_ch"]       = int(ch)
        state["ei_positions"] = ei_positions
        state["harm_cfg"]     = harm_cfg

        # (Optional) Diagnostic sweep – harmless no-op on membership
        state, _ = joint_utils.harm_map_diagnostic(state)

        print(f"[SPIKES AFTER PICKING]: {int(state['snips'].shape[2])}")
        # print(picked_sel)

        # ========== 3) Pruners (swappable order) ==========
        # P1: amplitude gate
        state, r1 = joint_utils.amplitude_gate_pruner(state, params=dict(topk=25, frac=0.75))

        print(f"[SPIKES AFTER amplitude gate]: {int(state['snips'].shape[2])}")

        # P2: split by harm map (two cells vs AB shard)
        state, r2 = joint_utils.two_cells_or_ab_shard_pruner(state, params=dict(
            rms_thr=10.0, dprime_thr=5.0, min_per_cluster=10, n_init=8, max_iter=60, lag_radius=0,
            p2p_thr=30.0, max_channels=80, min_channels=10, weight_by_p2p=True, weight_beta=0.7,
            rms_thr_support=10.0, asym_strong_z=2.0, asym_pure_z=1.0
        ))


        print(f"[SPIKES AFTER ab shard]: {int(state['snips'].shape[2])}")

        # P3: harm-map compliance (you can wrap this in a loop if desired)
        for it in range(1, MAX_ITERS + 1):
            # cur_ei = state["ei"]
            # fig, ax = plt.subplots(figsize=(15, 12))
            # pew.plot_ei_waveforms(
            #     cur_ei, ei_positions,
            #     ref_channel=int(ch), colors='black', scale=70.0, box_height=1.0, box_width=50.0, ax=ax
            # )
            # ax.set_title(f"Cur EI, iteration {it}")
            # plt.show()
            state, r3 = joint_utils.harm_compliance_pruner(state, params=dict(MEAN_THR=MEAN_THR, CHAN_THR=CHAN_THR, REF_THR=REF_THR))
            if r3["dropped"] == 0:
                break


        print(f"[SPIKES AFTER harm-map compliance]: {int(state['snips'].shape[2])}")

        # P4: bimodality split
        state, r4 = joint_utils.bimodality_split_pruner(state, params=dict(dprime_thr=5.0, min_per_cluster=5, rule="larger_n"))


        print(f"[SPIKES AFTER bimodality]: {int(state['snips'].shape[2])}")

        # ========== 4) Final plots ==========
        res_final = state["meta"].get("last_harm", None)
        # if res_final is not None:
        #     HM = np.asarray(res_final.get("harm_matrix", np.array([])))
        #     if HM.size and HM.ndim == 2 and HM.shape[0] > 0 and HM.shape[1] > 0:
        #         K, N = HM.shape
        #         plot_harm_heatmap(
        #             res_final, field="harm_matrix", sort_by_ptp=False,
        #             title=f"Final harm map (K={K}, N={N})"
        #         )
        #     else:
        #         K = int(HM.shape[0]) if HM.ndim == 2 else 0
        #         N = int(HM.shape[1]) if HM.ndim == 2 else 0
        #         print(f"[final] Harm map empty (K={K}, N={N}); skipping plot.")




        # Optional sanity check like your original
        proto_ei = state["ei"]
        p2p = proto_ei.max(axis=1) - proto_ei.min(axis=1)
        main_ch = int(np.argmax(p2p))

        if main_ch != ch:
            print(f"ABORT TEMPLATE: MAIN CHANNEL {main_ch} with amp {p2p[main_ch]:0.1f}, current CH {ch} with amp {p2p[ch]:0.1f}")
            flag = 0
            continue
        
        
        # Guard against the size of the proto template - if too few spikes, abort at this stage
        n_proto_spikes = int(state["snips"].shape[2])
        if n_proto_spikes <5:
            print(f"ABORT: proto template has too few spikes at this point. {n_proto_spikes} spikes")
            flag = 0
            continue

        print(f"[SCAN]: running continuous scan on channel {ch}")
        results = scan_continuous_harm_regions(
            raw_data=raw_data.T,
            final_ei_full=proto_ei,
            start_sample=0,
            stop_sample=max_length,   # exclusive
            mean_thr=-1.0,
            chan_thr=15.0,
            ref_thr=-3.0,
            center_index=40,
            region_max_len=20,
        )

        harm = results["harm_at_best"]
        n_sel, n_regions = harm.shape
        print(f"Spikes: {n_regions}, Channels: {n_sel}")
        if n_regions<2:
            print(f"ABORT: Scan returned {n_regions} spikes")
            flag = 0
            continue

        spike_times=np.asarray([r["best_idx"] for r in results["regions"]], dtype=np.int64)





        # # === INSERT ===


        # # raw_data is (total_samples, 512) — matches extract_snippets_fast_ram
        # snips, valid_times = extract_snippets_fast_ram(
        #     raw_data=raw_data,
        #     spike_times=spike_times,
        #     window=(-40, 80),
        #     selected_channels=np.arange(512, dtype=int),
        # )
        # # snips: (C, T, N). Average over spikes → EI
        # ei_from_best = snips.mean(axis=2).astype(np.float32)
        # ei_best_centered  = recenter_ei_to_ref_trough(ei_from_best,    center_index=results["center_index"])

        # res_split = None

        # res_split = split_first_good_channel_and_visualize(
        #         snips,                 # 512 x 121 x 661 (explained spikes, all channels)
        #         ei_best_centered,      # 512 x 121 (mean over explained spikes), centered
        #         ei_positions,
        #         rms_thr=10.0,
        #         dprime_thr=5.0,
        #         min_per_cluster=10,
        #         n_init=8, max_iter=60,
        #         lag_radius=0
        #     )

        # if res_split is not None:
        #     _ = plot_grouped_harm_maps_two_eis(
        #             res_split['EI0'], res_split['EI1'], snips, res_split['idx0'], res_split['idx1'],
        #             p2p_thr=30.0,      # same family as elsewhere
        #             max_channels=80,
        #             min_channels=10,
        #             lag_radius=0,
        #             weight_by_p2p=True,
        #             weight_beta=0.7,
        #             title_prefix=f"ch {int(ch)}"
        #         )

        #     metrics = classify_two_cells_vs_ab_shard(
        #         res_split['EI0'], res_split['EI1'], snips, res_split['idx0'], res_split['idx1'],
        #         p2p_thr=30.0, max_channels=80, min_channels=10,
        #         lag_radius=0, weight_by_p2p=True, weight_beta=0.7,
        #         rms_thr_support=10.0,
        #         asym_strong_z=2.0,  # tighten/loosen if needed
        #         asym_pure_z=1.0
        #     )

        #     fig, ax = plt.subplots(figsize=(15, 12))
        #     pew.plot_ei_waveforms(
        #         res_split['EI0'], ei_positions,
        #         ref_channel=int(ch), colors='black', scale=70.0, box_height=1.0, box_width=50.0, ax=ax
        #     )
        #     ax.set_title(f"Final EI0")
        #     plt.show()

        #     fig, ax = plt.subplots(figsize=(15, 12))
        #     pew.plot_ei_waveforms(
        #         res_split['EI1'], ei_positions,
        #         ref_channel=int(ch), colors='black', scale=70.0, box_height=1.0, box_width=50.0, ax=ax
        #     )
        #     ax.set_title(f"Final EI1")
        #     plt.show()

        # print(spike_times)


        # # === END INSERT ===












        mask_bank = mask_bank if 'mask_bank' in globals() else []

        print(f"Current templates: {len(mask_bank)+1}")

        # fig, ax = plt.subplots(1, 1, figsize=(10,12))
        # pew.plot_ei_waveforms(state["ei"], ei_positions, ref_channel=int(ch), ax=ax,
        #                     colors='C2', scale=70.0, box_height=1.0, box_width=50.0)
        # ax.set_title(f"Final proto EI, channel {ch}, {n_regions} spikes, total {len(mask_bank)+1} templates")
        # plt.tight_layout(); plt.show()



        entry = joint_utils.make_mask_entry(
            ei=proto_ei,
            spike_times=spike_times,
            relevant_ch=int(ch),      # mask when detecting on this channel later
            amp_thr=25.0,             # your default; adjust per-unit if needed
            center_index=40
        )
        mask_bank.append(entry)

        



# %%
print(snips.shape)

# %% [markdown]
# ### debug

# %%
print(np.sort(picked_sel))

# %% [markdown]
# ### save results to pickle

# %%
# Choose a path (e.g., alongside your notebook/project)
mask_bank_path = "/Volumes/Lab/Users/alexth/axolotl/masks/mask_bank_run6.pkl"

# On startup (before detection on first channel):
# mask_bank = joint_utils.load_mask_bank_pickle(mask_bank_path)

joint_utils.save_mask_bank_pickle(mask_bank_path, mask_bank)
print(f"Saved mask_bank ({len(mask_bank)} entries) → {mask_bank_path}")


# %% [markdown]
# #### load results from pickle

# %%
mask_bank_path = "/Volumes/Lab/Users/alexth/axolotl/masks/mask_bank_run6.pkl"
mask_bank = joint_utils.load_mask_bank_pickle(mask_bank_path)
print(f"Loaded {len(mask_bank)} entries")

# %%
snips, valid_times = extract_snippets_fast_ram(
    raw_data=raw_data,
    spike_times=np.asarray([70365, 192689, 221321, 732245]),
    window=(-40,80),
    selected_channels=np.arange(512, dtype=np.int32)
)
ei = snips.mean(axis=2).astype(np.float32)


fig, ax = plt.subplots(figsize=(20, 12))
pew.plot_ei_waveforms(
[mask_bank[6]["ei"], ei], ei_positions,
ref_channel=6, scale=70.0, box_height=1.0, box_width=50.0, ax=ax, colors=['black', 'red']
)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## KILOSORT

# %% [markdown]
# ### get kilosort units

# %%
import h5py
import numpy as np


ks_spike_h5_path = '/Volumes/Lab/Users/alexth/axolotl/201703151_kilosort_data001_spike_times.h5'
SAMPLE_RATE = 20000  # confirm this value

ks_spike_times = {}

with h5py.File(ks_spike_h5_path, 'r') as f:
    for key in f['/spikes']:
        unit_id = int(key.split('_')[1])
        spike_sec = f[f'/spikes/{key}'][:]
        spike_samples = (spike_sec * SAMPLE_RATE).astype(int)
        ks_spike_times[unit_id] = spike_samples

print(len(ks_spike_times))



# %%
import matplotlib.pyplot as plt

from axolotl_utils_ram import extract_snippets_fast_ram  # fast RAM path
from collision_utils import median_ei_adaptive           # robust EI (median, adaptive stride)
import plot_ei_waveforms as pew                          # grid plotter
from joint_utils import recenter_ei_to_ref_trough        # align trough to sample 40


# pick a unit to visualize
unit_id = sorted(ks_spike_times.keys())[0]  # or set explicitly, e.g. unit_id = 134
unit_id = 3
spike_samples = ks_spike_times[unit_id]

# optional: load EI positions from the same KS H5 if present
with h5py.File(ks_spike_h5_path, 'r') as f:
    ei_positions = f['/ei_positions'][:].T if '/ei_positions' in f else None

# ---- build EI from snippets (full array), align, plot ----
window = (-40, 80)                              # 121-sample EI, trough expected at index 40
all_ch = np.arange(raw_data.shape[1], dtype=int)

snips, valid_times = extract_snippets_fast_ram(
    raw_data=raw_data, 
    spike_times=spike_samples, 
    window=window, 
    selected_channels=all_ch
)
if snips.shape[2] == 0:
    raise RuntimeError("No valid snippets (all spikes near edges?).")

ei = median_ei_adaptive(snips).astype(np.float32)       # [512, 121]
ei = recenter_ei_to_ref_trough(ei, center_index=40)

ref_ch = int(np.argmin(ei.min(axis=1)))  # strongest negative trough

fig, ax = plt.subplots(figsize=(20, 12))
pew.plot_ei_waveforms(
    ei, ei_positions,
    ref_channel=ref_ch,
    scale=70.0, box_height=1.0, box_width=50.0, ax=ax
)
ax.set_title(f"KS unit {unit_id} | spikes used: {valid_times.size}", fontsize=12)
plt.tight_layout()
plt.show()


# %% [markdown]
# ### compute and save kilosort EIs

# %%
import numpy as np
import h5py, time, datetime
import matplotlib.pyplot as plt

from axolotl_utils_ram import extract_snippets_fast_ram
from collision_utils import median_ei_adaptive
import plot_ei_waveforms as pew
from joint_utils import recenter_ei_to_ref_trough

# --------------------- Config ---------------------
SAMPLE_RATE = 20_000
window = (-40, 80)                  # 121 samples; trough expected at index 40
center_index = 40
N_MAX_PER_EI  = 1000                       # <-- cap per-unit spikes
out_path = "/Volumes/Lab/Users/alexth/axolotl/kilosort_eis_data001_full_121sample.h5"


C = raw_data.shape[1]
T_ei = window[1] - window[0] + 1
unit_ids = np.array(sorted(ks_spike_times.keys()), dtype=np.int32)
U = unit_ids.size

# Try to grab ei_positions from the same H5 (optional)
ei_positions = None
try:
    with h5py.File(ks_spike_h5_path, "r") as f:
        if "/ei_positions" in f:
            # Expecting shape [2, C] or [C, 2]; normalize to [C, 2]
            arr = f["/ei_positions"][:]
            ei_positions = arr.T if arr.shape[0] == 2 else arr
            if ei_positions.shape != (C, 2):
                print(f"[warn] ei_positions shape {ei_positions.shape} != ({C},2); ignoring.")
                ei_positions = None
except Exception as e:
    print(f"[info] ei_positions not loaded: {e}")

# --------------------- Create output file ---------------------
with h5py.File(out_path, "w") as h5:
    dset_eis   = h5.create_dataset(
        "eis", shape=(U, C, T_ei), dtype="float32",
        chunks=(1, min(C, 128), T_ei), compression="gzip", compression_opts=4, shuffle=True
    )
    dset_uids  = h5.create_dataset("unit_ids", shape=(U,), dtype="int32")
    dset_nraw  = h5.create_dataset("n_spikes_raw", shape=(U,), dtype="int32")
    dset_nused = h5.create_dataset("n_spikes_used", shape=(U,), dtype="int32")
    dset_refch = h5.create_dataset("ref_channel", shape=(U,), dtype="int32")

    if ei_positions is not None:
        h5.create_dataset("ei_positions", data=ei_positions.astype(np.float32))

    # File-level attrs
    h5.attrs["sample_rate_Hz"]  = SAMPLE_RATE
    h5.attrs["window"]          = np.array(window, dtype=np.int32)
    h5.attrs["center_index"]    = center_index
    h5.attrs["ei_method"]       = "median_ei_adaptive"
    h5.attrs["created_utc"]     = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    h5.attrs["notes"]           = "Kilosort-aligned EIs for all units; trough recentered to sample 40 on strongest channel."

    # --------------------- Main loop ---------------------
    t0 = time.time()
    for i, uid in enumerate(unit_ids):
        spikes_full = np.asarray(ks_spike_times[uid], dtype=np.int64)
        # Subsample spikes if necessary: take the first N_MAX_PER_EI spikes deterministically
        spikes = spikes_full[:N_MAX_PER_EI] if spikes_full.size > N_MAX_PER_EI else spikes_full

        dset_uids[i] = uid
        dset_nraw[i] = spikes.size

        # Extract snippets on all channels
        snips, valid_times = extract_snippets_fast_ram(
            raw_data=raw_data,
            spike_times=spikes,
            window=window,
            selected_channels=np.arange(C, dtype=np.int32)
        )

        if snips.shape[2] == 0:
            # If all spikes fell near edges, write NaNs; mark counts
            dset_eis[i, :, :] = np.nan
            dset_nused[i] = 0
            dset_refch[i] = -1
            print(f"[{i+1}/{U}] unit {uid}: 0 usable snippets (edge).")
            continue

        # Robust EI
        ei = snips.mean(axis=2).astype(np.float32)   # [C, T_ei]
        # ei = median_ei_adaptive(snips).astype(np.float32)     # [C, T_ei]
        # Align trough to sample 40 based on the strongest channel
        ei = recenter_ei_to_ref_trough(ei, center_index=center_index)

        # Ref channel (strongest negative trough)
        ref_ch = int(np.argmin(ei.min(axis=1)))

        # Save
        dset_eis[i, :, :] = ei
        dset_nused[i]     = int(valid_times.size)
        dset_refch[i]     = ref_ch

        if (i+1) % 10 == 0 or i == U-1:
            elapsed = time.time() - t0
            print(f"[{i+1}/{U}] last uid={uid} | used {valid_times.size}/{spikes.size} | ref_ch={ref_ch} | {elapsed:.1f}s")

print(f"Done. Wrote {U} EIs to: {out_path}")


# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import plot_ei_waveforms as pew

# ---- path to the file we just wrote ----
ei_h5 = "/Volumes/Lab/Users/alexth/axolotl/kilosort_eis_data001_full_121sample.h5"

# Open HDF5 (keep handle around for lazy reads)
h5 = h5py.File(ei_h5, "r")

# Quick peek at what's inside
print("datasets:", list(h5.keys()))
unit_ids   = h5["unit_ids"][:]              # small; ok to load fully
n_raw      = h5["n_spikes_raw"][:]          # small vectors too
n_used     = h5["n_spikes_used"][:]
ref_ch_all = h5["ref_channel"][:]
C, T = h5["eis"].shape[1], h5["eis"].shape[2]
print(f"{unit_ids.size} units | C={C}, T={T}")
ei_positions = h5["ei_positions"][:] if "ei_positions" in h5 else None

# Map unit_id -> row index (unit_ids were saved sorted)
uid2idx = {int(uid): i for i, uid in enumerate(unit_ids)}

def load_ei_by_unit(uid: int):
    """Return (ei [C,T], ref_ch, n_raw, n_used). Reads only one EI from disk."""
    idx = uid2idx.get(int(uid), None)
    if idx is None:
        raise KeyError(f"unit_id {uid} not found.")
    ei = h5["eis"][idx, :, :]      # lazy slice reads one EI
    ref_ch = int(ref_ch_all[idx])
    return ei, ref_ch, int(n_raw[idx]), int(n_used[idx])

def plot_ei(uid: int):
    ei, ref_ch, n0, n1 = load_ei_by_unit(uid)
    if np.isnan(ei).all():
        print(f"unit {uid}: EI is NaN (no usable snippets).")
        return
    fig, ax = plt.subplots(figsize=(20, 12))
    pew.plot_ei_waveforms(
        ei, ei_positions if ei_positions is not None else None,
        ref_channel=ref_ch, scale=70.0, box_height=1.0, box_width=50.0, ax=ax
    )
    ax.set_title(f"KS unit {uid} | ref_ch={ref_ch} | used {n1}/{n0} spikes", fontsize=12)
    plt.tight_layout(); plt.show()

# 2) pick a unit and plot
uid_example = int(unit_ids[3])  # or a specific id, e.g. 134
plot_ei(uid_example)

# When done, close:
# h5.close()


# %% [markdown]
# ### compare KS to Ax

# %%
# Pre-reqs from earlier:
# h5, unit_ids, ref_ch_all, ei_positions
# mask_bank already in memory (list of entries)

res = joint_utils.find_ks_ax_matches(
    h5, mask_bank,
    amp_ratio_bounds=(0.7, 1.3),
    sig_thr=30.0,
    early_accept_all=0.9,   # all-channels cosine
    early_accept_excl=0.85,  # excluding-main cosine
    false_reject=0.60,
    plot_limit=30,
    ei_positions=ei_positions,
)

print("tested candidates:", res["tested_candidates"], "| plots made:", res["plotted"])

# Example queries:
ks_uid = int(unit_ids[0])
print("Matches for KS", ks_uid, res["ks_to_ax"].get(ks_uid, []))

# Pick some Ax uid you care about:
some_ax_uid = next(iter(res["ax_to_ks"])) if res["ax_to_ks"] else None
if some_ax_uid is not None:
    print("Matches for AX", some_ax_uid, res["ax_to_ks"][some_ax_uid])


# %% [markdown]
# # Find best KS - AX matches

# %% [markdown]
# ### AX -> KS

# %%
best_ax_to_ks = joint_utils.best_crossmatches(
    h5, mask_bank,
    direction="ax_to_ks",
    amp_ratio_bounds=(0.5, 1.5),
    sig_thr=30.0,
    accept_all=0.90,
    accept_excl=0.85,
    try_lags=(-1, 0, 1),
)
# Example: pick an AX uid you stored in entry["meta"]["unit_id"] (or its bank index if not set)
ax_uid = best_ax_to_ks["mapping"].keys().__iter__().__next__()
best_ax_to_ks["mapping"][ax_uid]


# %% [markdown]
# ### KS -> AX

# %%
best_ks_to_ax = joint_utils.best_crossmatches(
    h5, mask_bank,
    direction="ks_to_ax",
    amp_ratio_bounds=(0.5, 1.5),
    sig_thr=30.0,
    accept_all=0.90,
    accept_excl=0.85,
    try_lags=(-1, 0, 1),
    nan_floor=50.0,
)
# Example: get record for a specific KS uid
ks_uid = int(unit_ids[0])
best_ks_to_ax["mapping"][ks_uid]


# %% [markdown]
# ### Plot KS unit and main waveform by KS ID

# %%
u = 222                          # <-- UID in the title

uid2idx = {int(uid): i for i, uid in enumerate(unit_ids)}
i = uid2idx[u]                 # row index for that UID (likely 3)
spike_samples = np.asarray(ks_spike_times[u]).ravel()     # -> shape (N,)
ei_ks  = np.asarray(h5["eis"][i], dtype=np.float32)
C, T = ei_ks.shape
# main channel by most-negative trough
mins_per_ch = ei_ks.min(axis=1)                 # [C]
ch_main = int(np.argmin(mins_per_ch))          # channel with deepest negative trough
t_trough = int(np.argmin(ei_ks[ch_main]))       # sample index of the trough on that channel

fig, ax = plt.subplots(1, 1, figsize=(12,10))
pew.plot_ei_waveforms(ei_ks, ei_positions, ref_channel=ch_main,
                        colors='k', scale=70.0, box_height=1.0, box_width=50.0, ax=ax)
ax.set_title(f"KS uid={u} | idx={i} | main ch {ch_main} | <10M: {len(spike_samples[spike_samples < 10_000_000])} spikes out of {len(spike_samples)}")
plt.tight_layout(); plt.show()
# amplitudes
trough_amp = float(-ei_ks[ch_main, t_trough])   # positive magnitude of negative trough
p2p_amp = float(ei_ks[ch_main].max() - ei_ks[ch_main].min())
# plot
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
t = np.arange(T)
ax.plot(t, ei_ks[ch_main], lw=2)
ax.axvline(t_trough, ls="--", lw=1, alpha=0.6)
ax.plot([t_trough], [ei_ks[ch_main, t_trough]], "o", ms=6)
ax.set_xlabel("sample")
ax.set_ylabel("ADC")
ax.set_title(f"KS UID {u} | main ch {ch_main} | trough={trough_amp:.1f} ADC | p2p={p2p_amp:.1f} ADC")
plt.tight_layout()
plt.show()

print(len(spike_samples[spike_samples < 10_000_000]))


print(spike_samples[spike_samples < 10_000_000])

# %% [markdown]
# ### Print Ax units with specific main CH

# %%
req_ch = 149

ax_on_ch = []  # (ax_uid, bank_idx, main_ch, trough_amp_on_main)

for j, entry in enumerate(mask_bank):
    ei = np.asarray(entry["ei"], dtype=np.float32)  # [C,T]
    # main channel by deepest negative trough
    mins = ei.min(axis=1)                 # [C]
    ch_main = int(np.argmin(mins))        # channel of largest negative peak
    ch_main_amp = np.min(mins)
    if ch_main == req_ch:
        uid = int(entry.get("meta", {}).get("unit_id", j))
        trough_amp = float(-ei[ch_main].min())  # positive magnitude
        ax_on_ch.append((uid, j, ch_main, trough_amp))

        fig, ax = plt.subplots(1, 1, figsize=(12,10))
        pew.plot_ei_waveforms(ei, ei_positions, ref_channel=ch_main,
                                colors='k', scale=70.0, box_height=1.0, box_width=50.0, ax=ax)
        ax.set_title(f"AX uid={j} | main ch {ch_main}, amp {ch_main_amp:0.1f}")
        plt.tight_layout(); plt.show()

        # plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(ei[req_ch], lw=2)
        ax.plot(t, ei_ks[ch_main], lw=2)
        ax.set_xlabel("sample")
        ax.set_ylabel("ADC")
        plt.tight_layout()
        plt.show()

# Sort by descending trough magnitude for convenience
ax_on_ch.sort(key=lambda x: x[3], reverse=True)

print(f"Ax units with main channel == {req_ch}: {len(ax_on_ch)} found")
for uid, idx, ch_main, amp in ax_on_ch:
    print(f"AX uid={uid:6d} | bank_idx={idx:4d} | main_ch={ch_main} | trough={amp:.1f} ADC")

# %%
print(mask_bank[689]["spike_times"][40:100])

# 21368   22747   23605   32728   33054   46890   57587   58427   66787
#    70621   71550   74600   78385   82114   82251   89704   90265   95313
#    95732  111752  112316  112505  112868  118164  132936  144649  148768
#   156893  157596  160367  160864  161581  167379  173587  184453  184575
#   184941  196991  201517  202128  225687  230819  236983  251356  251594

# %% [markdown]
# ### Collect KS EI and spikes from ks UID

# %%
u = 36                          # <-- UID in the title

uid2idx = {int(uid): i for i, uid in enumerate(unit_ids)}

i = uid2idx[u]                 # row index for that UID (likely 3)
ei = np.asarray(h5["eis"][i], dtype=np.float32)


mins = ei.min(axis=1)                 # [C]
ch_main = int(np.argmin(mins))        # channel of largest negative peak
ch_main_amp = np.min(mins)
spike_samples = np.asarray(ks_spike_times[u]).ravel()     # -> shape (N,)

fig, ax = plt.subplots(figsize=(20, 12))
pew.plot_ei_waveforms(
    ei, ei_positions if ei_positions is not None else None,
    ref_channel=6, scale=70.0, box_height=1.0, box_width=50.0, ax=ax
)
plt.tight_layout(); 
ax.set_title(f"KS uid={u} | main ch {ch_main}, amp {ch_main_amp:0.1f}, spikes: {len(spike_samples)}")
plt.show()

print(spike_samples[:20])


# (optional) main channel
ref_ch = int(h5["ref_channel"][i]) if "ref_channel" in h5 else int(np.argmax(ei.max(1)-ei.min(1)))


# %% [markdown]
# ### Plot AX -> KS

# %%
import numpy as np
import matplotlib.pyplot as plt
import plot_ei_waveforms as pew

# H5 handles (same as before)
eis_ds      = h5["eis"]                # [Nks, C, T]
unit_ids    = h5["unit_ids"][:]        # [Nks]
ref_ch_all  = h5["ref_channel"][:] if "ref_channel" in h5 else None
uid2idx_ks  = {int(uid): i for i, uid in enumerate(unit_ids)}

# Map AX uid -> index in mask_bank
ax_uid2idx = {}
for i, entry in enumerate(mask_bank):
    ax_uid = int(entry.get("meta", {}).get("unit_id", i))
    ax_uid2idx[ax_uid] = i

def p2p_per_ch(ei):  # [C,T] -> [C]
    return (ei.max(axis=1) - ei.min(axis=1)).astype(np.float32)

def shift_ei_zero_pad(ei, lag):
    if lag == 0:
        return ei
    C, T = ei.shape
    out = np.zeros_like(ei)
    if lag > 0:
        out[:, lag:] = ei[:, :T-lag]
    else:
        s = -lag
        out[:, :T-s] = ei[:, s:]
    return out

n_plotted = 0

for ax_uid, rec in best_ax_to_ks["mapping"].items():
    ax_uid = int(ax_uid)

    # Get AX EI
    idx_ax = ax_uid2idx.get(ax_uid, None)
    if idx_ax is None:
        print(f"[skip] AX uid {ax_uid} not found in mask_bank.")
        continue
    ei_ax = np.asarray(mask_bank[idx_ax]["ei"], dtype=np.float32)

    # Decide if this AX unit needs plotting
    cos_all = rec.get("cos_all", np.nan)
    need_plot = (rec.get("uid_tgt", None) is None) or (not np.isfinite(cos_all)) or (cos_all < 0.60)
    if not need_plot:
        continue

    # AX main channel (source main channel recorded during matching)
    main_ch = int(rec.get("main_ch_src", np.argmax(p2p_per_ch(ei_ax))))

    # Build title
    title_bits = [f"AX {ax_uid} (main ch {main_ch})"]

    ks_uid = rec.get("uid_tgt", None)
    if ks_uid is None:
        # Plot AX only
        fig, ax = plt.subplots(1, 1, figsize=(20, 15))
        pew.plot_ei_waveforms(
            ei_ax, ei_positions, ref_channel=main_ch,
            colors="r", scale=70.0, box_height=1.0, box_width=50.0, ax=ax
        )
        title_bits.append("KS: none (no candidates)")
        ax.set_title(" | ".join(title_bits))
        plt.tight_layout(); plt.show()
        n_plotted += 1
        continue

    # Overlay AX with KS
    idx_ks = uid2idx_ks.get(int(ks_uid), None)
    if idx_ks is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 15))
        pew.plot_ei_waveforms(
            ei_ax, ei_positions, ref_channel=main_ch,
            colors="r", scale=70.0, box_height=1.0, box_width=50.0, ax=ax
        )
        title_bits.append(f"KS {ks_uid}: index invalid")
        ax.set_title(" | ".join(title_bits))
        plt.tight_layout(); plt.show()
        n_plotted += 1
        continue

    ei_ks = np.asarray(eis_ds[idx_ks], dtype=np.float32)

    # Apply best lag to the KS EI (target was KS in ax_to_ks)
    best_lag = int(rec.get("best_lag", 0))
    if best_lag != 0:
        ei_ks = shift_ei_zero_pad(ei_ks, best_lag)

    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    pew.plot_ei_waveforms(
        [ei_ks, ei_ax], ei_positions, ref_channel=main_ch,
        colors=["k", "r"], scale=70.0, box_height=1.0, box_width=50.0, ax=ax
    )
    c_all = rec.get("cos_all", np.nan)
    c_exc = rec.get("cos_excl", np.nan)
    reason = rec.get("reason", "—")
    title_bits.append(f"KS {int(ks_uid)} | cos_all={c_all:.2f} cos_exc={c_exc:.2f} | lag={best_lag} | {reason}")
    ax.set_title(" | ".join(title_bits))
    plt.tight_layout(); plt.show()
    n_plotted += 1

print(f"Plotted {n_plotted} AX units (no KS or cos_all<0.60).")


# %% [markdown]
# ### Plot KS -> AX

# %%
from collections import defaultdict

channel_counts = defaultdict(int)  # key = ref channel, value = # of unmatched units

for ks_uid, rec in best_ks_to_ax["mapping"].items():
    ks_uid = int(ks_uid)
    idx_ks = uid2idx.get(ks_uid, None)
    if idx_ks is None:
        continue

    cos_all = rec.get("cos_all", np.nan)
    need_count = (rec.get("uid_tgt", None) is None) or (not np.isfinite(cos_all)) or (cos_all < 0.60)
    if not need_count:
        continue

    main_ch = ref_ch_all[idx_ks] if ref_ch_all is not None else np.argmax(p2p_per_ch(np.asarray(eis_ds[idx_ks])))
    channel_counts[int(main_ch)] += 1

for ch in sorted(channel_counts.keys()):
    print(f"channel {ch} - {channel_counts[ch]} units")


# %%
import numpy as np
import matplotlib.pyplot as plt
import plot_ei_waveforms as pew

# H5 handles
eis_ds      = h5["eis"]                # [Nks, C, T]
unit_ids    = h5["unit_ids"][:]        # [Nks]
ref_ch_all  = h5["ref_channel"][:] if "ref_channel" in h5 else None
uid2idx     = {int(uid): i for i, uid in enumerate(unit_ids)}

def p2p_per_ch(ei):  # [C,T] -> [C]
    return (ei.max(axis=1) - ei.min(axis=1)).astype(np.float32)

n_plotted = 0

select_channel = 154

for ks_uid, rec in best_ks_to_ax["mapping"].items():
    ks_uid = int(ks_uid)
    idx_ks = uid2idx.get(ks_uid, None)
    if idx_ks is None:
        print(f"[skip] KS uid {ks_uid} not found in H5.")
        continue

    # Decide if this KS unit needs plotting
    cos_all = rec.get("cos_all", np.nan)
    need_plot = (rec.get("uid_tgt", None) is None) or (not np.isfinite(cos_all)) or (cos_all < 0.60)
    if not need_plot:
        continue

    if ref_ch_all[idx_ks] != select_channel:
        continue
    
    # Fetch KS EI and main channel
    ei_ks = np.asarray(eis_ds[idx_ks], dtype=np.float32)   # [C,T]
    if ref_ch_all is not None:
        main_ch = int(ref_ch_all[idx_ks])
    else:
        main_ch = int(np.argmax(p2p_per_ch(ei_ks)))

    # Build title pieces
    title_bits = [f"KS {ks_uid} (main ch {main_ch})"]

    # If an AX match exists but cos_all<0.60, overlay; else plot KS only
    ax_uid = rec.get("uid_tgt", None)
    if ax_uid is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 15))
        pew.plot_ei_waveforms(
            ei_ks, ei_positions, ref_channel=main_ch,
            colors="k", scale=70.0, box_height=1.0, box_width=50.0, ax=ax
        )
        title_bits.append("AX: none (no candidates)")
        ax.set_title(" | ".join(title_bits))
        plt.tight_layout(); plt.show()
        n_plotted += 1
        continue

    # Overlay with AX EI from mask_bank (index stored in idx_tgt)
    idx_ax = rec.get("idx_tgt", -1)
    if idx_ax < 0 or idx_ax >= len(mask_bank):
        fig, ax = plt.subplots(1, 1, figsize=(20, 15))
        pew.plot_ei_waveforms(
            ei_ks, ei_positions, ref_channel=main_ch,
            colors="k", scale=70.0, box_height=1.0, box_width=50.0, ax=ax
        )
        title_bits.append(f"AX {ax_uid}: index invalid ({idx_ax})")
        ax.set_title(" | ".join(title_bits))
        plt.tight_layout(); plt.show()
        n_plotted += 1
        continue

    ei_ax = np.asarray(mask_bank[idx_ax]["ei"], dtype=np.float32)
    # (Optional) apply best lag from rec if you want the very best overlay alignment
    best_lag = int(rec.get("best_lag", 0))
    if best_lag != 0:
        C, T = ei_ax.shape
        ei_ax_shift = np.zeros_like(ei_ax)
        if best_lag > 0:
            ei_ax_shift[:, best_lag:] = ei_ax[:, :T-best_lag]
        else:
            s = -best_lag
            ei_ax_shift[:, :T-s] = ei_ax[:, s:]
        ei_ax = ei_ax_shift

    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    pew.plot_ei_waveforms(
        [ei_ks, ei_ax], ei_positions, ref_channel=main_ch,
        colors=["k", "r"], scale=70.0, box_height=1.0, box_width=50.0, ax=ax
    )
    c_all = rec.get("cos_all", np.nan)
    c_exc = rec.get("cos_excl", np.nan)
    reason = rec.get("reason", "—")
    title_bits.append(f"AX {ax_uid} | cos_all={c_all:.2f} cos_exc={c_exc:.2f} | lag={int(rec.get('best_lag', 0))} | {reason}")
    ax.set_title(" | ".join(title_bits))
    plt.tight_layout(); plt.show()
    n_plotted += 1
    if n_plotted>10:
        break

print(f"Plotted {n_plotted} KS units (no AX or cos_all<0.60).")


# %% [markdown]
# ### something else - ignore

# %%
results = scan_continuous_harm_regions(
    raw_data=raw_data.T,
    final_ei_full=proto_ei,
    start_sample=0,
    stop_sample=2_000_000,   # exclusive
    mean_thr=-2.0,
    chan_thr=10.0,
    ref_thr=-5.0,
    center_index=40,
    region_max_len=20,
)

harm = results["harm_at_best"]
n_sel, n_regions = harm.shape
print(f"Spikes: {n_regions}, Channels: {n_sel}")
spike_times=np.asarray([r["best_idx"] for r in results["regions"]], dtype=np.int64)

mask_bank = mask_bank if 'mask_bank' in globals() else []

entry = joint_utils.make_mask_entry(
    ei=proto_ei,
    spike_times=spike_times,
    relevant_ch=int(ch),      # mask when detecting on this channel later
    amp_thr=25.0,             # your default; adjust per-unit if needed
    center_index=40
)
mask_bank.append(entry)



# %%

# 1) Plot compact harm map (ΔRMS per selected channel at each region’s best sample)
harm = results["harm_at_best"]             # (n_sel, n_regions)
sel_ch = results["selected_channels"]      # (n_sel,)
ref_ch = int(results["ref_channel"])      # integer channel id
v = np.nanpercentile(np.abs(harm), 95) if harm.size else 1.0


# harm: (n_channels, n_regions)  ->  X: (n_regions, n_channels)
X = harm.T

# Center features (channels). This “concatenates all channels” in the sense that
# every region’s feature vector is the ΔRMS across *all* selected channels.
Xc = X - X.mean(axis=0, keepdims=True)

# Optional (uncomment to z-score channels if you want equal weighting):
# Xc = Xc / (Xc.std(axis=0, ddof=1, keepdims=True) + 1e-12)

# PCA via SVD
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
scores = U * S  # rows = regions, cols = PCs
pc1, pc2 = scores[:, 0], scores[:, 1]
evr = (S**2) / (S**2).sum()  # explained variance ratio

# Plot PC1 vs PC2
plt.figure(figsize=(4,4))
plt.scatter(pc1, pc2, s=6, alpha=0.6)
plt.axhline(0, linestyle='--', linewidth=0.8)
plt.axvline(0, linestyle='--', linewidth=0.8)
plt.xlabel(f"PC1  ({evr[0]*100:.1f}% var)")
plt.ylabel(f"PC2  ({evr[1]*100:.1f}% var)")
plt.title("PCA of harm map (regions as points; features = channels)")
plt.tight_layout()
plt.show()


# ΔRMS histograms per channel (grid of small subplots)

# Threshold lines for reference (adjust if you used different gates)
REF_THR  = -5.0
CHAN_THR = 10.0

if harm.size == 0:
    print("harm_at_best is empty; nothing to plot.")
else:
    n_sel, n_regions = harm.shape

    # Robust global x-range so all histograms share axes
    all_vals = harm.ravel()
    x_min, x_max = np.nanpercentile(all_vals, [1, 99])
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        x_min, x_max = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        if x_min == x_max:
            x_min, x_max = x_min - 1.0, x_max + 1.0

    bins = 100

    # Grid size ~ square
    cols = int(np.ceil(np.sqrt(n_sel)))
    rows = int(np.ceil(n_sel / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 1.8), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for i in range(rows * cols):
        ax = axes.flat[i]
        if i < n_sel:
            ch_id = int(sel_ch[i])
            x = harm[i, :]  # ΔRMS samples for this channel across regions

            ax.hist(x, bins=bins, range=(x_min, x_max), histtype='stepfilled', alpha=0.7)
            ax.axvline(0.0, color='black', linestyle='--', linewidth=0.8)
            ax.axvline(CHAN_THR, color='red', linestyle=':', linewidth=0.8)

            if ch_id == ref_ch:
                ax.axvline(REF_THR, color='magenta', linestyle='-.', linewidth=0.8)
                ax.set_title(f"{ch_id} (ref)", fontsize=9)
            else:
                ax.set_title(f"{ch_id}", fontsize=9)

            ax.tick_params(labelsize=8)
        else:
            ax.axis('off')

    fig.suptitle("ΔRMS per channel (harm at best samples)", y=1.02)
    plt.tight_layout()
    plt.show()



plt.figure(figsize=(20, 6))
im = plt.imshow(harm, aspect='auto', cmap='coolwarm', vmin=-v, vmax=v, interpolation='nearest')
plt.colorbar(im, label='ΔRMS')
plt.yticks(np.arange(len(sel_ch)), sel_ch)
plt.xlabel('Region index')
plt.ylabel('Channel (selected)')
plt.title('Harm map at best samples (compact)')
plt.tight_layout()
plt.show()

# 2) Recompute EI from the best indices
best_indices = np.array([r["best_idx"] for r in results["regions"]], dtype=np.int64)

C, T = final_ei_full.shape
ref_ch = results["ref_channel"]
ci     = results["center_index"]
pre  = -ci
post = T - ci - 1
selected_channels = np.arange(C, dtype=int)  # full EI

spike_times=np.asarray([r["best_idx"] for r in results["regions"]], dtype=np.int64)

# raw_data is (total_samples, 512) — matches extract_snippets_fast_ram
snips, valid_times = extract_snippets_fast_ram(
    raw_data=raw_data,
    spike_times=spike_times,
    window=(pre, post),
    selected_channels=selected_channels,
)
# snips: (C, T, N). Average over spikes → EI
ei_from_best = snips.mean(axis=2).astype(np.float32)

# 3) Overlay recomputed EI with original (both recentred so trough aligns at sample 40)
ei_orig_centered  = recenter_ei_to_ref_trough(proto_ei, center_index=results["center_index"])
ei_best_centered  = recenter_ei_to_ref_trough(ei_from_best,    center_index=results["center_index"])

fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(111)
pew.plot_ei_waveforms(
    [ei_orig_centered, ei_best_centered],
    positions=ei_positions,
    ref_channel=ref_ch,
    scale=70.0,
    ax=ax,
    colors=['green', 'red'],
    alpha=[0.9, 0.9],
    linewidth=[0.6, 0.6],
    box_height=1.0,
    box_width=50.0,
    aspect=1.0,
)
ax.set_title('EI overlay: original (green) vs recomputed from best indices (red)')
plt.tight_layout()
plt.show()


wfs = snips[ref_ch, :, :]           # (T, N)
mean_wf = wfs.mean(axis=1)          # (T,)

plt.figure(figsize=(20, 12))
plt.plot(wfs, color='black', linewidth=0.25, alpha=0.08)  # ALL traces
plt.plot(mean_wf, color='red', linewidth=2.5, alpha=0.95)
plt.plot(ei_orig_centered[ref_ch], color='green', linewidth=2.5, alpha=0.95)
plt.axvline(ci, color='k', linestyle='--', linewidth=1, alpha=0.6)

plt.title(f"Ref channel {ref_ch}: all waveforms (N={wfs.shape[1]}) + new mean (red) + original mean (green)", fontsize=14)
plt.xlabel("Sample (aligned; trough at center_index)", fontsize=12)
plt.ylabel("μV", fontsize=12)
plt.tight_layout()
plt.show()




# ----- EI difference assessment (orig vs best) -----
_ = assess_ei_drift(
        ei_orig_centered,
        ei_best_centered,     # or final_ei_full if that's your current EI
        rms_thr=5.0,
        channel_ids=np.arange(ei_orig_centered.shape[0], dtype=int),
        title_prefix=f"ch {int(ch)}"
    )


# %%

# === Mix check: k-means per-channel on explained spikes (post-explain, using ei_best_centered) ===

# 1) Channels by RMS on ei_best_centered
rms = np.sqrt(np.mean(ei_best_centered**2, axis=1)).astype(np.float32)  # [C]
channels = np.flatnonzero(rms > 10.0).astype(int)
print(f"[mix-check] channels passing RMS>10: {channels.size} → {channels[:20]}{' ...' if channels.size>20 else ''}")

if channels.size == 0:
    print("[mix-check] No channels pass RMS>10; skipping k-means diagnostics.")
else:
    # 2) Subset snips and EI to those channels (keep global labels in channel_ids)
    snips_sub = snips[channels, :, :]            # [C_sel, L, N]
    E_sub     = ei_best_centered[channels, :]    # [C_sel, L]

    # 3) Run diagnostics (k=2 on full waveform), 4 panels per row
    diag = kmeans_split_diagnostics(
        snips_sub,
        E_sub,
        channel_ids=channels,   # label with GLOBAL ch ids
        rms_thr=0.0,            # we've already filtered by RMS>10
        n_init=8,
        max_iter=60,
        ncols=4,
        title_prefix="post-explain (ei_best_centered)"
    )

# === Split & visualize based on first good channel (post-explain) ===
res_split = None

res_split = split_first_good_channel_and_visualize(
        snips,                 # 512 x 121 x 661 (explained spikes, all channels)
        ei_best_centered,      # 512 x 121 (mean over explained spikes), centered
        ei_positions,
        rms_thr=10.0,
        dprime_thr=5.0,
        min_per_cluster=10,
        n_init=8, max_iter=60,
        lag_radius=0
    )

if res_split is not None:
    _ = plot_grouped_harm_maps_two_eis(
            res_split['EI0'], res_split['EI1'], snips, res_split['idx0'], res_split['idx1'],
            p2p_thr=30.0,      # same family as elsewhere
            max_channels=80,
            min_channels=10,
            lag_radius=0,
            weight_by_p2p=True,
            weight_beta=0.7,
            title_prefix=f"ch {int(ch)}"
        )

    metrics = classify_two_cells_vs_ab_shard(
        res_split['EI0'], res_split['EI1'], snips, res_split['idx0'], res_split['idx1'],
        p2p_thr=30.0, max_channels=80, min_channels=10,
        lag_radius=0, weight_by_p2p=True, weight_beta=0.7,
        rms_thr_support=10.0,
        asym_strong_z=2.0,  # tighten/loosen if needed
        asym_pure_z=1.0
    )

    fig, ax = plt.subplots(figsize=(15, 12))
    pew.plot_ei_waveforms(
        res_split['EI0'], ei_positions,
        ref_channel=int(ch), colors='black', scale=70.0, box_height=1.0, box_width=50.0, ax=ax
    )
    ax.set_title(f"Final EI0")
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 12))
    pew.plot_ei_waveforms(
        res_split['EI1'], ei_positions,
        ref_channel=int(ch), colors='black', scale=70.0, box_height=1.0, box_width=50.0, ax=ax
    )
    ax.set_title(f"Final EI1")
    plt.show()



# %%
# Suppose these exist after your pipeline:
#   proto_ei      : [C, L] EI aligned (use the same you already compute)
#   spike_times   : 1D np.int64 array of absolute samples for that unit
#   ch            : int, the channel you consider the "relevant channel" for masking

mask_bank = mask_bank if 'mask_bank' in globals() else []

entry = joint_utils.make_mask_entry(
    ei=proto_ei,
    spike_times=spike_times,
    relevant_ch=int(ch),      # mask when detecting on this channel later
    amp_thr=25.0,             # your default; adjust per-unit if needed
    center_index=40
)
mask_bank.append(entry)

print(len(mask_bank))


# %%
print(len(picked_sel))
print(len(spike_times))
print(np.sort(picked_sel))
print(np.sort(spike_times[res_split['idx0']]))
print(np.sort(spike_times[res_split['idx1']]))



start = 42091-40 # good examples: 83819, 65786
length = 121
raw_snippet = raw_data[start : start + length, 0:512].astype(np.float32).T

plt.figure(figsize=(25,20))
pew.plot_ei_waveforms([raw_snippet, mask_bank[4]["ei"]], ei_positions, scale=70.0, box_height=1.0, box_width=50.0, colors=['black', 'red'])
plt.show()

# %%

# === Mix check: k-means per-channel on explained spikes (post-explain, using ei_best_centered) ===

idx = 1

if idx==0:
    cur_ei = res_split['EI0']
    snips_tmp = snips[:,:,res_split['idx0']]
else:
    cur_ei = res_split['EI1']
    snips_tmp = snips[:,:,res_split['idx1']]

# 1) Channels by RMS on ei_best_centered
rms = np.sqrt(np.mean(cur_ei**2, axis=1)).astype(np.float32)  # [C]
channels = np.flatnonzero(rms > 10.0).astype(int)
print(f"[mix-check] channels passing RMS>10: {channels.size} → {channels[:20]}{' ...' if channels.size>20 else ''}")

if channels.size == 0:
    print("[mix-check] No channels pass RMS>10; skipping k-means diagnostics.")
else:
    # 2) Subset snips and EI to those channels (keep global labels in channel_ids)
    snips_sub = snips_tmp[channels, :, :]            # [C_sel, L, N]
    
    E_sub     = cur_ei[channels, :]    # [C_sel, L]

    # 3) Run diagnostics (k=2 on full waveform), 4 panels per row
    diag = kmeans_split_diagnostics(
        snips_sub,
        E_sub,
        channel_ids=channels,   # label with GLOBAL ch ids
        rms_thr=0.0,            # we've already filtered by RMS>10
        n_init=8,
        max_iter=60,
        ncols=4,
        title_prefix="post-explain (res_split['EI0'])"
    )

# === Split & visualize based on first good channel (post-explain) ===
res_split1 = None

res_split1 = split_first_good_channel_and_visualize(
        snips_tmp,                 # 512 x 121 x 661 (explained spikes, all channels)
        cur_ei,      # 512 x 121 (mean over explained spikes), centered
        ei_positions,
        rms_thr=10.0,
        dprime_thr=5.0,
        min_per_cluster=10,
        n_init=8, max_iter=60,
        lag_radius=0
    )

if res_split1 is not None:
    _ = plot_grouped_harm_maps_two_eis(
            res_split1['EI0'], res_split1['EI1'], snips, res_split1['idx0'], res_split1['idx1'],
            p2p_thr=30.0,      # same family as elsewhere
            max_channels=80,
            min_channels=10,
            lag_radius=0,
            weight_by_p2p=True,
            weight_beta=0.7,
            title_prefix=f"ch {int(ch)}"
        )

    metrics = classify_two_cells_vs_ab_shard(
        res_split1['EI0'], res_split1['EI1'], snips, res_split1['idx0'], res_split1['idx1'],
        p2p_thr=30.0, max_channels=80, min_channels=10,
        lag_radius=0, weight_by_p2p=True, weight_beta=0.7,
        rms_thr_support=10.0,
        asym_strong_z=2.0,  # tighten/loosen if needed
        asym_pure_z=1.0
    )

    fig, ax = plt.subplots(figsize=(15, 12))
    pew.plot_ei_waveforms(
        res_split1['EI0'], ei_positions,
        ref_channel=int(ch), colors='black', scale=70.0, box_height=1.0, box_width=50.0, ax=ax
    )
    ax.set_title(f"Final EI0")
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 12))
    pew.plot_ei_waveforms(
        res_split1['EI1'], ei_positions,
        ref_channel=int(ch), colors='black', scale=70.0, box_height=1.0, box_width=50.0, ax=ax
    )
    ax.set_title(f"Final EI1")
    plt.show()



# %% [markdown]
# ### other attempts

# %%
# --- probe a second unit on the SAME channel (single pass, no amplitude scaling) ---
def probe_next_unit_on_channel(
    ch: int,
    spike_times_unit1,            # 1-D array of global sample indices (neg-peak aligned) for the first unit
    proto_ei_full: np.ndarray,    # [C, L_ei] median EI for the first unit (full array)
    raw_data: np.ndarray,         # [T, C] int16 (or float) raw; we do NOT mutate it
    t0: int,                      # start sample (inclusive) of the 2M window
    t1: int,                      # end sample (exclusive) of the 2M window
    params: dict = None,
):
    """
    Single-channel residual-style probe for another unit on the same reference channel.
    No amplitude scaling. We zero (mask) windows around the known unit's spikes on channel ch,
    then threshold-detect new peaks, align by peak, and plot overlays.

    Steps:
      1) Copy the channel trace [t0:t1] to avoid mutating raw_data.
      2) Compute the mask window on ch from proto_ei_full[ch] as the contiguous span where |ei|>mask_abs_amp.
         Use the trough index as time 0 (spike times are neg-peak aligned).
      3) For each spike in spike_times_unit1 ∩ [t0,t1), zero the masked window on the copied trace.
      4) Threshold-detect (thr_neg), apply refractory, keep up to max_peaks strongest negatives.
      5) Extract snippets around each detected peak (pre/post samples), overlay-plot and histogram.

    Returns:
      result: dict with detected_peak_times (global), n_detected, mask_offsets, and a few sanity stats
      figs:   list of matplotlib Figures (overlay + hist)
    """
    if params is None:
        params = {}
    # --- knobs (safe defaults) ---
    mask_abs_amp   = float(params.get("mask_abs_amp", 25.0))   # EI|amp|>25 defines the zero-mask window on ch
    thr_neg        = float(params.get("thr_neg", -200.0))       # detection threshold on residual ch
    refractory     = int(params.get("refractory", 20))          # samples (20 @20kHz ≈ 1 ms)
    max_peaks      = int(params.get("max_peaks", 400))
    pre_samp       = int(params.get("pre_samples", 30))
    post_samp      = int(params.get("post_samples", 90))

    # --- 1) copy the channel trace window ---
    x = np.asarray(raw_data[t0:t1, ch], dtype=np.float32).copy()
    Nwin = x.size
    if Nwin <= (pre_samp + post_samp + 3):
        raise ValueError(f"[probe_next_unit] Window too short: Nwin={Nwin}")

    # --- 2) mask window from EI on this channel (|ei|>mask_abs_amp) ---
    ei_ch = np.asarray(proto_ei_full[ch], dtype=np.float32)
    ei_abs = np.abs(ei_ch)
    nz = np.flatnonzero(ei_abs > mask_abs_amp)
    if nz.size == 0:
        # fallback if EI is tiny or poorly centered
        trough_idx = int(np.argmin(ei_ch))
        left_rel, right_rel = -pre_samp, post_samp
    else:
        left_idx  = int(nz.min())
        right_idx = int(nz.max())
        trough_idx = int(np.argmin(ei_ch))  # negative peak index in EI
        left_rel  = left_idx - trough_idx
        right_rel = right_idx - trough_idx
        # sanity bound the mask so it can't exceed our snippet span
        left_rel  = int(max(left_rel, -pre_samp))
        right_rel = int(min(right_rel, post_samp))
    # Keep these for the log/return
    mask_offsets = (left_rel, right_rel)

    # --- 3) zero-mask the known spikes on the copied trace (NO scaling) ---
    st = np.asarray(spike_times_unit1, dtype=np.int64)
    # keep only spikes that fall inside [t0, t1)
    st = st[(st >= t0) & (st < t1)]
    # convert to local window indices and apply mask window
    for s in st:
        a = int((s - t0) + left_rel)
        b = int((s - t0) + right_rel)
        if b < 0 or a >= Nwin:
            continue  # completely out of window
        a = max(a, 0)
        b = min(b, Nwin - 1)
        x[a:b+1] = 0.0  # zero the masked interval

    # --- 4) threshold detect NEW negatives on residual trace (simple, robust) ---
    # candidates: strict local minima below threshold
    x_mid = x[1:-1]
    cand = np.where((x_mid < x[:-2]) & (x_mid <= x[2:]) & (x_mid <= thr_neg))[0] + 1
    if cand.size:
        # greedy refractory by amplitude (most negative first)
        order = np.argsort(x[cand])  # ascending (most negative first)
        picked = []
        last_kept = -10**9
        for idx in cand[order]:
            if picked and abs(idx - last_kept) < refractory:
                continue
            picked.append(int(idx))
            last_kept = int(idx)
            if len(picked) >= max_peaks:
                break
        det_idx_local = np.array(sorted(picked), dtype=np.int64)
    else:
        det_idx_local = np.empty(0, dtype=np.int64)

    # convert local indices to GLOBAL sample times
    det_times_global = det_idx_local + t0

    # --- 5) extract snippets (aligned by detected peak) and PLOT overlays ---
    figs = []

    if det_idx_local.size:
        # gather snippets on residual (for visualization)
        valid = (det_idx_local >= pre_samp) & (det_idx_local < (Nwin - post_samp))
        det_idx_local = det_idx_local[valid]
        det_times_global = det_times_global[valid]
        if det_idx_local.size:
            L = pre_samp + post_samp + 1
            snips = np.zeros((det_idx_local.size, L), dtype=np.float32)
            for i, t in enumerate(det_idx_local):
                snips[i, :] = x[t-pre_samp : t+post_samp+1]

            # Overlay plot
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(1,1,1)
            tvec = np.arange(-pre_samp, post_samp+1, dtype=np.int32)
            for i in range(snips.shape[0]):
                ax.plot(tvec, snips[i], linewidth=0.6, alpha=0.25)
            ax.plot(tvec, snips.mean(axis=0), linewidth=2.2, alpha=0.95)
            ax.axvline(0, color='k', linewidth=1.0, linestyle='--')
            ax.set_title(f"Ch {ch}: residual-detected snippets (n={snips.shape[0]})  thr={thr_neg:g}  mask_offsets={mask_offsets}")
            ax.set_xlabel("Samples (peak-aligned)")
            ax.set_ylabel("Voltage (raw units)")
            figs.append(fig)

            # Amplitude histogram at peak (optional, helpful)
            pk = snips[:, pre_samp]  # value at detected peak
            fig2 = plt.figure(figsize=(7, 4))
            ax2 = fig2.add_subplot(1,1,1)
            ax2.hist(pk, bins=40)
            ax2.set_title("Peak amplitudes of residual-detected spikes")
            ax2.set_xlabel("Amplitude (raw units)")
            ax2.set_ylabel("Count")
            figs.append(fig2)

    # --- result dict ---
    result = {
        "status": "ok",
        "ch": int(ch),
        "window": (int(t0), int(t1)),
        "mask_offsets": mask_offsets,            # (left_rel, right_rel) relative to trough
        "n_masked_spikes": int(st.size),
        "n_detected": int(det_times_global.size),
        "detected_peak_times": det_times_global, # global sample indices
        "thr_neg": thr_neg,
        "refractory": refractory,
    }
    print(f"[next-unit probe] ch={ch} window=({t0},{t1}) masked={st.size} found={result['n_detected']}  mask_offsets={mask_offsets} thr={thr_neg:g}")

    return result, figs


# %%
# Example: 2M samples starting at 10,000,000
ch = 3
t0, t1 = 0, 2_000_000

# spike_times_unit1: 1-D array of *global* sample indices for unit1 (neg-peak times)
# proto_ei_full: EI for unit1, shape [C, L_ei] (median)

params = {
    "mask_abs_amp": 25.0,   # |EI[ch]| > 25 → zero-mask window
    "thr_neg": -200.0,      # detection threshold on the masked trace
    "refractory": 40,       # samples
    "max_peaks": 400,
    "pre_samples": 20,
    "post_samples": 40,
}

res, figs = probe_next_unit_on_channel(
    ch=ch,
    spike_times_unit1=spike_times,
    proto_ei_full=ei_from_best,
    raw_data=raw_data,
    t0=t0, t1=t1,
    params=params,
)

# If you want to persist the figures:
# save_figs_to_pdf(figs, f"/mnt/data/next_unit_probe_ch{ch}_{t0}-{t1}.pdf")


# %%


# ---------- Source trace (EDIT THIS LINE to your array name) ----------
trace = raw_data[:,ch].astype(np.float32).T  # raw: shape (n_channels, n_samples)

x = trace

picked = np.array(res["detected_peak_times"], dtype=int)

if picked.size == 0:
    print("No events found—check threshold/sign/channel.")
else:
    # ---------- Extract waveforms ----------
    wf = np.stack([x[i - pre : i + post + 1] for i in picked], axis=0)  # shape (N, pre+post+1)
    t = np.arange(-pre, post + 1)

    # ---------- Select top-N by amplitude on ref channel (prescreen) ----------
    # Use negative-peak magnitude within the local window on channel `ch`
    amp_all = -wf.min(axis=1)  # shape (N,)

    k = min(to_select, wf.shape[0])
    sel_idx = np.argsort(-amp_all)[:k]   # indices of top-N by amplitude

    wf_sel  = wf[sel_idx]
    med_sel = np.median(wf_sel, axis=0)


    # ---------- Plots ----------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)

    ax = axes[0]
    ax.plot(t, wf.T, linewidth=0.5, alpha=0.4)
    ax.set_title(f"All waveforms (N={wf.shape[0]})  | ch {ch}")
    ax.set_xlabel("samples from peak"); ax.set_ylabel("ADC")

    ax = axes[1]
    ax.plot(t, wf_sel.T, linewidth=0.5, alpha=0.55, label=f"selected {k}")
    ax.plot(t, med_sel, linewidth=2.5, label="median(selected)")
    ax.set_title("Selected top-N by amplitude + median")
    ax.set_xlabel("samples from peak")

    plt.tight_layout()
    plt.show()

    # ===== EI from selected spikes (full array) =====

    # Use your selected spike times on ch
    times_sel = picked[sel_idx].astype(np.int64)

    # EI window (use your standard full EI window)
    window_ei = (-40, 80)   # center = -window_ei[0]
    C_full    = raw_data.shape[1]
    all_ch    = np.arange(C_full, dtype=int)

    # Extract snippets on ALL channels at those times
    snips_ei, valid_times = extract_snippets_fast_ram(raw_data, times_sel, window_ei, all_ch)
    if snips_ei.shape[2] == 0:
        raise RuntimeError("No valid snippets for EI (check edges / times).")

    # EI = mean across selected events
    ei_sel_full = median_ei_adaptive(snips_ei)   # [C_full, L_ei]


    # Identify main channel by the most-negative trough on the current EI
    try:
        ch0, t_neg0 = main_channel_and_neg_peak(ei_sel_full)
    except Exception:
        # fallback: same logic inline
        mins = ei_sel_full.min(axis=1)
        ch0  = int(np.argmin(mins))
        t_neg0 = int(np.argmin(ei_sel_full[ch0]))

    print(f"Main channel: {ch0}")
    amps = (-snips_ei[ch0, t_neg0, :]).astype(np.float32)   # positive trough magnitude per spike
    if amps.size:
        TOPK = 25
        FRAC = 0.75
        k = min(TOPK, amps.size)
        mu_top = float(np.mean(np.sort(amps)[-k:]))           # mean of top-k trough magnitudes
        thr_amp = FRAC * mu_top
        keep_amp = amps >= thr_amp
        n_drop = int((~keep_amp).sum())
        if n_drop > 0:
            print(f"amplitude gate on main ch {ch0} @t={t_neg0}: "
                    f"drop {n_drop}/{amps.size} (μ_top{k}={mu_top:.1f} → thr={thr_amp:.1f})")
            # print(f"Dropped: {times_sel[~keep_amp]}")
            snips_ei = snips_ei[:, :, keep_amp]
            if snips_ei.shape[2] == 0:
                print("amplitude gate removed all spikes; stopping.")
            else: # Rebuild EI on the reduced set to keep harm-map inputs consistent
                ei_sel_full = median_ei_adaptive(snips_ei)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3), sharey=True)

    ax = axes[0]
    ax.plot(snips_ei[ch0], linewidth=0.5, alpha=0.4)
    ax.set_title(f"All waveforms on main ch {ch0}")
    ax.set_xlabel("samples from peak"); ax.set_ylabel("ADC")

    ax = axes[1]
    ax.plot(snips_ei[ch], linewidth=0.5, alpha=0.4)
    ax.set_title(f"All waveforms on ref ch {ch}")
    ax.set_xlabel("samples from peak"); ax.set_ylabel("ADC")

    ax = axes[2]
    extra_ch_to_plot = 330
    ax.plot(snips_ei[extra_ch_to_plot], linewidth=0.5, alpha=0.4)
    ax.set_title(f"All waveforms on extra ch {extra_ch_to_plot}")
    ax.set_xlabel("samples from peak"); ax.set_ylabel("ADC")

    plt.tight_layout()
    plt.show()


    # Plot the EI (overlay-ready if you add more later)
    fig, ax = plt.subplots(1, 1, figsize=(10,12))
    plot_ei_waveforms.plot_ei_waveforms(
        ei_sel_full, ei_positions, ref_channel=int(ch), ax=ax,
        colors='C2', scale=70.0, box_height=1.0, box_width=50.0
    )
    ax.set_title(f"EI from selected spikes on ch {int(ch)} (k={len(sel_idx)})")
    plt.tight_layout(); plt.show()




    # 1) Preselect channels from EI (same rule the harm-map uses)
    chans_pre, ptp = select_template_channels(
        ei_sel_full, p2p_thr=p2p_thr, max_n=max_channels, min_n=min_channels, force_include_main=True
    )

    # Make sure the main (most-negative) channel is present
    ch_main, _ = main_channel_and_neg_peak(ei_sel_full)
    if ch_main not in chans_pre:
        # replace weakest with main, re-sort by p2p desc, keep unique
        pool = np.concatenate([chans_pre[:-1], [ch_main]])
        chans_pre = np.array(sorted(set(pool), key=lambda c: ptp[c], reverse=True), dtype=int)

    print(f"channels used: {len(chans_pre)} (main={int(ch_main)})")

    # 3) Harm-map on the reduced channel set
    #    Lock selection inside harm-map to use *exactly* these channels by setting p2p_thr very low and min=max=len(chans_pre).
    res = compute_harm_map_noamp(
        ei                   = ei_sel_full[chans_pre],   # [K, L]
        snips                = snips_ei[chans_pre],            # [K, L, N]
        p2p_thr              = -1e9,                 # include all provided chans
        max_channels         = len(chans_pre),
        min_channels         = len(chans_pre),
        lag_radius           = lag_radius,
        weight_by_p2p        = True,
        weight_beta          = 0.7,
        force_include_main   = True,
    )

    # 4) Plot
    plot_harm_heatmap(res, field="harm_matrix",
                    title=f"Harm map ΔRMS | ch {int(ch)} | N={valid_times.size}")


    # Work on a copy of your current snippets
    snips_cur = snips_ei.copy()
    C_avail, L, N0 = snips_cur.shape


    # === Split & visualize based on first good channel (post-explain) ===
    res_split = None
    res_split = split_first_good_channel_and_visualize(
            snips_cur,                 # 512 x 121 x 661 (explained spikes, all channels)
            ei_sel_full,      # 512 x 121 (mean over explained spikes), centered
            ei_positions,
            rms_thr=10.0,
            dprime_thr=5.0,
            min_per_cluster=10,
            n_init=8, max_iter=60,
            lag_radius=0
        )
    
    if res_split is not None:
        metrics = classify_two_cells_vs_ab_shard(
            res_split['EI0'], res_split['EI1'], snips_cur, res_split['idx0'], res_split['idx1'],
            p2p_thr=30.0, max_channels=80, min_channels=10,
            lag_radius=0, weight_by_p2p=True, weight_beta=0.7,
            rms_thr_support=10.0,
            asym_strong_z=2.0,  # tighten/loosen if needed
            asym_pure_z=1.0
        )

        if metrics["label"] == "two cells":
            if len(res_split['idx0'])>=len(res_split['idx1']):
                snips_cur = snips_cur[:,:,res_split['idx0']]
            else:
                snips_cur = snips_cur[:,:,res_split['idx1']]

    C_avail, L, N0 = snips_cur.shape

    print(f"[start] snippets: C={C_avail}, L={L}, N={N0}")

    final_res = None
    final_ei_full = None
    bimodal_plotted = False
    bimodal_payload = None   # store first detected bimodal split to finalize at the end




    for it in range(1, MAX_ITERS + 1):
        # --- EI from current spikes (no lags, straight mean) ---
        ei_full = median_ei_adaptive(snips_cur)     # [C_avail, L]
     

        # --- Harm map with fresh channel selection each round ---
        res_it = compute_harm_map_noamp(
            ei=ei_full, snips=snips_cur,
            p2p_thr=50.0,               # your usual selector; adjust if needed
            max_channels=C_avail,       # allow selector to choose from what's available
            min_channels=10,
            lag_radius=LAG_RADIUS,
            weight_by_p2p=True, weight_beta=0.7,
            force_include_main=True
        )
        HM    = np.asarray(res_it["harm_matrix"])                 # [K_sel, N_cur]
        sel   = np.asarray(res_it["selected_channels"], int)      # [K_sel]
        K_sel, N_cur = HM.shape

        # --- Build the reject mask from your three rules ---
        mean_d  = HM.mean(axis=0)          # [N_cur]
        max_d   = HM.max(axis=0)           # [N_cur]

        # ref channel row (if not selected, we can't check REF_THR; warn & skip that term)
        ref_matches = np.where(sel == int(ch))[0]
        if ref_matches.size == 0:
            print(f"[iter {it}] WARNING: ref ch {int(ch)} not in selected_channels this round; skipping REF_THR check.")
            ref_ok = np.ones(N_cur, dtype=bool)
        else:
            ref_row = int(ref_matches[0])
            ref_d   = HM[ref_row]          # [N_cur]
            ref_ok  = (ref_d < REF_THR)

        keep = (mean_d <= MEAN_THR) & (max_d <= CHAN_THR) & ref_ok
        n_bad = int((~keep).sum())
        print(f"[iter {it}] N={N_cur} | reject={n_bad} "
            f"(mean>{MEAN_THR}: {int((mean_d>MEAN_THR).sum())}, "
            f"any>{CHAN_THR}: {int((max_d>CHAN_THR).sum())}, "
            f"ref>={REF_THR}: {int((~ref_ok).sum())})")

        # --- Debug: which channels cause the max-Δ failures? ---
        if n_bad > 0:
            from collections import Counter
            sel_global = np.asarray(res_it["selected_channels"], dtype=int)   # rows → global channel ids
            bad_spikes = np.where(max_d > CHAN_THR)[0]                        # failing spike indices
            # argmax row for each failing spike (row is within selected_channels)
            row_of_max = np.argmax(HM[:, bad_spikes], axis=0)
            ch_of_max  = sel_global[row_of_max]                               # map to global channel ids
            val_of_max = HM[row_of_max, bad_spikes]

            # summary: top offending channels by count
            counts = Counter(ch_of_max.tolist()).most_common(15)
            print(f"[iter {it}] channels hitting max Δ>{CHAN_THR} (top 15):")
            for ch_id, cnt in counts:
                print(f"  ch {int(ch_id):4d}  →  {cnt} spikes")

            # first few concrete examples (spike index, channel, Δ value)
            head = min(10, bad_spikes.size)
            for s_idx, ch_id, val in zip(bad_spikes[:head], ch_of_max[:head], val_of_max[:head]):
                print(f"    spike {int(s_idx):5d}: ch {int(ch_id):4d}, Δ={float(val):.2f}")


        if n_bad == 0:
            # default: no split, use current survivors as final
            final_res = res_it
            final_ei_full = ei_full

            # One more bimodality check if we never looked (e.g., no rejections happened)
            if bimodal_payload is None:
                tmp = check_bimodality_and_plot(
                    snips_cur, res_it, ei_positions, ref_ch=int(ch),
                    dprime_thr=5.0, min_per_cluster=5
                )
                if tmp and tmp.get("hit", False):
                    bimodal_payload = tmp

            if bimodal_payload and bimodal_payload.get("hit", False):
                # expose cohort EIs for inspection
                ei_lo_bimodal = bimodal_payload["ei_lo"]
                ei_hi_bimodal = bimodal_payload["ei_hi"]
                idx_lo = bimodal_payload["idx_lo"]
                idx_hi = bimodal_payload["idx_hi"]

                # choose cohort by larger N
                # amp_lo = float(-ei_lo_bimodal[int(ch)].min())
                # amp_hi = float(-ei_hi_bimodal[int(ch)].min())
                pick_hi = (len(idx_hi) >= len(idx_lo))
                chosen_idx = idx_hi if pick_hi else idx_lo

                # --- guard against stale/out-of-range indices ---
                chosen_idx = np.asarray(chosen_idx, dtype=np.int64).ravel()
                Ncur = snips_cur.shape[2]
                bad = (chosen_idx < 0) | (chosen_idx >= Ncur)
                if bad.any():
                    print(f"[bimodality] WARNING: {bad.sum()} invalid indices (min={chosen_idx.min()}, "
                        f"max={chosen_idx.max()}, N={Ncur}). Filtering.")
                    chosen_idx = chosen_idx[~bad]

                if chosen_idx.size == 0:
                    print("[bimodality] cohort empty after filtering; skipping split.")
                    final_res = res_it
                    final_ei_full = ei_full
                else:
                    # restrict survivors to chosen cohort and recompute EI + harm-map
                    snips_fin = snips_cur[:, :, chosen_idx]
                    final_ei_full = median_ei_adaptive(snips_fin)


                final_res = compute_harm_map_noamp(
                    ei=final_ei_full, snips=snips_fin,
                    p2p_thr=50.0, max_channels=snips_fin.shape[0], min_channels=10,
                    lag_radius=LAG_RADIUS, weight_by_p2p=True, weight_beta=0.7,
                    force_include_main=True
                )

                # plots: new harm-map and the final EI
                plot_harm_heatmap(
                    final_res, field="harm_matrix",
                    title=f"Final harm map (post-bimodal split; picked {'high' if pick_hi else 'low'} amp cohort)"
                )
                try:
                    fig, ax = plt.subplots(figsize=(20, 12))
                    pew.plot_ei_waveforms(
                        final_ei_full, ei_positions,
                        ref_channel=int(ch), scale=70.0, box_height=1.0, box_width=50.0, ax=ax
                    )
                    ax.set_title(f"Final EI | ch {int(ch)} | cohort={'high' if pick_hi else 'low'}")
                    plt.show()
                except Exception as e:
                    print(f"[final EI] plotting skipped: {e}")

            print("[done] All events satisfy thresholds.")
            break



        # Kick failing spikes, recompute everything fresh next round
        snips_cur = snips_cur[:, :, keep]
        if snips_cur.shape[2] == 0:
            print(f"[iter {it}] all spikes rejected; stopping.")
            final_res = res_it
            final_ei_full = ei_full
            break

        print(f"[iter {it}] kept {snips_cur.shape[2]}/{N_cur} after harm-map pruning")

        # Bimodality check on the post-kick set (capture FIRST hit only)
        if bimodal_payload is None:
            # Recompute a quick harm-map on the trimmed set so channel selection is accurate
            res_tmp = compute_harm_map_noamp(
                ei=median_ei_adaptive(snips_cur),
                snips=snips_cur,
                p2p_thr=50.0, max_channels=snips_cur.shape[0], min_channels=10,
                lag_radius=LAG_RADIUS, weight_by_p2p=True, weight_beta=0.7,
                force_include_main=True
            )
            tmp = check_bimodality_and_plot(
                snips_cur, res_tmp, ei_positions, ref_ch=int(ch),
                dprime_thr=5.0, min_per_cluster=5
            )
            if tmp and tmp.get("hit", False):
                bimodal_payload = tmp




    else:
        # fell out by MAX_ITERS
        final_res = res_it
        final_ei_full = ei_full
        print("[stop] Reached MAX_ITERS; some violations may remain.")

    # =========================
    # Plots: final harm map & EI
    # =========================
    if final_res is not None:
        plot_harm_heatmap(final_res, field="harm_matrix", sort_by_ptp=False,
                        title=f"Final harm map after pruning "
                                f"(K={len(final_res['selected_channels'])}, N={final_res['harm_matrix'].shape[1]})")

    if final_ei_full is not None:
        # If snips_cur covered the full array, this is full; otherwise it covers available channels only.
        C_full = ei_positions.shape[0]
        L_ei   = final_ei_full.shape[1]
        ei_plot = (final_ei_full if final_ei_full.shape[0] == C_full
                else np.pad(final_ei_full, ((0, C_full - final_ei_full.shape[0]), (0, 0)), mode='constant'))

        fig, ax = plt.subplots(1, 1, figsize=(10,12))
        pew.plot_ei_waveforms(ei_plot, ei_positions, ref_channel=int(ch), ax=ax,
                            colors='C2', scale=70.0, box_height=1.0, box_width=50.0)
        ax.set_title(f"Final EI after pruning (events kept = {final_res['harm_matrix'].shape[1] if final_res else 'n/a'})")
        plt.tight_layout(); plt.show()

# try:
proto_ei = final_ei_full if 'final_ei_full' in locals() else ei_full
# Peak-to-peak per channel
p2p = proto_ei.max(axis=1) - proto_ei.min(axis=1)
main_ch = int(np.argmax(p2p))
if main_ch != ch:
    print(f"ABORT TEMPLATE: MAIN CHANNEL {main_ch} with amp {p2p[main_ch]:0.1f}, current CH {ch} with amp {p2p[ch]:0.1f}")




