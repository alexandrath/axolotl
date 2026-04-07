import numpy as np

def bin_spikes_matlab_style(spikes_sec, triggers_sec, refresh=2):
    """
    Spike binning to match MATLAB: all in seconds, double precision.

    Parameters:
        spikes_sec: spike times in seconds (1D numpy array)
        triggers_sec: trigger times in seconds (1D numpy array)
        refresh: monitor refresh interval per stimulus frame

    Returns:
        frame_indices: stimulus frame index for each spike (0-based, like MATLAB output minus 1)
    """
    spikes = spikes_sec.astype(np.float64)
    triggers = triggers_sec.astype(np.float64)

    # Align spikes relative to first trigger, same as: spikes{i} - triggers(1)
    spikes = spikes - triggers[0]
    spikes = spikes[spikes > 0]

    mean_trigger_interval = np.mean(np.diff(triggers))  # seconds per 100 monitor frames

    # Convert to stimulus frame index as floating point
    spike_frame_pos = spikes / mean_trigger_interval * (100 / refresh)

    # Digitize with 1-based edges, then shift to 0-based frame indices
    frame_indices = np.floor(spike_frame_pos - 0.45).astype(int)


    return frame_indices


def bin_spikes_matlab_style_robust_autorepair(
    spikes_sec: np.ndarray,
    triggers_sec: np.ndarray,
    refresh: int = 2,
    *,
    expected_frames_per_block: int = 100,   # 100 monitor frames per trigger block
    drop_before_first: bool = True,
    drop_after_last: bool = True,
    duplicate_tol: float = 0.20,            # intervals < 20% of median → bounce/duplicate
    bin_shift: float = 0.95,                # MATLAB-style binning: floor(x - 0.95)
    return_diagnostics: bool = False,
):
    """
    Robust spike→stimulus-frame mapping that matches your MATLAB binning and stops drift:

    - Use global MEDIAN(diff(triggers)) as nominal seconds per 100 monitor frames (frozen).
    - Collapse bounce/duplicate triggers (very short dt).
    - Insert synthetic triggers only when an interval implies 2×,3×… blocks (missed TTL).
    - For normal jitter (101/102 by duration), still count 100 monitor frames.
    - Place spikes with phase-accurate local period; bin with floor(x - 0.95).
    """
    spikes = np.asarray(spikes_sec, dtype=np.float64)
    trig_orig = np.asarray(triggers_sec, dtype=np.float64)
    assert trig_orig.ndim == 1 and trig_orig.size >= 2

    # Boundary mask (strict '>' at first trigger; '<' last trigger)
    m_spk = np.ones_like(spikes, dtype=bool)
    if drop_before_first:
        m_spk &= (spikes > trig_orig[0])
    if drop_after_last:
        m_spk &= (spikes < trig_orig[-1])
    spk = spikes[m_spk]
    if spk.size == 0:
        return (np.empty(0, np.int32),
                dict(spike_mask=m_spk) if return_diagnostics else np.empty(0, np.int32))

    # Collapse duplicate/bounce triggers (very short dt relative to median)
    def collapse_bounces(trigs, tol):
        dt = np.diff(trigs)
        med = np.median(dt)
        bad = np.where(dt < tol * med)[0]
        if bad.size == 0:
            return trigs, np.array([], dtype=int)
        keep = np.ones_like(trigs, dtype=bool)
        keep[bad + 1] = False  # drop the later trigger of each too-short interval
        return trigs[keep], np.where(~keep)[0]

    trig_clean, removed_dup_idx = collapse_bounces(trig_orig, duplicate_tol)

    # Freeze nominal seconds per 100 monitor frames from PRE-REPAIR median
    dt_clean = np.diff(trig_clean)
    sec_per_100 = float(np.median(dt_clean))                   # seconds per 100 monitor frames (frozen)

    # Detect & repair missed triggers (interval implies 2×,3×… blocks)
    kb_clean = np.rint(dt_clean / sec_per_100).astype(int)     # implied blocks per interval
    missed_idx = np.where(kb_clean >= 2)[0]

    ins_times = []
    if missed_idx.size:
        repaired = [trig_clean[0]]
        for i, dt in enumerate(dt_clean):
            blocks = max(1, int(np.rint(dt / sec_per_100)))
            if blocks >= 2:
                for k in range(1, blocks):
                    t_ins = trig_clean[i] + dt * (k / blocks)  # even split
                    repaired.append(t_ins)
                    ins_times.append(t_ins)
            repaired.append(trig_clean[i + 1])
        trig_used = np.array(repaired, dtype=np.float64)
    else:
        trig_used = trig_clean

    # Recompute intervals AFTER repair (do NOT change sec_per_100)
    dt_rep = np.diff(trig_used)

    # Blocks per repaired interval, rounded to integers relative to frozen sec_per_100
    kb_rep = np.rint(dt_rep / sec_per_100).astype(int)
    kb_rep[kb_rep < 1] = 1

    # TRUE monitor-frame count per interval:
    #   100 for good intervals; 200/300... ONLY when a trigger was missed.
    N_monitor = kb_rep * expected_frames_per_block  # 100, 200, 300, ...

    # Local seconds-per-monitor-frame for phase-accurate placement inside each interval
    tau_local = dt_rep / N_monitor  # sec per monitor frame, per interval

    # Cumulative monitor frames at trigger boundaries
    frames_at_trigger = np.concatenate(([0], np.cumsum(N_monitor)))  # length = len(trig_used)

    # Place spikes using local phase (no deficit fudge needed)
    idx = np.searchsorted(trig_used, spk, side='right') - 1
    idx = np.clip(idx, 0, N_monitor.size - 1)
    t0 = trig_used[idx]
    f_local = (spk - t0) / tau_local[idx]
    # keep strictly inside [0, N_i)
    eps = 1e-9
    f_local = np.clip(f_local, 0.0, N_monitor[idx] - eps)

    # Continuous stim-frame coordinate and MATLAB-style binning
    M_total = frames_at_trigger[idx] + f_local           # global monitor-frame coordinate
    X = M_total / float(refresh)                         # continuous stim-frame coordinate
    frame_idx = np.floor(X - bin_shift).astype(np.int32) # floor(x - 0.95)

    if not return_diagnostics:
        return frame_idx

    # Optional diagnostics
    extra_mon = N_monitor - expected_frames_per_block
    cum_extra_mon = np.concatenate(([0], np.cumsum(extra_mon)))
    # integer stim-frame deficit (for reference only; not used in placement)
    deficit = np.floor_divide(cum_extra_mon, refresh).astype(int)
    per_int_drop_stim = np.diff(deficit).astype(int)

    # Where those deficits would step on the stim axis (reference)
    stim_actual = np.concatenate(([0.0], np.cumsum(N_monitor))) / float(refresh)
    dropped_stim_idx = []
    run = 0
    for i, k in enumerate(per_int_drop_stim):
        if k > 0:
            start = int(np.floor(stim_actual[i + 1]))
            dropped_stim_idx.extend(range(start + run, start + run + k))
            run += k

    diag = dict(
        spike_mask=m_spk,
        triggers_original=trig_orig,
        triggers_clean_no_dups=trig_clean,
        triggers_used=trig_used,
        removed_trigger_indices=removed_dup_idx,
        missed_trigger_intervals=missed_idx,
        inserted_trigger_times=np.asarray(ins_times, dtype=np.float64),
        sec_per_100=sec_per_100,
        dt_repaired=dt_rep,
        kb_repaired=kb_rep,
        N_monitor=N_monitor,
        tau_local=tau_local,
        frames_at_trigger=frames_at_trigger,
        refresh=refresh,
        bin_shift=bin_shift,
        extra_monitor_frames=extra_mon,
        cum_extra_monitor_frames=cum_extra_mon,
        deficit_at_triggers=deficit,
        dropped_stim_frame_indices=np.asarray(dropped_stim_idx, dtype=int),
    )
    return frame_idx, diag
