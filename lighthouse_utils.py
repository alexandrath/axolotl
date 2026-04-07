# -*- coding: utf-8 -*-
"""
lighthouse_utils.py
-------------------
Adaptive lighthouse discovery on a *single channel* from raw data.

Pipeline per channel:
  1) Detect local minima (no refractory).
  2) Negative-side histogram -> find "valley" window by an adaptive count-based rule:
       left_count >= required_ratio(valley_count) * valley_count
       and left_count >= min_valid_count
     where required_ratio(V) = base_ratio + floor(V / ratio_step), clipped to [min,max].
  3) Partition by amplitude at the channel’s minima:
       LEFT  : vals < valley_low_edge
       RIGHT : vals >= valley_low_edge      (includes valley and beyond)
  4) Build EI_left_high / EI_left_low from k_left_low/k_left_high left spikes
     (optionally downsample left if very large).
     *Abort early* unless cosine(EI_left_low, EI_left_high) >= base_cos_min.
  5) Sweep RIGHT in *consecutive* windows of size Nwin, stepping by step_size (e.g., 100, step 50).
     For each window, build EI_right_win and compute cosine(EI_left_low, EI_right_win).
     Stop when *two consecutive windows* fall below a *relative* threshold:
        cos_rwin < max(abs_cos_min, base_cos - cos_slack)
     Exclude the first failing window from acceptance.
  6) Return accepted spikes (LEFT + accepted RIGHT windows), frontier metadata, and diagnostics.

Notes:
  * All snippets extracted on ALL channels (C x L x N).
  * Reuses your existing utilities for snippet extraction and EI comparison.
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


from axolotl_utils_ram import extract_snippets_fast_ram
from joint_utils import cosine_two_eis   
from collision_utils import median_ei_adaptive   
import plot_ei_waveforms as pew
# Plain functions (no classes), with fixed imports exactly as requested.
# Drop-in utilities to (1) find a negative-amplitude “valley”, (2) split events,
# (3) build left-high/left-low EIs from VALID spikes only, (4) filter valley spikes
# in batches against the left-low EI, and (5) finalize a big EI from all left + accepted
# valley spikes with an optional cap on how many snippets to extract.
#
# Usage example is at the bottom.


# ------------------------
# 1) FIND VALLEY & EVENTS
# ------------------------
def find_valley_and_times(
    raw_data,                # (T_total, C), int16 or float
    ch,                      # int, channel index to analyze
    *,
    window=(-40, 80),        # (pre, post) samples; used to ensure event windows fit inside [start, stop)
    start=0,                 # inclusive start sample for analysis
    stop=None,               # exclusive stop sample for analysis; None → end of recording
    bin_width=10.0,          # ADC units per histogram bin
    valley_bins=5,           # number of consecutive bins forming the “valley” (e.g. 5 → 50 ADC)
    min_valid_count=300,     # require Left ≥ this many spikes
    ratio_base=3,            # base multiplier for required Left/Valley ratio (e.g. 3)
    ratio_step=100,          # each extra 100 valley spikes raises the ratio by +1
    ratio_floor=2,           # never require less than 2×
    ratio_cap=10,            # **cap** required ratio at 10×
    right_k=500,             # return ≥ this many “right-of-left” spikes (valley or valley+fill)
    min_trough=None          # optional signed floor on minima; exclude vals < min_trough (e.g. -1200)
):
    """
    Scan all negative-amplitude bins (fixed width), compute 5-bin sums, pick the **leftmost local peak**
    (deep cluster center), then the **lowest valley to its right**. Accept if:

        Left_count ≥ required_ratio(Valley_count) * Valley_count  AND  Left_count ≥ min_valid_count,

    where required_ratio(V) = min(ratio_cap, max(ratio_floor, ratio_base + floor(V/ratio_step))).

    Returns:
      {
        'accepted': bool,
        'valley_low': float, 'valley_high': float,
        'left_times':   int64[NL], 'left_vals':   float[NL],        # chronological
        'valley_times': int64[NV], 'valley_vals': float[NV],        # chronological (just the valley band)
        'left_count': int, 'valley_count': int, 'required_ratio': float,
        'analysis_span': (start, stop),

        # NEW: “right-of-left” selection for downstream windows:
        #  — if valley_count ≥ right_k: return the ENTIRE valley (≥ right_k)
        #  — else return VALLEY + the most-negative spikes to the right of the valley (>= valley_high)
        #     until total reaches right_k
        'rightk_times_by_amp':  int64[NR],  # amplitude-sorted (most negative first)
        'rightk_vals_by_amp':   float[NR],  # aligned values
        'rightk_times_sorted':  int64[NR],  # chronological view of the same selection
      }
    """

    def _empty_step1(_start, _stop):
        return dict(
            accepted=False,
            valley_low=None, valley_high=None,
            left_times=np.array([], dtype=np.int64),
            left_vals=np.array([], dtype=np.float32),
            valley_times=np.array([], dtype=np.int64),
            valley_vals=np.array([], dtype=np.float32),
            left_count=0, valley_count=0, required_ratio=None,
            analysis_span=(_start, _stop),
            rightk_times_by_amp=np.array([], dtype=np.int64),
            rightk_vals_by_amp=np.array([], dtype=np.float32),
            rightk_times_sorted=np.array([], dtype=np.int64),
            amp_hist_counts=np.array([], dtype=np.int32),
            amp_hist_edges=np.array([], dtype=np.float32),
            all_times=np.array([], dtype=np.int64),
            all_vals=np.array([], dtype=np.float32),
        )

    T, C = raw_data.shape
    if stop is None:
        stop = T
    if ch < 0 or ch >= C:
        raise ValueError(f"Channel {ch} out of range 0..{C-1}")

    pre, post = int(window[0]), int(window[1])
    if pre >= 0 or post <= 0:
        raise ValueError("`window` must straddle the event (use negative pre, positive post)")

    # ---- 1) Local minima (no refractory), filtered to fit the EI window in [start, stop)
    x = raw_data[start:stop, ch].astype(np.float32, copy=False)
    if x.size < 3:
        return _empty_step1(start, stop)

    msk = (x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:])
    idx_local = np.where(msk)[0] + 1
    if idx_local.size == 0:
        return _empty_step1(start, stop)

    times_abs = start + idx_local.astype(np.int64)
    vals = x[idx_local].astype(np.float32)

    ok = (times_abs + pre >= start) & (times_abs + post < stop)
    times_abs = times_abs[ok]
    vals      = vals[ok]

    # Optional artifact floor on signed minima:
    # keep only minima that are not "too deep", e.g. min_trough=-1200 excludes vals < -1200
    if min_trough is not None:
        keep_amp = vals >= float(min_trough)
        times_abs = times_abs[keep_amp]
        vals      = vals[keep_amp]

    if times_abs.size == 0:
        return _empty_step1(start, stop)
    
    # Keep all detected minima that survived window + amplitude filtering
    all_times = times_abs.astype(np.int64, copy=False)
    all_vals  = vals.astype(np.float32, copy=False)

    # ---- 2) Negative-side histogram up to 0
    vmin = float(np.min(vals))
    if vmin >= 0:
        return _empty_step1(start, stop)

    bw = float(bin_width)
    low_edge  = bw * np.floor(vmin / bw)
    high_edge = 0.0
    edges = np.linspace(low_edge, high_edge, int(round((high_edge - low_edge) / bw)) + 1, dtype=np.float64)
    if edges.size < valley_bins + 1:
        low_edge = low_edge - bw * ((valley_bins + 1) - edges.size + 1)
        edges = np.linspace(low_edge, high_edge, int(round((high_edge - low_edge) / bw)) + 1, dtype=np.float64)

    counts, edges = np.histogram(vals, bins=edges)
    nb = counts.size
    if nb < valley_bins + 1:
        return _empty_step1(start, stop)


    # ---- 3) Sliding 5-bin sums over fully negative windows
    kernel = np.ones(int(valley_bins), dtype=np.int64)
    S = np.convolve(counts, kernel, mode='valid')         # length = nb - valley_bins + 1
    win_low_edges  = edges[:-valley_bins]
    win_high_edges = edges[valley_bins:]
    neg_mask = (win_high_edges <= 0.0)
    if not np.any(neg_mask):
        return _empty_step1(start, stop)


    Sn    = S[neg_mask]
    lows  = win_low_edges[neg_mask]
    highs = win_high_edges[neg_mask]
    m     = Sn.size

    # ---- 4) Right-anchored noise peak, then valley to its LEFT (robust split)
    # ---- 4) Peaks & valleys of Sn (over negative-only windows)
    loc_peak = np.zeros(m, dtype=bool)
    if m >= 3:
        loc_peak[1:-1] = (Sn[1:-1] > Sn[:-2]) & (Sn[1:-1] >= Sn[2:])
        if Sn[0] > Sn[1]:
            loc_peak[0] = True
        if Sn[-1] > Sn[-2]:
            loc_peak[-1] = True
    else:
        loc_peak[np.argmax(Sn)] = True
    peak_idxs = np.where(loc_peak)[0]

    loc_valley = np.zeros(m, dtype=bool)
    if m >= 3:
        loc_valley[1:-1] = (Sn[1:-1] < Sn[:-2]) & (Sn[1:-1] <= Sn[2:])
        if Sn[0] < Sn[1]:
            loc_valley[0] = True
        if Sn[-1] < Sn[-2]:
            loc_valley[-1] = True
    else:
        loc_valley[np.argmin(Sn)] = True
    valley_idxs = np.where(loc_valley)[0]
    if valley_idxs.size == 0:
        return _empty_step1(start, stop)


    # ---- 5) Choose the LEFTMOST valley that still has a sizeable left bump
    # "Sizeable" = the mass of events strictly left of the valley_low is >= min_valid_count.
    # We compute this cheaply from the histogram prefix-sum (no per-event scan).
    prefix = np.concatenate(([0], np.cumsum(counts)))  # length nb+1, prefix[k]=sum(counts[:k])

    chosen = None
    for j in valley_idxs:  # iterate from leftmost to right
        vlow = float(lows[j])   # this equals an entry in `edges`
        # index in `counts` that corresponds to bins strictly below vlow
        k_idx = int(np.searchsorted(edges, vlow, side='left'))
        left_est = int(prefix[k_idx])          # estimated #events with vals < vlow
        # require a left peak to exist (nearest one to the left in Sn-space)
        has_left_peak = np.any(peak_idxs < j)
        if has_left_peak and left_est >= int(min_valid_count):
            chosen = j
            break

    if chosen is None:
        # Fallback: keep previous behavior (pick nearest valley left of the right-anchored noise peak)
        # Identify a right-anchored dominant peak and take the nearest valley to its left.
        last_frac = 0.35
        right_bound = max(0, int(np.floor((1.0 - last_frac) * m)))
        right_candidates = peak_idxs[peak_idxs >= right_bound]
        if right_candidates.size == 0:
            right_peak = int(peak_idxs[np.argmax(Sn[peak_idxs])])
        else:
            right_peak = int(right_candidates[np.argmax(Sn[right_candidates])])
        # nearest valley strictly to the left
        cand = valley_idxs[valley_idxs < right_peak]
        if cand.size == 0:
            return _empty_step1(start, stop)

        valley_rel = int(cand[-1])  # nearest to the right peak
    else:
        valley_rel = int(chosen)

    valley_low  = float(lows[valley_rel])
    valley_high = float(highs[valley_rel])



    # ---- 6) Event-level counts using actual minima
    left_ev_mask     = (vals < valley_low)
    valley_ev_mask   = (vals >= valley_low) & (vals < valley_high)
    right_of_val_mask= (vals >= valley_high)   # strictly to the right of the valley band

    left_times  = times_abs[left_ev_mask]
    left_vals_  = vals[left_ev_mask]
    valley_times_ = times_abs[valley_ev_mask]
    valley_vals_  = vals[valley_ev_mask]
    right_times_  = times_abs[right_of_val_mask]
    right_vals_   = vals[right_of_val_mask]

    left_count   = int(left_ev_mask.sum())
    valley_count = int(valley_ev_mask.sum())

    req = min(ratio_cap, max(ratio_floor, ratio_base + (valley_count // int(max(1, ratio_step)))))
    accepted = (left_count >= max(min_valid_count, req * max(1, valley_count)))

    # Sort outputs chronologically
    li = np.argsort(left_times);    vi = np.argsort(valley_times_)
    left_times  = left_times[li].astype(np.int64, copy=False)
    left_vals_  = left_vals_[li].astype(np.float32, copy=False)
    valley_times_ = valley_times_[vi].astype(np.int64, copy=False)
    valley_vals_  = valley_vals_[vi].astype(np.float32, copy=False)

    # ---- 7) Build the “top-right” selection for BL/TR support:
    # exactly right_k strongest spikes on the RIGHT side (vals >= valley_low)
    right_ev_mask_full = (vals >= valley_low)
    right_times_full = times_abs[right_ev_mask_full]
    right_vals_full  = vals[right_ev_mask_full]

    r_full_order = np.argsort(right_vals_full)   # ascending => most negative first
    right_times_amp = right_times_full[r_full_order]
    right_vals_amp  = right_vals_full[r_full_order]

    keep_n = min(int(right_k), right_times_amp.size)
    rightk_times_by_amp = right_times_amp[:keep_n].astype(np.int64, copy=False)
    rightk_vals_by_amp  = right_vals_amp[:keep_n].astype(np.float32, copy=False)

    rk_sort = np.argsort(rightk_times_by_amp)
    rightk_times_sorted = rightk_times_by_amp[rk_sort]

    return dict(
        accepted=bool(accepted),
        valley_low=valley_low, valley_high=valley_high,
        left_times=left_times, left_vals=left_vals_,
        valley_times=valley_times_, valley_vals=valley_vals_,
        left_count=int(left_count), valley_count=int(valley_count), required_ratio=float(req),
        analysis_span=(start, stop),
        rightk_times_by_amp=rightk_times_by_amp,
        rightk_vals_by_amp=rightk_vals_by_amp,
        rightk_times_sorted=rightk_times_sorted,
        amp_hist_counts=counts.astype(np.int32, copy=False),
        amp_hist_edges=edges.astype(np.float32, copy=False),
        all_times=all_times,
        all_vals=all_vals,
    )




def build_left_low_high_eis(
    raw_data: np.ndarray,
    left_times: np.ndarray,      # int64, chronological, already filtered to [start,stop] with window safety
    left_vals: np.ndarray,       # float, same length/order as left_times (negative = deeper trough)
    *,
    window: tuple = (-40, 80),
    left_cap: int = 500,         # max per group; we take k = min(left_cap, N_left//3) for LOW/MID/HIGH
    reducer: str = 'median',     # 'median' (via median_ei_adaptive) or 'mean'
    rms_gate: float = 10.0,      # pass same gate everywhere for stable nch_used
    best_align_lag: int = 3,
    use_abs: bool = True
):
    """
    Build LOW/MID/HIGH EIs from LEFT spikes without extracting all snippets.

    Steps
      1) Determine k = min(left_cap, N_left//3).
      2) Sort LEFT by amplitude (more negative = deeper).
         - HIGH: k most-negative
         - LOW:  k least-negative
         - MID:  k around the amplitude median (central k from the remaining middle band)
      3) Extract only those 3*k snippets (all channels).
      4) Build EIs for LOW/MID/HIGH with chosen reducer.
      5) Compute pairwise cosines with consistent rms_gate / best_align_lag.

    Returns dict:
      {
        'ei_left_low':   (C,L),
        'ei_left_mid':   (C,L),
        'ei_left_high':  (C,L),

        'base_cosine':   float,   # cos(low, high)  (kept for backward compatibility)
        'cos_low_mid':   float,   # cos(low, mid)
        'cos_mid_high':  float,   # cos(mid, high)

        'base_nch':      int,
        'base_lag':      int,
        'base_olen':     int,

        'k_left':        int,     # k actually used per group
        'left_low_times':   int64[k_used],   # valid times used in LOW EI
        'left_mid_times':   int64[k_used],   # valid times used in MID EI
        'left_high_times':  int64[k_used],   # valid times used in HIGH EI
      }
    """
    assert left_times.shape[0] == left_vals.shape[0], "left_times/left_vals length mismatch"
    N_left = int(left_times.shape[0])
    if N_left < 3:
        raise RuntimeError("Not enough LEFT spikes to build low/mid/high EIs")

    # Equal counts for LOW/MID/HIGH
    k = min(int(left_cap), N_left // 3)
    if k == 0:
        raise RuntimeError("k=0 after min(left_cap, N_left//3); not enough LEFT spikes")

    # Amplitude ordering once (no extraction yet)
    # left_vals: negative numbers ⇒ more negative means deeper (larger |amplitude|)
    order = np.argsort(left_vals)         # ascending: most negative (deepest) first

    idx_high = order[:k]                  # deepest k
    idx_low  = order[-k:]                 # least-negative k (closest to valley)

    # Remaining middle band and central-k selection for MID
    mid_band = order[k: N_left - k]
    mid_len = int(mid_band.size)
    if mid_len < k:
        # This shouldn't happen with k = floor(N/3), but guard anyway.
        # Fall back to taking the central portion of the whole array.
        center = N_left // 2
        half = k // 2
        start = max(0, center - half)
        end = min(N_left, start + k)
        idx_mid = order[start:end]
        if idx_mid.size < k:
            # Ensure size k by padding from neighbors without overlap
            need = k - idx_mid.size
            # take from left side just before 'start' if available, else from right after 'end'
            take_left = min(need, max(0, start - k))
            pad_left = order[start - take_left:start]
            need -= pad_left.size
            pad_right = order[end:end + need] if need > 0 else np.array([], dtype=order.dtype)
            idx_mid = np.concatenate([pad_left, idx_mid, pad_right])
    else:
        # take central k from the middle band
        start_rel = (mid_len - k) // 2
        idx_mid = mid_band[start_rel:start_rel + k]

    # Times for each group (absolute sample indices)
    times_high = left_times[idx_high].astype(np.int64, copy=False)
    times_low  = left_times[idx_low].astype(np.int64,  copy=False)
    times_mid  = left_times[idx_mid].astype(np.int64,  copy=False)

    # Extract only those groups (all channels)
    C = raw_data.shape[1]
    all_ch = np.arange(C, dtype=np.int32)

    sn_high, t_high_valid = extract_snippets_fast_ram(
        raw_data=raw_data, spike_times=times_high, window=window, selected_channels=all_ch
    )
    sn_low,  t_low_valid  = extract_snippets_fast_ram(
        raw_data=raw_data, spike_times=times_low,  window=window, selected_channels=all_ch
    )
    sn_mid,  t_mid_valid  = extract_snippets_fast_ram(
        raw_data=raw_data, spike_times=times_mid,  window=window, selected_channels=all_ch
    )

    sn_high = sn_high.astype(np.float32, copy=False)
    sn_low  = sn_low.astype(np.float32,  copy=False)
    sn_mid  = sn_mid.astype(np.float32,  copy=False)

    if sn_high.shape[2] == 0 or sn_low.shape[2] == 0 or sn_mid.shape[2] == 0:
        raise RuntimeError("No valid snippets in one of LOW/MID/HIGH groups (check window/edges).")

    # Build EIs
    if reducer == 'median':
        ei_high = median_ei_adaptive(sn_high)
        ei_low  = median_ei_adaptive(sn_low)
        ei_mid  = median_ei_adaptive(sn_mid)
    else:
        ei_high = sn_high.mean(axis=2)
        ei_low  = sn_low.mean(axis=2)
        ei_mid  = sn_mid.mean(axis=2)

    # Pairwise cosine similarities with consistent gating/lag
    base_cos, base_nch, base_lag, base_olen = cosine_two_eis(
        ei_low, ei_high, rms_gate=rms_gate, use_abs=use_abs, best_align_lag=best_align_lag
    )
    cos_low_mid, _, _, _ = cosine_two_eis(
        ei_low, ei_mid, rms_gate=rms_gate, use_abs=use_abs, best_align_lag=best_align_lag
    )
    cos_mid_high, _, _, _ = cosine_two_eis(
        ei_mid, ei_high, rms_gate=rms_gate, use_abs=use_abs, best_align_lag=best_align_lag
    )

    # Return valid times actually used (extractor may drop edge-near requests)
    return {
        'ei_left_low':  ei_low,
        'ei_left_mid':  ei_mid,
        'ei_left_high': ei_high,

        'base_cosine':  float(base_cos),      # cos(low, high)
        'cos_low_mid':  float(cos_low_mid),   # cos(low, mid)
        'cos_mid_high': float(cos_mid_high),  # cos(mid, high)

        'base_nch':     int(base_nch),
        'base_lag':     int(base_lag),
        'base_olen':    int(base_olen),

        'k_left':       int(k),
        'left_low_times':  t_low_valid.astype(np.int64,  copy=False),
        'left_mid_times':  t_mid_valid.astype(np.int64,  copy=False),
        'left_high_times': t_high_valid.astype(np.int64, copy=False),
    }


# -------------------------------------------------
# 3) FILTER VALLEY SPIKES IN BATCHES vs LEFT-LOW EI
# -------------------------------------------------
def filter_valley_spikes_by_batches(
    raw_data: np.ndarray,
    rightk_times_by_amp: np.ndarray,  # int64[NR], amplitude-sorted (most negative first) from find_valley_and_times
    left_times: np.ndarray,           # int64, chronological LEFT times (already window-safe)
    left_vals:  np.ndarray,           # float, same length/order as left_times (negative = deeper)
    *,
    window: tuple = (-40, 80),
    batch_size: int = 30,             # 20–30 works well
    reducer: str = 'median',          # 'median' or 'mean' for EI building
    tail_n: int = 100,                # size of each LEFT tail group to define baseline (shallowest 100 + next 100)
    abs_cos_min: float = 0.85,        # hard floor for acceptable cosine
    base_cos_cap: float = 0.97,       # cap baseline so floor isn't too strict
    cos_slack: float = 0.05,          # dyn_floor = max(abs_cos_min, min(base_tail_cos, cap) - cos_slack)
    rms_gate: float = 10.0,           # use same gate everywhere to stabilize nch_used
    best_align_lag: int = 3,
    use_abs: bool = True,
):
    """
    Batch-filter the 'right-of-left' selection (valley plus next-right spikes) using a dynamic cosine
    threshold derived from *LEFT-tail* stability:

      base_tail_cos = cos( EI(shallowest tail_n LEFT), EI(next-shallowest tail_n LEFT) )

    Each right-side batch (consecutive 'batch_size' spikes in amplitude order) is accepted iff:

      cos( EI(shallowest LEFT tail), EI(batch) ) >= dyn_floor

    with dyn_floor = max(abs_cos_min, min(base_tail_cos, base_cos_cap) - cos_slack).

    Only the 2*tail_n shallowest LEFT spikes are extracted to build the baseline tails.
    """
    T, C = raw_data.shape
    pre, post = window
    all_ch = np.arange(C, dtype=np.int32)

    # -------------------------
    # 1) Build LEFT-tail baseline WITHOUT bulk extraction
    # -------------------------
    assert left_times.shape[0] == left_vals.shape[0], "left_times/left_vals length mismatch"
    N_left = left_times.shape[0]
    if N_left < 2 * max(1, tail_n):
        # Not enough LEFT to build two tails
        return {
            "accepted_right_times": np.array([], dtype=np.int64),
            "batch_cosines": [],
            "base_tail_cosine": np.nan,
            "dyn_floor": np.nan,
            "ref_ei": None,
        }

    # Shallowest (least-negative) first: sort descending by value (less negative → larger value)
    ord_desc = np.argsort(left_vals)[::-1]
    pool_idx = ord_desc[: 2 * tail_n]
    pool_times_amp = left_times[pool_idx].astype(np.int64, copy=False)  # amplitude-ordered selection

    # Extract ONLY those 2*tail_n snippets (all channels)
    sn_pool, t_pool_valid = extract_snippets_fast_ram(
        raw_data=raw_data,  # (T,C)
        spike_times=pool_times_amp,
        window=window,
        selected_channels=all_ch,
    )
    sn_pool = sn_pool.astype(np.float32, copy=False)
    t_pool_valid = t_pool_valid.astype(np.int64, copy=False)
    if sn_pool.shape[2] == 0:
        return {
            "accepted_right_times": np.array([], dtype=np.int64),
            "batch_cosines": [],
            "base_tail_cosine": np.nan,
            "dyn_floor": np.nan,
            "ref_ei": None,
        }

    # Re-establish amplitude order among *valid* tail pool:
    # Map absolute time -> amplitude rank in 'pool_times_amp'
    rank_map = {int(t): r for r, t in enumerate(pool_times_amp.tolist())}
    ranks = np.array([rank_map.get(int(t), -1) for t in t_pool_valid], dtype=int)
    keep = ranks >= 0
    if not np.any(keep):
        return {
            "accepted_right_times": np.array([], dtype=np.int64),
            "batch_cosines": [],
            "base_tail_cosine": np.nan,
            "dyn_floor": np.nan,
            "ref_ei": None,
        }
    t_pool_valid = t_pool_valid[keep]
    sn_pool = sn_pool[:, :, keep]
    ranks = ranks[keep]

    # Sort valid pool by amplitude (ascending depth → shallowest first)
    order = np.array(np.argsort(ranks), dtype=int)  # shallowest (least negative) first
    t_pool_valid = t_pool_valid[order]
    sn_pool      = sn_pool[:, :, order]

    # Split into two adjacent shallow tails; adapt k if some were dropped at edges
    k_avail = sn_pool.shape[2] // 2
    k1 = min(tail_n, k_avail)
    if k1 == 0:
        return {
            "accepted_right_times": np.array([], dtype=np.int64),
            "batch_cosines": [],
            "base_tail_cosine": np.nan,
            "dyn_floor": np.nan,
            "ref_ei": None,
        }
    sn_tail1 = sn_pool[:, :, :k1]          # shallowest k1
    sn_tail2 = sn_pool[:, :, k1 : 2 * k1]  # next-shallowest k1
    t_tail1  = t_pool_valid[:k1]
    t_tail2  = t_pool_valid[k1 : 2 * k1]

    # Build tail EIs
    if reducer == "median":
        ei_tail1 = median_ei_adaptive(sn_tail1)
        ei_tail2 = median_ei_adaptive(sn_tail2)
    else:
        ei_tail1 = sn_tail1.mean(axis=2)
        ei_tail2 = sn_tail2.mean(axis=2)

    # Baseline cosine at the valley boundary (use same gating everywhere)
    base_tail_cos, _, _, _ = cosine_two_eis(
        ei_tail1, ei_tail2, rms_gate=rms_gate, use_abs=use_abs, best_align_lag=best_align_lag
    )
    dyn_floor = max(float(abs_cos_min), min(float(base_tail_cos), float(base_cos_cap)) - float(cos_slack))

    # -------------------------
    # 2) Batch over right-of-valley pool (amplitude-sorted), accept until first failure
    # -------------------------
    r_times = np.asarray(rightk_times_by_amp, dtype=np.int64)
    NR = r_times.size
    accepted_right = []
    batch_cosines = []
    i = 0
    while i < NR:
        j = min(i + batch_size, NR)
        win_times = r_times[i:j]
        if win_times.size == 0:
            break

        sn_win, t_win_valid = extract_snippets_fast_ram(
            raw_data=raw_data, spike_times=win_times, window=window, selected_channels=all_ch
        )
        sn_win = sn_win.astype(np.float32, copy=False)
        if sn_win.size == 0 or sn_win.shape[2] == 0:
            break  # treat as failure at boundary

        if reducer == "median":
            ei_win = median_ei_adaptive(sn_win)
        else:
            ei_win = sn_win.mean(axis=2)

        c, _, _, _ = cosine_two_eis(
            ei_tail1, ei_win, rms_gate=rms_gate, use_abs=use_abs, best_align_lag=best_align_lag
        )
        batch_cosines.append(float(c))

        if c >= dyn_floor:
            accepted_right.append(t_win_valid.astype(np.int64, copy=False))
            i = j
        else:
            break  # first failing batch is excluded; stop

    acc = np.sort(np.concatenate(accepted_right)) if accepted_right else np.array([], dtype=np.int64)

    return {
        "accepted_right_times": acc,
        "batch_cosines": batch_cosines,
        "base_tail_cosine": float(base_tail_cos),
        "dyn_floor": float(dyn_floor),
        "ref_ei": ei_tail1,  # EI of shallowest LEFT tail (the reference used for thresholding)
        "left_tail1_times": t_tail1,
        "left_tail2_times": t_tail2,
    }



# ---------------------
# 4) FINALIZE BIGGER EI
# ---------------------

# ---------------------
# 4) FINALIZE BIGGER EI
# ---------------------
def finalize_ei_with_cap(
    raw_data,
    left_times,                 # all *left* times (from find_valley_and_times)
    accepted_valley_times,      # valley times accepted by filter_valley_spikes_by_batches
    *,
    window=(-40, 80),
    final_max_spikes=None,      # e.g. 10_000 to cap extraction cost; None => use all available
    reducer='median',           # 'median' or 'mean'
    rng_seed=123,               # used only if we need to subsample
    ref_ch=None                 # <<< NEW: detection/LH reference channel to compute per-spike amplitudes on
):
    """
    Build the final *all-channel* EI from ALL left spikes + accepted valley spikes.
    Per-spike amplitudes are computed as the **signed trough (negative)** on the *reference* channel `ref_ch`.

    Returns:
      {
        'final_times': int64[N_final],          # union of left + accepted_right (sorted)
        'final_times_used': int64[N_used],      # times actually extracted for EI (capped if needed)
        'final_snips': (C, L, N_used) float32,  # (large) snippets used for EI; you can drop before saving
        'final_ei': (C, L) float32,             # all-channel EI
        'final_amplitudes': float[N_final],     # <<< signed troughs on ref_ch (e.g., -400)
        'main_ch': int,                         # channel of most-negative trough in the EI
      }
    """
    import numpy as np
    T, C = raw_data.shape
    all_ch = np.arange(C, dtype=np.int32)

    # Union of all spikes to include in final EI
    left_times = np.asarray(left_times, dtype=np.int64)
    accepted_valley_times = np.asarray(accepted_valley_times, dtype=np.int64)
    final_times = np.unique(np.concatenate([left_times, accepted_valley_times]))
    N_total = final_times.size
    if N_total == 0:
        raise RuntimeError("No spikes to finalize.")

    # Optional subsampling to cap extraction work
    if (final_max_spikes is not None) and (N_total > int(final_max_spikes)):
        rng = np.random.default_rng(rng_seed)
        pick = rng.choice(N_total, size=int(final_max_spikes), replace=False)
        times_used = final_times[np.sort(pick)]
    else:
        times_used = final_times

    # Extract all-channel snippets for the sampled set (to build EI)
    sn_final, times_used_valid = extract_snippets_fast_ram(
        raw_data=raw_data,
        spike_times=times_used,
        window=window,
        selected_channels=all_ch
    )
    sn_final = sn_final.astype(np.float32, copy=False)
    times_used_valid = times_used_valid.astype(np.int64, copy=False)

    if sn_final.shape[2] == 0:
        raise RuntimeError("No valid final snippets extracted (check start/stop vs window).")

    # Build the final EI
    if reducer == 'median':
        final_ei = median_ei_adaptive(sn_final)
    else:
        final_ei = sn_final.mean(axis=2)

    # Determine 'main' channel from EI (for reference) and validate ref_ch
    main_ch = int(np.argmin(final_ei.min(axis=1)))
    if ref_ch is None:
        raise ValueError("finalize_ei_with_cap: ref_ch must be provided (the lighthouse detection channel).")
    ref_ch = int(ref_ch)
    if not (0 <= ref_ch < C):
        raise ValueError(f"finalize_ei_with_cap: ref_ch={ref_ch} out of range 0..{C-1}")

    # Use trough index from the *reference channel* in the EI to align amplitudes
    trough_idx_ref = int(np.argmin(final_ei[ref_ch]))
    lo = max(0, trough_idx_ref - 1)
    hi = min(final_ei.shape[1] - 1, trough_idx_ref + 1)

    # Compute signed trough per spike on the reference channel (cheap single-channel extract)
    sn_ref, times_final_valid = extract_snippets_fast_ram(
        raw_data=raw_data,
        spike_times=final_times,
        window=window,
        selected_channels=np.array([ref_ch], dtype=np.int32)
    )
    sn_ref = sn_ref.astype(np.float32, copy=False)
    times_final_valid = times_final_valid.astype(np.int64, copy=False)

    if sn_ref.shape[2] > 0:
        # signed trough (negative), with 1-sample neighborhood robustness
        amps_valid_signed = sn_ref[0, lo:hi + 1, :].min(axis=0)
        amp_map = {int(t): float(a) for t, a in zip(times_final_valid, amps_valid_signed)}
        final_amplitudes = np.array([amp_map.get(int(t), np.nan) for t in final_times], dtype=np.float32)
    else:
        final_amplitudes = np.full(final_times.shape, np.nan, dtype=np.float32)

    return dict(
        final_times=final_times,
        final_times_used=times_used_valid,     # actual extracted (≤ final_times if capped)
        final_snips=sn_final,                  # NOTE: huge; drop before saving if undesired
        final_ei=final_ei,
        final_amplitudes=final_amplitudes,     # <<< signed troughs on ref_ch (negative)
        main_ch=main_ch
    )


import numpy as np
import matplotlib.pyplot as plt

def compute_prev_metrics(
    spike_times,           # 1D array of spike times in *samples* (absolute or relative), any order
    *,
    sample_rate_hz,        # e.g. 20000
    window_ms=50.0         # length of preceding window for counting (ms)
):
    """
    For each spike, compute:
      • isi_prev_ms: time to previous spike (ms), NaN for the first in time
      • prev_count : number of spikes in the preceding window of length `window_ms` (exclude current spike)

    Returns a dict with arrays aligned to the *input* order:
      {
        'isi_prev_ms': float[N],
        'prev_count' : int[N],
        'sorted_times': int64[N],      # for reference
        'order'      : int[N],         # indices that sort the input times
        'inv_order'  : int[N]          # inverse permutation (to map sorted->original)
      }

    Complexity: O(N) via a two-pointer sweep; robust for large N.
    """
    t = np.asarray(spike_times, dtype=np.int64)
    if t.ndim != 1:
        raise ValueError("spike_times must be a 1D array")

    N = t.size
    if N == 0:
        return dict(
            isi_prev_ms=np.array([], dtype=np.float64),
            prev_count =np.array([], dtype=np.int32),
            sorted_times=np.array([], dtype=np.int64),
            order=np.array([], dtype=np.int64),
            inv_order=np.array([], dtype=np.int64)
        )

    # Sort by time (stable), keep permutation to map back to original order
    order = np.argsort(t, kind='mergesort')
    t_sorted = t[order]
    inv_order = np.empty(N, dtype=np.int64)
    inv_order[order] = np.arange(N, dtype=np.int64)

    # Previous-ISI in samples → ms
    isi_prev_ms_sorted = np.empty(N, dtype=np.float64)
    isi_prev_ms_sorted[0] = np.nan
    if N > 1:
        dt = np.diff(t_sorted).astype(np.float64)
        isi_prev_ms_sorted[1:] = (dt * 1000.0) / float(sample_rate_hz)

    # Preceding-window counts via two-pointer
    w_samp = int(round(float(window_ms) * 1e-3 * float(sample_rate_hz)))
    j = 0
    prev_count_sorted = np.empty(N, dtype=np.int32)
    for i in range(N):
        ti = t_sorted[i]
        while j < i and (ti - t_sorted[j] > w_samp):
            j += 1
        # spikes in (ti - w_samp, ti] excluding the current spike → i - j
        prev_count_sorted[i] = i - j

    # Map metrics back to original input order
    isi_prev_ms = isi_prev_ms_sorted[inv_order]
    prev_count  = prev_count_sorted[inv_order]

    return dict(
        isi_prev_ms=isi_prev_ms,
        prev_count=prev_count,
        sorted_times=t_sorted,
        order=order,
        inv_order=inv_order
    )


def plot_amp_vs_prev_metrics(
    spike_times,           # 1D array of spike times in *samples*
    amplitudes,            # 1D array of per-spike amplitudes (same length/order as spike_times)
    *,
    sample_rate_hz,        # e.g. 20000
    window_ms=50.0,
    max_points=200000,     # optional downsampling for very large N
    alpha=0.25,
    figsize=(12, 4),
    x_label="Spike amplitude (arb. units)",
    title_suffix=""
):
    """
    Convenience plotting wrapper:
      • computes previous-ISI (ms) and preceding-window counts
      • shows two scatter plots side-by-side:
           (amplitude vs previous-ISI)  and  (amplitude vs count in previous window)

    Points are optionally subsampled to `max_points` for speed.
    """
    amps = np.asarray(amplitudes, dtype=np.float64)
    if amps.ndim != 1:
        raise ValueError("amplitudes must be a 1D array")
    if amps.shape[0] != len(spike_times):
        raise ValueError("spike_times and amplitudes must have the same length")

    feats = compute_prev_metrics(spike_times, sample_rate_hz=sample_rate_hz, window_ms=window_ms)
    isi_ms  = feats['isi_prev_ms']
    counts  = feats['prev_count']

    # Drop NaN amplitudes or NaN ISIs from plotting; keep alignment
    mask = np.isfinite(amps) & np.isfinite(isi_ms) & np.isfinite(counts)
    if not np.any(mask):
        raise RuntimeError("No finite data to plot after masking NaNs")

    a = amps[mask]
    isi = isi_ms[mask]
    cnt = counts[mask]

    # Optional subsampling for speed
    if a.size > max_points:
        idx = np.random.default_rng(123).choice(a.size, size=max_points, replace=False)
        a, isi, cnt = a[idx], isi[idx], cnt[idx]

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # Scatter: amplitude vs previous ISI (ms)
    axes[0].scatter(a, isi, s=4, alpha=alpha, edgecolors='none')
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(f"Time to previous spike (ms)")
    axes[0].set_title(f"Prev ISI vs amplitude{(' — ' + title_suffix) if title_suffix else ''}")
    axes[0].grid(True, alpha=0.2)

    # Scatter: amplitude vs count in previous window
    axes[1].scatter(a, cnt, s=4, alpha=alpha, edgecolors='none')
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(f"Spikes in previous {window_ms:.0f} ms")
    axes[1].set_title(f"Preceding count vs amplitude{(' — ' + title_suffix) if title_suffix else ''}")
    axes[1].grid(True, alpha=0.2)

    return dict(fig=fig, axes=axes, amplitudes=a, isi_prev_ms=isi, prev_count=cnt)




def plot_amp_prev_scatter_and_bar(
    spike_times,             # 1D array of spike times in *samples*
    amplitudes,              # 1D array of per-spike amplitudes (aligned with spike_times)
    *,
    sample_rate_hz,          # e.g. 20000
    window_ms=50.0,          # length of preceding window for counts
    max_k=None,              # show bars up to this count; None → use observed max
    min_count=1,             # require ≥ this many samples to show a bar
    error='sem',             # 'sem' (default) or 'std' for error bars
    max_points=200_000,      # optional subsample for the scatter
    alpha=0.25,
    figsize=(12, 4),
    x_label="Spike amplitude (arb. units)",
    title_suffix=""
):
    """
    Left panel: scatter of amplitude vs time to previous spike (ms).
    Right panel: bar chart of mean amplitude vs number of spikes in the preceding window (length `window_ms`),
                 with error bars (SEM by default).

    Returns dict with plotting artifacts and the per-bin stats.
    """
    # --- compute previous-ISI and preceding-window counts (aligned to input order)
    feats = compute_prev_metrics(spike_times, sample_rate_hz=sample_rate_hz, window_ms=window_ms)
    isi_ms  = feats['isi_prev_ms']
    counts  = feats['prev_count']
    amps    = np.asarray(amplitudes, dtype=np.float64)

    if amps.shape[0] != counts.shape[0]:
        raise ValueError("amplitudes and spike_times must have the same length")

    # Mask out any NaNs before plotting/statistics
    mask = np.isfinite(amps) & np.isfinite(isi_ms) & np.isfinite(counts)
    a = amps[mask]
    isi = isi_ms[mask]
    cnt = counts[mask].astype(int)

    # --- left panel: scatter (optionally subsample for speed)
    if a.size > max_points:
        sel = np.random.default_rng(123).choice(a.size, size=max_points, replace=False)
        a_sc = a[sel]; isi_sc = isi[sel]
    else:
        a_sc, isi_sc = a, isi

    # --- right panel: group amplitudes by preceding-count
    # Optionally cap the number of bars
    observed_max = int(cnt.max()) if cnt.size else 0
    Kmax = observed_max if (max_k is None) else int(max_k)
    # Collapse tail counts > Kmax into the last bin if we capped
    cnt_clip = np.minimum(cnt, Kmax)

    n = np.bincount(cnt_clip, minlength=Kmax + 1)
    sum1 = np.bincount(cnt_clip, weights=a,    minlength=Kmax + 1)
    sum2 = np.bincount(cnt_clip, weights=a*a,  minlength=Kmax + 1)

    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(n > 0, sum1 / n, np.nan)
        # sample variance (unbiased) where n>=2
        var  = np.where(n > 1, (sum2 - (sum1**2) / n) / (n - 1), np.nan)
        if error == 'std':
            err = np.sqrt(np.maximum(var, 0.0))
        else:  # 'sem'
            err = np.where(n > 0, np.sqrt(np.maximum(var, 0.0) / n), np.nan)

    # optionally filter bins with too few samples
    show = n >= int(min_count)
    ks_plot   = np.arange(Kmax + 1)[show]
    means_plot= mean[show]
    errs_plot = err[show]
    ns_plot   = n[show]

    # --- plotting
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # Scatter: amplitude vs previous ISI
    axes[0].scatter(a_sc, isi_sc, s=4, alpha=alpha, edgecolors='none')
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("Time to previous spike (ms)")
    axes[0].set_title(f"Prev ISI vs amplitude{(' — ' + str(title_suffix)) if title_suffix else ''}")
    axes[0].grid(True, which='both', alpha=0.2)

    # Bar chart: mean amplitude vs preceding-count
    bars = axes[1].bar(ks_plot, means_plot, yerr=errs_plot, color='C1', alpha=0.9, capsize=3)
    axes[1].set_xlabel(f"Spikes in previous {window_ms:.0f} ms")
    axes[1].set_ylabel("Mean spike amplitude")
    axes[1].set_title(f"Mean amplitude vs preceding count{(' — ' + str(title_suffix)) if title_suffix else ''}")
    axes[1].set_xticks(ks_plot)
    axes[1].grid(axis='y', alpha=0.2)

    # Optional: annotate counts above bars (small font)
    for k, h, n_k, rect in zip(ks_plot, means_track if 'means_track' in locals() else means_plot, ns_plot, bars):
        if np.isfinite(h):
            axes[1].text(rect.get_x() + rect.get_width()/2., h, f"n={int(n_k)}",
                         ha='center', va='bottom', fontsize=8, color='black', rotation=0)

    return {
        'fig': fig,
        'ax_scatter': axes[0],
        'ax_bar': axes[1],
        'ks': ks_plot,
        'mean_amplitude': means_plot,
        'error': errs_plot,
        'counts': ns_plot,
        'scatter_points': a_sc.size
    }

def _channel_rms(ei):
    # ei: (C, L)
    return np.sqrt(np.mean(ei.astype(np.float32)**2, axis=1))

def _align_snippets_to_ref(snips, ref_ch, center_idx, max_lag):
    """
    Align each snippet to the trough on ref_ch within ±max_lag around center_idx.
    snips: (C, L, N) float32
    Returns aligned copy (C, L, N), and the per-spike shift (int, + means shifted right).
    """
    C, L, N = snips.shape
    aligned = np.zeros_like(snips)
    shifts = np.zeros(N, dtype=np.int32)

    lo = max(0, center_idx - max_lag)
    hi = min(L, center_idx + max_lag + 1)

    # Find trough indices in the constrained window
    ref_tr = snips[ref_ch, :, :]
    trough_idx = np.argmin(ref_tr[lo:hi, :], axis=0) + lo
    delta = center_idx - trough_idx  # positive: shift right
    shifts[:] = delta.astype(np.int32)

    # Apply integer shifts with zero padding
    for i in range(N):
        s = int(shifts[i])
        if s == 0:
            aligned[:, :, i] = snips[:, :, i]
        elif s > 0:
            aligned[:, s:, i] = snips[:, :L - s, i]
        else:
            s = -s
            aligned[:, :L - s, i] = snips[:, s:, i]

    return aligned, shifts

def lighthouse_morpho_pc_gmm(
    raw_data,
    step1: dict,
    step2: dict,
    accepted_right_times: np.ndarray,
    *,
    window=(-40, 80),
    k_extreme=100,              # #lowest and #highest left spikes for channel selection
    rms_gate=5.0,               # channel kept if RMS in either EI >= gate
    max_channels=24,            # cap for speed; None to keep all significant channels
    best_align_lag=3,
    n_pcs=12,                   # total PCs across channels×time
    sample_for_pca=4000,        # subsample for PCA fit when N is huge
    amplitude_normalize=True,   # normalize by ref trough for feature building (shape-only PCs)
    bic_threshold=10.0,
    cos_threshold=0.95,
    random_state=0,
    use_kmeans=False            # set True to use k-means instead of GMM (no BIC)
) -> dict:
    """
    Morphology-driven clustering:
      1) Pick significant channels from 100-lowest & 100-highest LEFT spikes.
      2) Extract snippets for all accepted spikes (LEFT + RIGHT) on those channels only.
      3) Baseline, align to reference channel trough; optionally amplitude-normalize for features.
      4) PCA → PC features; GMM(1) vs GMM(2) or k-means(2).
      5) If 2-comp wins AND EI cosine between cluster EIs < cos_threshold → flag as mixture.

    Returns:
      {
        'sig_channels': int32[M],      # absolute channel indices used
        'ref_channel': int,            # absolute index
        'times_used': int64[N],        # chronological times used (after extraction validity)
        'labels2': int8[N],            # cluster labels for K=2 (GMM or k-means)
        'cluster_sizes': (int, int),

        # Model selection
        'bic1': float or None, 'bic2': float or None, 'delta_bic': float or None,
        'support_2comp': bool,

        # EI-shape agreement between clusters (computed on aligned, unnormalized snippets):
        'cos_clusters': float, 'cos_nch': int, 'cos_lag': int, 'cos_olen': int,

        # Debug stats:
        'n_spikes_total': int, 'n_spikes_left': int, 'n_spikes_right': int,
        'n_channels_used': int, 'n_pcs': int
      }
    """
    from axolotl_utils_ram import extract_snippets_fast_ram
    from collision_utils import median_ei_adaptive
    try:
        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture
        from sklearn.cluster import KMeans
    except Exception as e:
        raise RuntimeError("scikit-learn is required (PCA/GMM/KMeans).") from e

    # ---------- 0) Gather inputs ----------
    left_times = np.asarray(step1['left_times'], dtype=np.int64)
    left_vals  = np.asarray(step1['left_vals'],  dtype=np.float32)  # negative minima
    right_times_in = np.asarray(accepted_right_times, dtype=np.int64)

    # Build extremes from LEFT (exactly k_extreme or as many as possible)
    N_left = left_times.size
    if N_left < max(10, k_extreme*2):
        k_ext = max(1, min(k_extreme, N_left // 2))
    else:
        k_ext = k_extreme

    order = np.argsort(left_vals)          # ascending: most negative first (deepest)
    idx_high = order[:k_ext]
    idx_low  = order[-k_ext:]

    times_high = left_times[idx_high]
    times_low  = left_times[idx_low]

    # ---------- 1) Compute EIs for extremes on ALL channels; pick significant channels ----------
    C = raw_data.shape[1]
    all_ch = np.arange(C, dtype=np.int32)

    sn_high, _ = extract_snippets_fast_ram(raw_data, times_high, window=window, selected_channels=all_ch)
    sn_low,  _ = extract_snippets_fast_ram(raw_data, times_low,  window=window, selected_channels=all_ch)
    sn_high = sn_high.astype(np.float32, copy=False)
    sn_low  = sn_low.astype(np.float32,  copy=False)

    # EIs (median reduces collision bias)
    ei_high = median_ei_adaptive(sn_high)   # (C, L)
    ei_low  = median_ei_adaptive(sn_low)    # (C, L)

    rms_h = _channel_rms(ei_high)
    rms_l = _channel_rms(ei_low)
    sig_mask = (rms_h >= float(rms_gate)) | (rms_l >= float(rms_gate))
    sig_channels = np.where(sig_mask)[0].astype(np.int32)

    if sig_channels.size == 0:
        # Nothing strong → bail as single-unit (no evidence to split)
        return {
            'sig_channels': np.array([], dtype=np.int32),
            'ref_channel': int(np.argmin(np.min(ei_high, axis=1))),
            'times_used': np.array([], dtype=np.int64),
            'labels2': np.array([], dtype=np.int8),
            'cluster_sizes': (0, 0),
            'bic1': None, 'bic2': None, 'delta_bic': None, 'support_2comp': False,
            'cos_clusters': 1.0, 'cos_nch': 0, 'cos_lag': 0, 'cos_olen': 0,
            'n_spikes_total': 0, 'n_spikes_left': int(N_left), 'n_spikes_right': int(right_times_in.size),
            'n_channels_used': 0, 'n_pcs': 0
        }

    # Optionally cap to top-N channels by RMS on HIGH EI
    if max_channels is not None and sig_channels.size > int(max_channels):
        top_idx = np.argsort(rms_h[sig_channels])[::-1][:int(max_channels)]
        sig_channels = sig_channels[top_idx]

    # Choose ref channel within sig set: most negative trough on HIGH EI
    trough_vals = np.min(ei_high[sig_channels, :], axis=1)  # negative numbers
    ref_channel = int(sig_channels[int(np.argmin(trough_vals))])

    # ---------- 2) Extract snippets for all events on sig channels ----------
    all_times = np.concatenate([left_times, right_times_in]).astype(np.int64, copy=False)
    sn_all, valid_times = extract_snippets_fast_ram(
        raw_data, all_times, window=window, selected_channels=sig_channels
    )
    sn_all = sn_all.astype(np.float32, copy=False)  # (M, L, N)

    # Track how many from each side survived validity filtering
    # Map valid_times back to left/right counts
    lt_set = set(left_times.tolist())
    is_left_used = np.array([t in lt_set for t in valid_times], dtype=bool)

    M, L, N = sn_all.shape
    center_idx = int(-int(window[0]))

    # ---------- 3) Baseline subtract (per spike, per channel) ----------
    # Use pre-event region up to center - best_align_lag - 1
    pre_end = max(1, center_idx - best_align_lag - 1)
    if pre_end > 0:
        baseline = np.mean(sn_all[:, :pre_end, :], axis=1, keepdims=True)  # (M,1,N)
        sn_bs = sn_all - baseline
    else:
        sn_bs = sn_all.copy()

    # Align to ref channel trough (within sig set)
    ref_local_idx = int(np.where(sig_channels == ref_channel)[0][0])
    sn_aligned, shifts = _align_snippets_to_ref(sn_bs, ref_local_idx, center_idx, best_align_lag)

    # For PC features only: amplitude-normalize by ref trough so PCA sees shape, not size
    if amplitude_normalize:
        ref_tr = sn_aligned[ref_local_idx, :, :]                      # (L, N)
        a_ref = -np.min(ref_tr, axis=0).astype(np.float32)            # positive
        a_ref[a_ref < 1e-6] = 1.0
        sn_feat = sn_aligned / a_ref[np.newaxis, np.newaxis, :]
    else:
        sn_feat = sn_aligned

    # ---------- 4) PCA features (channels×time → n_pcs) ----------
    X = sn_feat.reshape(M * L, N).T  # (N, M*L)
    n_components = int(min(n_pcs, max(1, min(X.shape[0] - 1, X.shape[1]))))

    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=random_state)

    if X.shape[0] > int(sample_for_pca):
        rng = np.random.default_rng(int(random_state))
        idx_fit = rng.choice(X.shape[0], size=int(sample_for_pca), replace=False)
        pca.fit(X[idx_fit, :])
    else:
        pca.fit(X)

    Xp = pca.transform(X)  # (N, n_components)

    # ---------- 5) Clustering ----------
    bic1 = None; bic2 = None; delta_bic = None; labels2 = None
    support_2 = False

    if use_kmeans:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=2, n_init=20, random_state=random_state)
        labels2 = km.fit_predict(Xp).astype(np.int8)
        # Approximate 1-cluster "BIC-like" via total variance (not strictly comparable).
        # We leave BICs as None when using k-means.
        support_2 = True  # you’ll vet with EI-cosine next
    else:
        from sklearn.mixture import GaussianMixture
        g1 = GaussianMixture(n_components=1, covariance_type='full', n_init=3,
                             reg_covar=1e-6, random_state=random_state).fit(Xp)
        bic1 = float(g1.bic(Xp))
        g2 = GaussianMixture(n_components=2, covariance_type='full', n_init=10,
                             reg_covar=1e-6, random_state=random_state).fit(Xp)
        bic2 = float(g2.bic(Xp))
        delta_bic = bic1 - bic2
        labels2 = g2.predict(Xp).astype(np.int8)
        support_2 = bool(delta_bic is not None and delta_bic > float(bic_threshold))

    # ---------- 6) EI agreement between clusters (shape test) ----------
    # Build EIs on aligned (but NOT amplitude-normalized) snippets to keep real energy patterns.
    # Split indices by cluster
    if N == 0:
        cos_clusters = 1.0; cos_nch = 0; cos_lag = 0; cos_olen = 0
        n0 = 0; n1 = 0
    else:
        idx0 = np.where(labels2 == 0)[0]
        idx1 = np.where(labels2 == 1)[0]
        n0, n1 = int(idx0.size), int(idx1.size)

        if n0 == 0 or n1 == 0:
            cos_clusters = 1.0; cos_nch = 0; cos_lag = 0; cos_olen = 0
        else:
            from collision_utils import median_ei_adaptive
            from numpy import float32, int64
            ei0 = median_ei_adaptive(sn_aligned[:, :, idx0])
            ei1 = median_ei_adaptive(sn_aligned[:, :, idx1])

            # cosine on the same channels, same gate/lag as earlier conventions
            from numpy import asarray
            from typing import Tuple
            # reuse your function
            base_cos, base_nch, base_lag, base_olen = cosine_two_eis(
                ei0, ei1, rms_gate=rms_gate, use_abs=True, best_align_lag=best_align_lag
            )
            cos_clusters = float(base_cos); cos_nch = int(base_nch)
            cos_lag = int(base_lag); cos_olen = int(base_olen)

    # Final verdict: 2-comp only if model supports it AND shapes differ
    support_2_final = bool(support_2 and (cos_clusters < float(cos_threshold)))

    return {
        'sig_channels': sig_channels,
        'ref_channel': int(ref_channel),
        'times_used': valid_times.astype(np.int64, copy=False),
        'labels2': labels2,
        'cluster_sizes': (n0, n1),

        'bic1': bic1, 'bic2': bic2, 'delta_bic': delta_bic,
        'support_2comp': support_2_final,

        'cos_clusters': float(cos_clusters),
        'cos_nch': int(cos_nch), 'cos_lag': int(cos_lag), 'cos_olen': int(cos_olen),

        'n_spikes_total': int(N),
        'n_spikes_left': int(is_left_used.sum()),
        'n_spikes_right': int(N - is_left_used.sum()),
        'n_channels_used': int(sig_channels.size),
        'n_pcs': int(n_components)
    }
