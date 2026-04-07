"""collision_utils.py
Utility helpers for axon‑based spike‑collision modelling.

The module now has **two tiers of lag handling**:

1. **`quick_unit_filter()`** – a *fast* pass that uses cross‑correlation on the
   **peak channel only** to align each template, computes ΔRMS on the full
   selected‑channel set, and returns the subset of units that produce a
   negative (i.e. improving) ΔRMS below a user threshold (default –5).
   This is the new gate you asked for on 9 Jul 2025.

2. **`scan_unit_lags()`** – the existing exhaustive per‑unit lag scan (using
   `lag_delta_rms`) but now run **only on the units accepted by the quick
   filter**.

The downstream API (evaluate_local_group → resolve_snippet →
accumulate_unit_stats etc.) is unchanged.

All numeric hyper‑parameters are kwargs with sensible defaults.

---------------------------------------------------------------------
Public symbols
--------------
roll_zero, tempered_weights, quick_unit_filter, lag_delta_rms,
scan_unit_lags, score_active_set, evaluate_local_group,
accumulate_unit_stats, accept_units, micro_align_units,
subtract_overlap_tail
"""

from __future__ import annotations

from itertools import product
import numpy as np
from typing import Dict, List, Sequence, Tuple, Iterable
from collections import defaultdict
import itertools
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import correlate  # fast C‑implementation
import plot_ei_waveforms

from compare_eis import compare_eis


# ------------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------------

def roll_zero(arr: np.ndarray, lag: int) -> np.ndarray:
    """Shift 1‑D array by *lag* samples with zero‑padding (no wrap‑around)."""
    out = np.zeros_like(arr)
    if lag > 0:
        out[lag:] = arr[:-lag]
    elif lag < 0:
        out[:lag] = arr[-lag:]
    else:
        out[:] = arr
    return out

def roll_zero_all(ei: np.ndarray, lag: int) -> np.ndarray:
    """Shift all channels in [C, T] EI by lag samples, with zero-padding."""
    out = np.zeros_like(ei)
    if lag > 0:
        out[:, lag:] = ei[:, :-lag]
    elif lag < 0:
        out[:, :lag] = ei[:, -lag:]
    else:
        out[:] = ei
    return out

def tempered_weights(p2p_vec: np.ndarray, chans: Iterable[int], *, beta: float = 0.5) -> np.ndarray:
    """Normalised weights *w_c ∝ (p2p_c)^β* over **chans**."""
    w = p2p_vec[list(chans)] ** beta
    s = w.sum()
    return w / s if s else w



def delta_rms(x, y, weights): 

    rms_raw = np.sqrt((x**2     ).mean(axis=1))   # [C]
    rms_res = np.sqrt(((x-y)**2).mean(axis=1))    # [C]
    return np.sum(weights * (rms_res - rms_raw))


def best_shift_ei(trace_rw, trace_ei, max_shift=40):
    """
    Find the best right shift (0…max_shift) to apply to EI to match trace_rw.

    Returns
    -------
    best_lag : int  (0…max_shift)
    """
    L = len(trace_rw)
    scores = []

    for lag in range(max_shift + 1):  # 0…40 inclusive
        # EI shifted right: pad on left with zeros
        ei_shifted = np.zeros_like(trace_ei)
        ei_shifted[lag:] = trace_ei[:L - lag]

        s = np.dot(trace_rw, ei_shifted)
        scores.append((lag, s))

        # weights = aligned.max(axis=1) - aligned.min(axis=1)
        # weights[weights>200] = 200 # clips weights

        # rms_res = np.sqrt(((raw_sel-aligned)**2).mean(axis=1))
        # delta = np.sum(weights * (rms_res - rms_raw[sel_ch]))

    best_lag = max(scores, key=lambda x: x[1])[0]
    return best_lag

# ------------------------------------------------------------------
# 0.  FAST PEAK‑CHANNEL FILTER (new)
# ------------------------------------------------------------------
def quick_unit_filter(
    unit_ids,
    raw_snippet: np.ndarray,
    unit_info: dict,
    *,
    delta_thr: float = 0.0,
    rms_raw
):
    """Fast x-corr gate; behaviour now identical to the original inline code."""
    rows = []

    
    for uid in unit_ids:
        ei       = unit_info[uid]['ei']
        peak_ch  = unit_info[uid]['peak_channel']
        sel_ch   = unit_info[uid]['selected_channels']

        if len(sel_ch) == 0:
            # print(f"Skipping {uid}: no selected_channels")
            continue

        trace_rw = raw_snippet[sel_ch]
        trace_ei = ei[sel_ch]

        weights = trace_ei.max(axis=1) - trace_ei.min(axis=1)
        weights[weights>200] = 200 # clips weights
        scores = []
        for lag in range(41):    
            shifted_ei = roll_zero_all(trace_ei, lag)
            rms_res = np.sqrt(((trace_rw-shifted_ei)**2).mean(axis=1))
            delta = np.sum(weights * (rms_res - rms_raw[sel_ch]))
            scores.append((lag, delta))
        
        lag0, delta = min(scores, key=lambda x: x[1])

        # trace_rw = raw_snippet[peak_ch]
        # trace_ei = ei[peak_ch]

        # lag0 = best_shift_ei(trace_rw, trace_ei, max_shift=40)

        # aligned  = np.roll(ei[sel_ch], lag0, axis=1)

        # # RMS on *all* samples of selected channels  (no mask)
        # raw_sel = raw_snippet[sel_ch]
        # weights = aligned.max(axis=1) - aligned.min(axis=1)
        # weights[weights>200] = 200 # clips weights

        # rms_res = np.sqrt(((raw_sel-aligned)**2).mean(axis=1))
        # delta = np.sum(weights * (rms_res - rms_raw[sel_ch]))

        # delta = delta_rms(raw_sel, aligned, weights)

        # rms_pre  = sum(np.sqrt(np.mean(raw_sel**2, axis=1)))
        # rms_post = sum(np.sqrt(np.mean((raw_sel - aligned)**2, axis=1)))
        # delta    = rms_post - rms_pre

        # if uid=='unit_134':
        #     print(f"delta for 134: {delta}, =lag {lag0}. Recomputing:")
        #     aligned1  = np.roll(ei[sel_ch], 1, axis=1)

        #     # RMS on *all* samples of selected channels  (no mask)
        #     raw_sel1 = raw_snippet[sel_ch]
        #     weights1 = aligned1.max(axis=1) - aligned1.min(axis=1)
        #     weights1[weights1>200] = 200 # clips weights

        #     delta1 = delta_rms(raw_sel1, aligned1, weights1)
        #     print(f"delta for 134: {delta1},  with lag 2 ")


        if delta < delta_thr:          # improvement only
            rows.append({
                'uid': uid,
                'lag': lag0,
                # 'rms_pre': rms_pre,
                # 'rms_post': rms_post,
                'delta': delta,
                'peak_ch': peak_ch,
            })

    return pd.DataFrame(rows)

# 3) build channel‑to‑unit index  ---------------------------------------

def build_channel_index(good_units, unit_info):
    """
    good_units :  • list / array / pandas-Series of uid strings   OR
                 • DataFrame with a 'uid' column
    unit_info  : {uid: {'selected_channels': [...]}}
    Returns    : {channel: [uids]}
    """
    # allow both input types
    if hasattr(good_units, "columns") and "uid" in good_units.columns:
        uid_iter = good_units["uid"]
    else:
        uid_iter = good_units

    from collections import defaultdict
    ch_map = defaultdict(list)
    for uid in uid_iter:
        for ch in unit_info[uid]["selected_channels"]:
            ch_map[ch].append(uid)
    return dict(sorted(ch_map.items()))

# ------------------------------------------------------------------
# 1.  PER‑UNIT ΔRMS SWEEP   (unchanged)
# ------------------------------------------------------------------

def lag_delta_rms(
    uid: int,
    raw_snippet: np.ndarray,                # (C,T)
    p2p_all: Dict[int, np.ndarray],
    unit_info: Dict[int, Dict],
    *,
    beta: float = 0.5,
    amp_thr: float = 25.0,
    mask_thr: float = 5.0,
    max_lag: int = 60,                      # *range* around peak position
    rms_raw
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scan lags so that the EI's absolute peak (index *peak_idx*) lands in
    snippet samples 40 … 80.  No further gating needed.

    Returns
    -------
    lags   : 1-D array of tested lags
    score  : ΔRMS for each lag (-inf where channels had no weight)
    """
    ei        = unit_info[uid]["ei"]
    # peak_idx  = 40 # EI center - convention

    # SNIP_LEN = raw_snippet.shape[1]
    # mid   = SNIP_LEN // 2          # 60
    # base  = mid - peak_idx         # lag that centres the EI peak at 60
    # # legal lag window so spike peak ∈ [40,80]
    # lag_min = base - max_lag       # 60-peak_idx-max_lag
    # lag_max = base + max_lag       # 60-peak_idx+max_lag
    # lags    = np.arange(lag_min, lag_max + 1)

    peak_idx      = 40   # EI peak index, by convention
    target_center = 60   # desired landing for the EI peak *inside the snippet*
    base          = target_center - peak_idx   # 60 - 40 = 20, invariant across T

    lag_min = base - max_lag
    lag_max = base + max_lag
    lags    = np.arange(lag_min, lag_max + 1)

    a_ch   = p2p_all[uid]
    chans  = [c for c in unit_info[uid]["selected_channels"]
              if a_ch[c] >= amp_thr]
    if not chans:
        raise ValueError("no channels above amp_thr")

    W     = tempered_weights(a_ch, chans, beta=beta)
    score = np.zeros_like(lags, dtype=np.float32)

    for i, lag in enumerate(lags):
        # s = 0.0

        shifted_ei = roll_zero_all(ei[chans], lag)

        # RMS on *all* samples of selected channels  (no mask)
        raw_sel = raw_snippet[chans]
        weights = shifted_ei.max(axis=1) - shifted_ei.min(axis=1)
        weights[weights>200] = 200 # clips weights

        rms_res = np.sqrt(((raw_sel-shifted_ei)**2).mean(axis=1))
        delta = np.sum(weights * (rms_res - rms_raw[chans]))

        # delta = delta_rms(raw_sel, shifted_ei, weights)

        # for w, ch in zip(W, chans):
        #     tmpl = roll_zero(ei[ch], lag)
        #     raw  = raw_snippet[ch]
        #     m    = np.abs(tmpl) > mask_thr
        #     if not m.any():
        #         continue
        #     rms_raw = np.sqrt(np.mean(raw[m] ** 2))
        #     rms_res = np.sqrt(np.mean((raw[m] - tmpl[m]) ** 2))
        #     s += w * (rms_raw - rms_res)
        score[i] = -delta # invert for historical reasons
    return lags, score

# ------------------------------------------------------------------
# 2.  SCAN TOP‑K LAGS FOR A SET OF UNITS   (slim wrapper)
# ------------------------------------------------------------------

def scan_unit_lags(
    unit_ids: Iterable[int],
    raw_snippet: np.ndarray,
    p2p_all: Dict[int, np.ndarray],
    unit_info: Dict[int, Dict],
    *,
    beta: float = 0.5,
    amp_thr: float = 25.0,
    mask_thr: float = 5.0,
    max_lag: int = 60,
    top_k: int = 3,
    rms_raw
) -> Dict[int, List[int]]:
    """
    Build lag_dict {uid: [top_k lags]} for the supplied *unit_ids*.
    Only lags whose EI peak maps to snippet samples 40-80 are kept.
    """
    lag_dict = {}

    for uid in unit_ids:
        try:
            lags, score = lag_delta_rms(
                uid, raw_snippet, p2p_all, unit_info,
                beta=beta, amp_thr=amp_thr,
                mask_thr=mask_thr, max_lag=max_lag, rms_raw=rms_raw
            )
            # discard lags that scored -inf (geometric gate failed)
            keep = np.isfinite(score)
            if not keep.any():
                continue

            lags  = lags[keep]
            score = score[keep]
            order = np.argsort(score)[::-1][:top_k]
            lag_dict[uid] = [int(lags[j]) for j in order]

        except Exception:
            # unit had no channels above amp_thr or other issue – skip it
            continue

    return lag_dict


# ─────────────────────────── combo‑scoring primitives ─────────────────────────

def tempered_weights(p2p_vec: np.ndarray, chans: Sequence[int], beta: float = 0.5) -> np.ndarray:
    w = p2p_vec[chans] ** beta
    return w / w.sum()


def score_active_set(
    active_dict: Dict[int, int],          # {uid: lag}
    union_chans: Sequence[int],
    raw_local: np.ndarray,                # [C_union, T]
    unit_info: Dict[int, Dict],
    p2p_all: Dict[int, np.ndarray],
    rolled_bank,
    *,
    beta: float = 0.5,
    rms_raw,
    debug: bool = False,
    chan_weights=None,
) -> float:
    """
    Return weighted ΔRMS for the given unit set.
    """
    if not active_dict:
        return 0.0


    # tmpl_sum = np.zeros_like(raw_local, dtype=np.float32)
    # for u, lag in active_dict.items():
    #     shifted = np.array(
    #         [roll_zero(unit_info[u]["ei"][c], lag) for c in union_chans]
    #     )
    #     tmpl_sum += shifted

    tmpl_sum = None
    for idx, (uid, lag) in enumerate(active_dict.items()):
        if idx == 0:
            tmpl_sum = rolled_bank[(uid, lag)].copy()
        else:
            tmpl_sum += rolled_bank[(uid, lag)]

    tmpl_sum = tmpl_sum[union_chans]

    # choose weight
    if chan_weights is None:
        weights = np.zeros(len(union_chans), dtype=np.float32)
        for k, c in enumerate(union_chans):
            a_u = max(p2p_all[u][c] for u in active_dict)
            weights[k] = a_u ** beta
    else:
        weights = chan_weights # same weight both calls

    
    rms_res = np.sqrt(((raw_local-tmpl_sum)**2).mean(axis=1))
    if debug:
        diffs = weights * (rms_res - rms_raw[union_chans])

        for ch, val in zip(union_chans, diffs):
            print(f"channel {ch}: {val:.1f}")
    delta = np.sum(weights * (rms_res - rms_raw[union_chans]))

    score = -delta

    return score


# ------------------------------------------------------------------
# STEP 2 – GREEDY BEAM SEARCH  (no additive assumption needed)
# ------------------------------------------------------------------
def beam_combo_search(units, lag_dict, union_chans, raw_local,
                      unit_info, p2p_all,rolled_bank,
                      *, beta=0.5, beam=4, rms_raw):
    """
    Parameters
    ----------
    units        : list[int]               # pruned working set W
    lag_dict     : {uid: [lag1, lag2, …]}
    union_chans  : list[int]               # channels used in score
    raw_local    : ndarray (len(union_chans), T)
    Returns
    -------
    best_combo   : dict {'lags': {uid:lag}, 'score': ΔRMS}
    """
    # each beam element is (active_dict, score_float)
    beams = [({}, 0.0)]

    for u in units:
        new_beams = []
        for active, sc in beams:
            # option 0 – skip this unit
            new_beams.append((active, sc))

            # option k – place unit at each candidate lag
            for L in lag_dict[u]:
                active2 = dict(active)   # shallow copy
                active2[u] = L
                s2 = score_active_set(
                        active2, union_chans,
                        raw_local, unit_info, p2p_all, rolled_bank,
                        beta=beta, rms_raw=rms_raw
                     )
                new_beams.append((active2, s2))

        # keep best `beam`
        new_beams.sort(key=lambda t: t[1], reverse=True)
        beams = new_beams[:beam]

    best_active, best_score = beams[0]
    return {'lags': best_active, 'score': best_score}


def prune_combo(active, union_chans, raw_local,
                unit_info, p2p_all, rolled_bank,rms_raw,
                *, beta=0.5):
    """
    Iteratively remove units with non-positive marginal ΔRMS.
    """
    changed = True
    while changed and active:
        changed = False
        score_full = score_active_set(active, union_chans,
                                    raw_local, unit_info, p2p_all,rolled_bank,
                                    beta=beta, rms_raw=rms_raw)
        worst_uid, worst_gain = None, None
        for u in list(active):
            s_minus = score_active_set({k:v for k,v in active.items() if k!=u},
                                    union_chans, raw_local,
                                    unit_info, p2p_all,rolled_bank,
                                    beta=beta, rms_raw=rms_raw)
            gain = score_full - s_minus
            if worst_gain is None or gain < worst_gain:
                worst_uid, worst_gain = u, gain
        if worst_gain is not None and worst_gain <= 0:
            del active[worst_uid]
            changed = True
    return active
# ----------------------------------------------------------------------
# Beam-search version – returns TWO objects to match resolve_snippet()
# ----------------------------------------------------------------------
def evaluate_local_group(
    c0: int,
    working_units: Sequence[int],
    raw_snippet: np.ndarray,
    unit_info: Dict[int, Dict],
    lag_dict: Dict[int, List[int]],
    p2p_all: Dict[int, np.ndarray],
    rolled_bank,
    *,
    amp_thr: float = 25.0,
    mask_thr: float = 5.0,
    beta: float = 0.5,
    beam: int = 4,
    rms_raw,
    plot: bool = False,
):
    """Pick best lag-combo for one anchor channel using beam search.

    Returns
    -------
    best_combo      : dict  {'lags': {uid: lag}, 'score': ΔRMS}
    per_unit_delta  : dict  {uid: marginal ΔRMS vs empty set}
    """
    # 1. union of strong channels
    union = {c0}
    for u in working_units:
        union.update(np.where(p2p_all[u] >= amp_thr)[0])
    union_chans = sorted(union)
    raw_local   = raw_snippet[union_chans]

    # 2. beam search over units × candidate lags -----------------------
    best_combo = beam_combo_search(
        units       = working_units,
        lag_dict    = lag_dict,
        union_chans = union_chans,
        raw_local   = raw_local,
        unit_info   = unit_info,
        p2p_all     = p2p_all,
        rolled_bank = rolled_bank,
        beta        = beta,
        beam        = beam,
        rms_raw     = rms_raw
    )


    # prune the beam result
    pruned_lags = prune_combo(
        dict(best_combo['lags']),    # copy
        union_chans, raw_local,
        unit_info, p2p_all,rolled_bank,rms_raw
    )
    best_combo['lags']  = pruned_lags
    best_combo['score'] = score_active_set(
                            pruned_lags, union_chans,
                            raw_local, unit_info, p2p_all,rolled_bank,
                            beta=beta, rms_raw=rms_raw)


    full_score = best_combo['score'] if best_combo else 0.0

    # 3. per-unit marginal ΔRMS  (empty set is baseline)
    per_unit_delta = {}
    for u in working_units:
        per_unit_delta[u] = full_score if u in best_combo.get('lags', {}) else 0.0

    return best_combo, per_unit_delta

    # results = []
    # for combo in combos:
    #     active = {u: lag for u, lag in zip(working_units, combo) if lag is not None}
    #     if not active:
    #         continue
    #     s = score_active_set(active, union_chans, raw_local, unit_info, p2p_all, mask_thr=mask_thr, beta=beta)
    #     results.append({"lags": active, "score": s})

    # if not results:
    #     raise RuntimeError("No valid combinations found.")

    # results.sort(key=lambda d: d["score"], reverse=True)
    # best = results[0]

    # # per‑unit marginal contribution -----------------------------
    # full_score = best["score"]
    # per_unit_delta = {}
    # for u in working_units:
    #     subset = {k: v for k, v in best["lags"].items() if k != u}
    #     s_subset = score_active_set(subset, union_chans, raw_local, unit_info, p2p_all, mask_thr=mask_thr, beta=beta)
    #     per_unit_delta[u] = full_score - s_subset

    # if plot:
    #     plt.figure(figsize=(3, 2))
    #     plt.bar(list(per_unit_delta.keys()), list(per_unit_delta.values()))
    #     plt.title(f"Channel {c0} — marginal ΔRMS")
    #     plt.tight_layout()

    # return best, per_unit_delta

MAX_W_UNITS = 15             # hard cap per anchor channel

# ------------------------------------------
# quick helper – is a unit already “settled”?
def is_certain(uid, unit_log, pos_thresh=3, neg_thresh=3):
    """
    Return True if this uid was either:
      • placed with positive ΔRMS ≥ twice, OR
      • examined ≥ (pos+neg) times but never had ΔRMS > 0
    """
    rec = unit_log.get(uid)
    if not rec:
        return False
    pos = sum(d > 0 for d in rec["deltas"])
    neg = sum(d <= 0 for d in rec["deltas"])
    return (pos >= pos_thresh) or (neg >= neg_thresh and pos == 0)
# ------------------------------------------

# ─────────────────────────── snippet‑level resolver ────────────────────────────

def resolve_snippet(
    raw_snippet: np.ndarray,
    good_units: Sequence[int],
    channel_to_units: Dict[int, List[int]],
    lag_dict: Dict[int, List[int]],
    unit_info: Dict[int, Dict],
    p2p_all: Dict[int, np.ndarray],
    *,
    amp_thr: float = 25.0,
    beta: float = 0.5,
    beam: int = 4,
    rms_raw,
    rolled_bank,
    ei_positions,
    plot: bool = False,
):
    """Run the unresolved‑channel loop for one snippet.

    Returns
    -------
    best_combo      : dict  {"lags": {uid:lag}, "score": ΔRMS}
    per_unit_delta  : dict  {uid: cumulative ΔRMS across anchor iterations}
    combo_history   : list  full trace of best_combo per anchor iteration
    """
    raw_ptp = raw_snippet.ptp(axis=1)
    unresolved_chans = {ch for uid in good_units for ch in unit_info[uid]["selected_channels"]}

    unit_log = defaultdict(lambda: {"deltas": [], "lags": []})
    combo_history = []

    while unresolved_chans:
        # guard_loop = len(unresolved_chans)
        c0 = max(unresolved_chans, key=lambda c: raw_ptp[c])

        W_full = channel_to_units[c0]
        # print(f"Starting W: {W_full}, channel {c0}")

        # --- 1. drop units we are already certain about -----------------
        W = [u for u in W_full if not is_certain(u, unit_log)]

        # --- 2. amplitude-based pruning to ≤ MAX_W_UNITS ----------------
        if len(W) > MAX_W_UNITS:
            W.sort(key=lambda u: p2p_all[u][c0], reverse=True)
            W = W[:MAX_W_UNITS]
        # print(f"Working list on channel {c0}: {W}")
        # fall back: if everything was pruned, keep the top-amp unit
        if not W:
            W = [max(W_full, key=lambda u: p2p_all[u][c0])]

        
        # print(f"Final W: {len(W)}")

        best_combo, per_unit_delta_anchor = evaluate_local_group(
            c0, W, raw_snippet, unit_info, lag_dict, p2p_all,rolled_bank,
            amp_thr=amp_thr, beta=beta, beam=beam,rms_raw=rms_raw, plot=plot
        )
        combo_history.append(best_combo)

        # print(best_combo)

        # accumulate logs
        for uid in per_unit_delta_anchor:
            unit_log[uid]["deltas"].append(per_unit_delta_anchor[uid])
            unit_log[uid]["lags"].append(best_combo["lags"].get(uid, math.nan))

        # mark explained channels
        W_set = set(W)
        # print(f"removed channels for combo: {W} with best: {best_combo}")
        for ch in list(unresolved_chans):
            # keep only units that are still uncertain
            remaining = [u for u in channel_to_units[ch] if not is_certain(u, unit_log)]
            if set(remaining).issubset(W_set):
                # print(ch)
                unresolved_chans.remove(ch)

        # print(f"CHANNELS LEFT: {len(unresolved_chans)}")

        # if len(unresolved_chans)==guard_loop:
        #     print(f"STUCK ON CHANNELS, {len(unresolved_chans)}: {unresolved_chans}")
        #     print(best_combo)
        #     print(unit_info['unit_185']['selected_channels'])
        #     print(channel_to_units[220])


    # ------------------------------------------------------------
    # Build initial global combo from accumulated logs
    # ------------------------------------------------------------

    # keep only units that were actually placed at least once
    unit_log = {u: rec for u, rec in unit_log.items()
                if not np.all(np.isnan(rec["lags"]))}

    active = {}
    for u, rec in unit_log.items():
        if np.all(np.isnan(rec["lags"])):
            continue
        lag = float(np.nanmedian(rec["lags"]))
        if np.isnan(lag):
            continue
        if u not in lag_dict:
            continue
        lag_rounded = int(round(lag))
        allowed_lags = lag_dict[u]
        if lag_rounded not in allowed_lags:
            # snap to closest allowed lag
            lag_rounded = min(allowed_lags, key=lambda l: abs(l - lag_rounded))
        active[u] = lag_rounded

    union_chans = sorted({c for u in active
                            for c in unit_info[u]["selected_channels"]})
    raw_local = raw_snippet[union_chans]

    # ------------------------------------------------------------
    # Iterative marginal-gain pruning *on the final set*
    # ------------------------------------------------------------
    # C, T = raw_local.shape
    # mask = np.zeros_like(raw_local, dtype=bool)
    # for c in range(C):
    #     sigma = np.median(np.abs(raw_local[c])) / 0.6745
    #     mask[c] = np.abs(raw_local[c]) > 4 * sigma

    import matplotlib.pyplot as plt
    from plot_ei_waveforms import plot_ei_waveforms

    def _template_sum(active_dict, union_chans, rolled_bank):
        """Return [len(union_chans), T] sum of rolled templates in active_dict."""
        if not active_dict:
            return np.zeros_like(raw_local, dtype=np.float32)

        tmpl_sum = np.zeros_like(raw_local, dtype=np.float32)
        for uid, lag in active_dict.items():
            rolled = rolled_bank[(uid, lag)][union_chans]   # slice to local chans
            tmpl_sum += rolled
        return tmpl_sum

    def marginal_prune(active_dict, local_beta):
        while len(active_dict) > 1:
            worst_uid, worst_gain = None, 0.0

            for uid in list(active_dict):
                sel = unit_info[uid]['selected_channels']
                weights = np.zeros(len(union_chans), dtype=np.float32)
                for k, c in enumerate(union_chans):
                    if c in sel:
                        weights[k] = (p2p_all[uid][c]) ** local_beta

                w_sum = weights.sum()
                if w_sum > 0:                       # normalise to unit ℓ¹ norm
                    weights /= w_sum

                full = score_active_set(active_dict, union_chans, raw_local,
                                        unit_info, p2p_all, rolled_bank,
                                        beta=local_beta, rms_raw=rms_raw,
                                        chan_weights=weights)

                alt = dict(active_dict); alt.pop(uid)
                minus = score_active_set(alt, union_chans, raw_local,
                                        unit_info, p2p_all, rolled_bank,
                                        beta=local_beta, rms_raw=rms_raw,
                                        chan_weights=weights)

                gain = full - minus
                if gain <= worst_gain:         # note: worst_gain starts at 0
                    worst_uid, worst_gain = uid, gain

            if worst_uid is None or worst_gain > 0:   # no harmful units remain
                break

            # print(f"Pruned {worst_uid}  (gain {worst_gain:.2f})")

            # if worst_uid=='unit_237':
            #     for uid in list(active_dict):
            #         sel = unit_info[uid]['selected_channels']
            #         weights = np.zeros(len(union_chans), dtype=np.float32)
            #         for k, c in enumerate(union_chans):
            #             if c in sel:
            #                 weights[k] = (p2p_all[uid][c]) ** local_beta

            #         w_sum = weights.sum()
            #         if w_sum > 0:                       # normalise to unit ℓ¹ norm
            #             weights /= w_sum

            #         full = score_active_set(active_dict, union_chans, raw_local,
            #                                 unit_info, p2p_all, rolled_bank,
            #                                 beta=local_beta, rms_raw=rms_raw,
            #                                 chan_weights=weights)

            #         alt = dict(active_dict); alt.pop(uid)
            #         minus = score_active_set(alt, union_chans, raw_local,
            #                                 unit_info, p2p_all, rolled_bank,
            #                                 beta=local_beta, rms_raw=rms_raw,
            #                                 chan_weights=weights)
            #         gain = full - minus
            #         print(f"unit: {uid}, gain: {gain}")


            active_dict.pop(worst_uid)        # remove single worst unit, then loop

        return active_dict

    # def marginal_prune(active_dict, local_beta):
    #     """
    #     Iteratively remove any unit whose marginal gain (using weights derived
    #     *only* from that unit’s own P2P values) is ≤ 0.
    #     """
    #     while len(active_dict) > 1:          # ← guard: stop if only one left
    #         removed = False

    #         for uid in list(active_dict):          # deterministic order
    #             # ----------------------------------------------------------
    #             # Build weight vector for *this* unit only
    #             # ----------------------------------------------------------
    #             sel = set(unit_info[uid]["selected_channels"])   # its own chans
    #             weights = np.zeros(len(union_chans), dtype=np.float32)

    #             for k, c in enumerate(union_chans):
    #                 if c in sel:
    #                     weights[k] = (p2p_all[uid][c]) ** local_beta    # ONLY uid

    #             # ----------------------------------------------------------
    #             # Score WITH the unit (full set)
    #             # ----------------------------------------------------------
    #             score_full = score_active_set(
    #                 active_dict, union_chans, raw_local,
    #                 unit_info, p2p_all, rolled_bank,
    #                 beta=local_beta, rms_raw=rms_raw,
    #                 chan_weights=weights
    #             )

    #             # ----------------------------------------------------------
    #             # Score WITHOUT the unit
    #             # ----------------------------------------------------------
    #             alt = dict(active_dict)
    #             alt.pop(uid)

    #             score_minus = score_active_set(
    #                 alt, union_chans, raw_local,
    #                 unit_info, p2p_all, rolled_bank,
    #                 beta=local_beta, rms_raw=rms_raw,
    #                 chan_weights=weights
    #             )

    #             if uid == "unit_113":

    #                 ei_with = _template_sum(active_dict, union_chans, rolled_bank)
    #                 alt = dict(active_dict); alt.pop(uid)
    #                 ei_without = _template_sum(alt, union_chans, rolled_bank)

    #                 for title, ei1 in [
    #                     ("WITH unit_113", ei_with),
    #                     ("WITHOUT unit_113", ei_without),
    #                 ]:
    #                     plt.figure(figsize=(25, 12))
    #                     plt.suptitle(title, fontsize=14)
    #                     plot_ei_waveforms(
    #                         [ei1, raw_local],           # first template-sum, then raw
    #                         ei_positions[union_chans],
    #                         scale=60.0,
    #                         box_height=1.0,
    #                         box_width=50.0,
    #                         colors=["red", "blue"],
    #                         aspect=0.5,
    #                     )
    #                     plt.show()
    #                 print(active_dict)


    #             gain = score_full - score_minus
    #             print(f"unit {uid:>8}   gain = {gain:8.3f}")

    #             if gain <= 0:                # hurts or neutral → drop
    #                 active_dict.pop(uid)
    #                 removed = True
    #                 break                    # restart with smaller set

    #         if not removed:                  # nothing pruned this pass
    #             break

    #     return active_dict




    # print(f"active: {active}")
    pruned = []
    if len(active) > 0:
        pruned = marginal_prune(active, local_beta=0.5)
    # print(f"pruned: {pruned}")


    # ------------------------------------------------------------
    # Per-unit marginal ΔRMS on pruned set
    # ------------------------------------------------------------
    per_unit_delta = {}
    if len(pruned) > 0:
        for uid in list(pruned):          # deterministic order
            # ----------------------------------------------------------
            # Build weight vector for *this* unit only
            # ----------------------------------------------------------
            sel = set(unit_info[uid]["selected_channels"])   # its own chans
            weights = np.zeros(len(union_chans), dtype=np.float32)

            for k, c in enumerate(union_chans):
                if c in sel:
                    weights[k] = (p2p_all[uid][c]) ** beta    # ONLY uid

            # ----------------------------------------------------------
            # Score WITH the unit (full set)
            # ----------------------------------------------------------
            score_full = score_active_set(
                pruned, union_chans, raw_local,
                unit_info, p2p_all, rolled_bank,
                beta=beta, rms_raw=rms_raw,
                chan_weights=weights
            )

            # ----------------------------------------------------------
            # Score WITHOUT the unit
            # ----------------------------------------------------------
            alt = dict(pruned)
            alt.pop(uid)

            score_minus = score_active_set(
                alt, union_chans, raw_local,
                unit_info, p2p_all, rolled_bank,
                beta=beta, rms_raw=rms_raw,
                chan_weights=weights
            )

            gain = score_full - score_minus
            per_unit_delta[uid] = score_full - score_minus
            # print(f"unit {uid:>8}   gain = {gain:8.3f}")





    # score_full = score_active_set(pruned, union_chans,
    #                             raw_local, unit_info, p2p_all,rolled_bank,
    #                             beta=beta, rms_raw=rms_raw)
    # for u in pruned:
    #     alt = dict(pruned); alt.pop(u)
    #     score_minus = score_active_set(alt, union_chans,
    #                                 raw_local, unit_info, p2p_all,rolled_bank,
    #                                 beta=beta, rms_raw=rms_raw)
    #     per_unit_delta[u] = score_full - score_minus     # marginal gain

    if len(pruned) > 0:
        best_combo_global = {"lags": pruned, "score": score_full}
    else:
        best_combo_global = {}

    return best_combo_global, per_unit_delta, combo_history
    # aggregate over anchor iterations -------------------------
    # per_unit_delta = {uid: float(np.nansum(rec["deltas"])) for uid, rec in unit_log.items()}
    # agg_lags = {uid: float(np.nanmedian(rec["lags"])) for uid, rec in unit_log.items()}
    # best_combo_global = {"lags": agg_lags, "score": float(np.nansum(list(per_unit_delta.values())))}

    # return best_combo_global, per_unit_delta, combo_history

# ───────────────────────────── acceptance & tuning ────────────────────────────

def _robust_median(x: Sequence[float]) -> float:
    return float(np.nanmedian(x)) if len(x) else np.nan


def _robust_mad(x: Sequence[float], med: float) -> float:
    x = np.asarray(x, float)
    return float(np.nanmedian(np.abs(x - med))) if len(x) else np.nan


def accumulate_unit_stats(unit_log: Dict[int, Dict]) -> Dict[int, Dict]:
    """Aggregate *unit_log* into per-unit statistics dictionary."""
    stats = {}
    for uid, rec in unit_log.items():
        d = np.asarray(rec["deltas"], float)            # ΔRMS per iteration
        L = np.asarray(rec["lags"],   float)            # lag or NaN

        pos_mask   = d > 0
        lag_mask   = ~np.isnan(L) & pos_mask            # only finite lags

        pos_delta  = d[pos_mask]
        good_lags  = L[lag_mask]

        stats[uid] = {
            "delta_sum" : float(np.nansum(pos_delta)),
            "delta_pos" : float(np.nansum(pos_delta)),
            "delta_neg" : float(np.nansum(d[d < 0])),
            "count_pos" : int(pos_mask.sum()),
            "count_neg" : int((d <= 0).sum()),
            "lag_med"   : np.nanmedian(good_lags) if good_lags.size else np.nan,
            "lag_mad"   : np.nanmedian(np.abs(good_lags - np.nanmedian(good_lags)))
                          if good_lags.size else np.inf      # treat “no data” as very inconsistent
        }
    return stats


def accept_units(
    stats: Dict[int, Dict],
    *,
    pos_min: float = 20.0,
    net_min: float = 10.0,
    h_max: float = 0.3,
    lag_mad_max: float = 2,
    lag_med_max: float = 38,
) -> Tuple[List[int], List[int]]:
    """Return (accepted_uids, rejected_uids) based on hard thresholds."""
    accepted = []
    for uid, s in stats.items():
        P = s["delta_pos"]
        N = s["delta_neg"]
        H = abs(N) / P if P else np.inf
        net = P + N
        if P >= pos_min and net >= net_min and H <= h_max and s["lag_mad"] <= lag_mad_max and s["lag_med"] <= lag_med_max:
            accepted.append(uid)
    rejected = [u for u in stats if u not in accepted]
    return accepted, rejected



def micro_align_units(
    accepted: Sequence[int],
    stats: Dict[int, Dict],
    unit_info: Dict[int, Dict],
    raw_snippet: np.ndarray,
    p2p_all: Dict[int, np.ndarray],   
    rms_raw,
    *,
    mask_thr: float = 5.0,
    beta: float = 0.5,
    micro_sweep: int = 2,
) -> Dict[int, int]:
    """Fine‑tune lags around median using ±*micro_sweep* neighbourhood."""
    final_lags: Dict[int, int] = {}
    final_deltas: Dict[int, int] = {}
    for uid in accepted:
        best_lag = int(round(stats[uid]["lag_med"]))
        best_score = -np.inf
        for d in range(-micro_sweep, micro_sweep + 1):
            lag = best_lag + d
            sel_ch = unit_info[uid]["selected_channels"]

            ei_full = unit_info[uid]["ei"].astype(np.float32)  # [512,121]
            rolled_bank = {
                (uid, lag): np.roll(ei_full, shift=lag, axis=1)
            }
            score = score_active_set({uid: lag}, sel_ch, raw_snippet[sel_ch], unit_info, p2p_all, rolled_bank, beta=beta, rms_raw=rms_raw)

            if score > best_score:
                best_score = score
                final_lags[uid] = lag
        
        final_deltas[uid] = best_score

    return final_lags, final_deltas

# ───────────────────────────── overlap subtraction ────────────────────────────

def subtract_overlap_tail(
    raw_next_snip: np.ndarray,  # [C,T]  (modified in‑place)
    accepted_prev: Dict[int, int],  # {uid: lag}
    unit_info: Dict[int, Dict],
    p2p_all: Dict[int, np.ndarray],
    *,
    overlap: int = 20,
    abs_thr: float = 2.0,
) -> np.ndarray:
    """Subtract tails of templates that spill into the next snippet window."""
    C, T = raw_next_snip.shape
    for uid, lag_prev in accepted_prev.items():
        ei = unit_info[uid]["ei"]
        tmpl = roll_zero(ei, lag_prev)  # aligned to *prev* origin
        start = tmpl.shape[1] - overlap
        end = start + T
        if start >= tmpl.shape[1]:
            continue
        tmpl_slice = tmpl[:, max(0, start) : min(end, tmpl.shape[1])]
        dst_start = max(0, -start)
        dst_end = dst_start + tmpl_slice.shape[1]
        chan_mask = p2p_all[uid] >= abs_thr
        raw_next_snip[chan_mask, dst_start:dst_end] -= tmpl_slice[chan_mask]
    return raw_next_snip


from collections import OrderedDict
from scipy.signal import correlate
import numpy as np

# ---------------------------------------------------------------------
def _peak_channel(ei):
    """index of channel with largest |P2P| in EI"""
    return int(np.argmax(ei.ptp(axis=1)))

def _best_lag(raw_chan, ei_chan,
              peak_sample=40, max_lag=6):
    """
    Dot-product lag search around ±max_lag
    so that the EI peak ends near `peak_sample`.
    """
    # full x-corr
    xcor  = correlate(raw_chan, ei_chan, mode='full')
    lags  = np.arange(-len(raw_chan) + 1, len(raw_chan))
    lag_raw = lags[np.argmax(xcor)]

    # keep EI peak inside window  (40 ± max_lag)
    p_idx   = np.argmax(np.abs(ei_chan))
    base    = len(raw_chan)//2 - p_idx          # shift that puts peak at centre
    lag_low = base - max_lag
    lag_hi  = base + max_lag
    return int(np.clip(lag_raw, lag_low, lag_hi))

# ---------------------------------------------------------------------
def marginal_gain(active_dict,
                  union_chans,
                  raw_local,
                  unit_info,
                  p2p_all,
                  *,
                  mask_thr=5.0,
                  beta=0.5,
                  max_lag=6,
                  peak_sample=40,
                  score_fn=score_active_set):
    """
    Return (uid, best_lag, gain) of the *first* unit that improves
    Δ-RMS when added to `active_dict`.  If none do, return None.
    """

    # --- 0. cache the current score -------------------------
    base_score = score_fn(active_dict, union_chans,
                          raw_local, unit_info, p2p_all,
                          mask_thr=mask_thr, beta=beta)

    # --- 1. iterate over all candidate units ----------------
    for uid in unit_info.keys():
        if uid in active_dict:
            continue

        ei      = unit_info[uid]["ei"]
        pch     = _peak_channel(ei)

        # skip if peak channel not inside this snippet
        if pch not in union_chans:
            continue
        c_idx    = union_chans.index(pch)

        # ----- find best lag on that channel -----
        lag = _best_lag(raw_local[c_idx], ei[pch],
                        peak_sample=peak_sample,
                        max_lag=max_lag)

        # ----- trial score with the unit added ---
        trial = OrderedDict(active_dict)
        trial[uid] = lag

        new_score = score_fn(trial, union_chans,
                             raw_local, unit_info, p2p_all,
                             mask_thr=mask_thr, beta=beta)
        
        if uid=='unit_14':
            print(uid)
            print(new_score, base_score)

        gain = new_score - base_score
        if gain > 0:
            print(uid, lag, gain)
            # return uid, lag, gain      # stop at first improvement

    # nothing helped
    return None



import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.mixture import GaussianMixture


from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.mixture import GaussianMixture
import numpy as np

def per_channel_gmm_bimodality(
    ei, snips, n_top=5, min_sep=2.0,
    include_ref40=True, win40=3, min_cluster_size=20,
    pool_ids=None  # optional: original indices for caching across rounds
):
    """
    For each channel, evaluate 2-GMM separation at multiple candidate times:
      - channel t_peak = argmax |ei[c]|
      - optional window around t_peak with step 2: t_peak ± {0,2,4,...,win40}
    Picks the best time per channel by separation and returns top n_top channels.

    Returns: list[dict] with keys:
      'sep'            : float separation (d'-style) used for ranking
      'chan','t'       : channel, sample
      'vmin','vmax'    : scalar range at (chan,t)
      'thr'            : decision threshold (midpoint of component means)
      'mu_lo','mu_hi'  : comp means (ordered by value)
      'std_lo','std_hi': comp stds
      'n_lo','n_hi'    : counts per side
      'polarity'       : sign of ei[chan,t]; candidate side follows this sign
      'cand_mask'      : boolean mask over CURRENT pool (selected component)
      'other_mask'     : boolean mask over CURRENT pool (complement)
      'cand_idx','other_idx'         : indices into CURRENT pool
      'cand_idx_global','other_idx_global' : indices into ORIGINAL pool (if pool_ids is given)
      'amp_lo','amp_hi','score_amp'  : p2p on this channel for each side (medians along spikes)
    """
    C, T, N = snips.shape
    out = []

    for c in range(C):
        t_peak = int(np.argmax(np.abs(ei[c])))
        cand_times = {t_peak}
        if include_ref40:
            for d in range(-win40, win40+1, 2):
                t = t_peak + d
                if 0 <= t < T:
                    cand_times.add(t)

        best = None  # (sep, t, vmin, vmax, labels, mu1, mu2, std1, std2)

        for t in cand_times:
            v = snips[c, t, :].reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
            try:
                gmm.fit(v)
            except Exception:
                continue

            labels = gmm.predict(v)                    # 0/1 per spike
            counts = np.bincount(labels, minlength=2)
            if counts.min() < min_cluster_size:
                continue

            mu = gmm.means_.flatten()
            std = np.array([np.sqrt(gmm.covariances_[k][0, 0]) for k in (0, 1)], dtype=float)

            # order comps by mean value: lo < hi
            order = np.argsort(mu)
            mu_lo, mu_hi = mu[order[0]], mu[order[1]]
            std_lo, std_hi = std[order[0]], std[order[1]]
            lab_lo = (labels == order[0])
            lab_hi = (labels == order[1])

            sep = abs(mu_hi - mu_lo) / np.sqrt(0.5 * (std_lo**2 + std_hi**2))

            if sep >= min_sep and (best is None or sep > best[0]):
                vflat = v.ravel()
                best = (
                    float(sep), int(t),
                    float(vflat.min()), float(vflat.max()),
                    lab_lo.copy(), lab_hi.copy(),
                    float(mu_lo), float(mu_hi),
                    float(std_lo), float(std_hi)
                )

        if best is None:
            continue

        sep, t_best, vmin, vmax, lab_lo, lab_hi, mu_lo, mu_hi, std_lo, std_hi = best

        # choose candidate side consistent with polarity of pooled EI at (c, t_best)
        pol = float(np.sign(ei[c, t_best]))
        if pol < 0:
            cand_mask = lab_lo
            other_mask = lab_hi
        else:
            cand_mask = lab_hi
            other_mask = lab_lo

        n_lo = int(lab_lo.sum()); n_hi = int(lab_hi.sum())
        cand_idx  = np.where(cand_mask)[0]
        other_idx = np.where(other_mask)[0]

        if pool_ids is not None:
            cand_idx_global  = pool_ids[cand_idx]
            other_idx_global = pool_ids[other_idx]
        else:
            cand_idx_global = None
            other_idx_global = None

        # per-channel medians for quick amplitude ranking (no full EI yet)
        wav_lo = np.median(snips[c, :, lab_lo], axis=1) if n_lo > 0 else np.zeros(T, dtype=np.float32)
        wav_hi = np.median(snips[c, :, lab_hi], axis=1) if n_hi > 0 else np.zeros(T, dtype=np.float32)
        amp_lo = float(wav_lo.max() - wav_lo.min())
        amp_hi = float(wav_hi.max() - wav_hi.min())
        score_amp = max(amp_lo, amp_hi)

        # simple decision threshold (midpoint of means)
        thr = 0.5 * (mu_lo + mu_hi)

        out.append(dict(
            sep=sep, chan=int(c), t=int(t_best),
            vmin=vmin, vmax=vmax,
            thr=float(thr),
            mu_lo=float(mu_lo), mu_hi=float(mu_hi),
            std_lo=float(std_lo), std_hi=float(std_hi),
            n_lo=n_lo, n_hi=n_hi,
            polarity=pol,
            cand_mask=cand_mask, other_mask=other_mask,
            cand_idx=cand_idx, other_idx=other_idx,
            cand_idx_global=cand_idx_global, other_idx_global=other_idx_global,
            amp_lo=amp_lo, amp_hi=amp_hi, score_amp=score_amp
        ))

    # keep the top n_top by separation (same behavior as before)
    out.sort(key=lambda d: -d['sep'])
    return out[:n_top]


def per_channel_gmm_bimodality_simple(ei, snips, n_top=5, min_sep=2.0,
                               include_ref40=True, win40=3, min_cluster_size=20):
    """
    For each channel, evaluate 2-GMM separation at multiple candidate times:
      - channel's own t_peak = argmax |ei[c]|
      - optional window [40-win40, 40+win40]
    Return top (sep, c, t, vmin, vmax).
    """
    C, T, N = snips.shape
    results = []
    for c in range(C):
        t_peak = int(np.argmax(np.abs(ei[c])))
        cand_times = {t_peak}
        if include_ref40:
            for d in range(-win40, win40+1, 2):
                t = t_peak + d
                if 0 <= t < T:
                    cand_times.add(t)

        best = None
        for t in cand_times:
            v = snips[c, t, :].reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
            try:
                gmm.fit(v)
            except Exception:
                continue
            labels = gmm.predict(v)
            counts = np.bincount(labels, minlength=2)
            if counts.min() < min_cluster_size:
                continue

            mu1, mu2 = gmm.means_.flatten()
            std1 = np.sqrt(gmm.covariances_[0][0, 0])
            std2 = np.sqrt(gmm.covariances_[1][0, 0])
            sep = abs(mu1 - mu2) / np.sqrt(0.5 * (std1**2 + std2**2))
            if sep >= min_sep and (best is None or sep > best[0]):
                best = (sep, c, t, float(v.min()), float(v.max()))
        if best is not None:
            results.append(best)

    results.sort(key=lambda x: -x[0])
    return results[:n_top]

import numpy as np
from compare_eis import compare_eis

def per_channel_hist_otsu_candidates(
    ei: np.ndarray,             # [C, T] pooled EI
    snips: np.ndarray,          # [C, T, N] snippets
    n_top: int = 5,
    *,
    sim_thr: float = 0.90,      # similarity above this → reject channel
    win40: int = 2,             # scan times {t_peak} ∪ {40±d, d<=win40}
    bins: int = 64,
    min_cluster_size: int = 10
):
    C, T, N = snips.shape
    results = []

    for c in range(C):
        # print(c)
        # candidate times: channel's t_peak and a window around 40
        t_peak = int(np.argmax(np.abs(ei[c])))
        cand_times = {t_peak}
        for d in range(-win40, win40 + 1):
            t = t_peak + d
            if 0 <= t < T:
                cand_times.add(t)

        best = None  # (score_amp, t_best, thr_best, n_lo, n_hi, mask_lo_best)

        for t in cand_times:
            v = snips[c, t, :].astype(np.float32)

            # robust range for histogram
            p1, p99 = np.percentile(v, [1, 99])
            if not np.isfinite(p1) or not np.isfinite(p99) or p1 >= p99:
                continue
            edges = np.linspace(p1, p99, bins + 1)
            hist, _ = np.histogram(v, bins=edges)
            if hist.sum() == 0:
                continue

            # Otsu threshold on bin midpoints
            p = hist.astype(np.float64) / hist.sum()
            mids = 0.5 * (edges[:-1] + edges[1:])
            w0 = np.cumsum(p)
            mu0 = np.cumsum(p * mids)
            muT = mu0[-1]
            w1 = 1.0 - w0
            denom = (w0 * w1) + 1e-12
            sigma_b2 = ((muT * w0 - mu0) ** 2) / denom

            # choose interior maximum
            if sigma_b2.size < 3:
                continue
            k = int(np.nanargmax(sigma_b2[1:-1])) + 1
            thr = mids[k]

            mask_lo = v <= thr
            n_lo = int(mask_lo.sum())
            n_hi = int((~mask_lo).sum())
            if min(n_lo, n_hi) < min_cluster_size:
                continue

            # channel-local median waveforms & p2p amplitudes for ranking
            wav_lo = np.mean(snips[c, :, mask_lo], axis=1)
            wav_hi = np.mean(snips[c, :, ~mask_lo], axis=1)
            amp_lo = float(wav_lo.max() - wav_lo.min())
            amp_hi = float(wav_hi.max() - wav_hi.min())
            score_amp = max(amp_lo, amp_hi)

            # keep best time for this channel by larger-cluster p2p
            # if best is not None:
            #     print(f"Channel {c}, thr: {thr:0.2f}, current best {best[2]:0.2f}")
            #     print(thr<best[2])
            if (best is None) or (score_amp < best[0]):
                best = (score_amp, t, thr, n_lo, n_hi, mask_lo.copy())

        # If no valid split for this channel, skip
        if best is None:
            continue

        score_amp, t_best, thr_best, n_lo, n_hi, mask_lo_best = best

        # # Build full EIs for similarity veto (your compare_eis, lag-tolerant)
        # ei1 = np.mean(snips[:, :, mask_lo_best], axis=2) if n_lo > 0 else None
        # ei2 = np.mean(snips[:, :, ~mask_lo_best], axis=2) if n_hi > 0 else None
        # if ei1 is None or ei2 is None:
        #     continue

        # sim_mat = compare_eis([ei1, ei2], max_lag=2)
        # sim_val = float(sim_mat[0][1])  # similarity between the two EIs

        # # If templates look the same (amp-only split), reject channel
        # if sim_val > sim_thr:
        #     continue

        # Keep candidate: sorted later by score_amp
        results.append((score_amp, c, t_best, float(thr_best), n_lo, n_hi))

    # Rank by amplitude of larger cluster on its channel (desc) and return top-N
    results.sort(key=lambda x: x[3])

    # After filling results
    # print(f"{'Score':>10} {'Chan':>6} {'t_best':>8} {'Thr':>8} {'n_lo':>6} {'n_hi':>6}")
    # for score_amp, c, t_best, thr_best, n_lo, n_hi in results:
    #     print(f"{score_amp:10.3f} {c:6d} {t_best:8d} {thr_best:8.3f} {n_lo:6d} {n_hi:6d}")


    return results[:n_top]


# def per_channel_gmm_bimodality(ei, snips, n_top=5, min_sep=2.0):
#     """
#     Estimates bimodality via 2-Gaussian GMM fit per channel (at peak sample).
#     Scores by mean separation in units of pooled std.

#     Arguments:
#       ei      : [C,T] – pooled EI
#       snips   : [C,T,N] – raw snippets
#       n_top   : how many to show
#       min_sep : only report channels with clear separation (like d-prime > X)

#     Output:
#       Prints top channels with strongest bimodality
#     """
#     C, T, N = snips.shape
#     results = []
#     for c in range(C):
#         t_peak = np.argmax(np.abs(ei[c]))
#         v = snips[c, t_peak, :].reshape(-1, 1)

#         gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
#         try:
#             gmm.fit(v)
#         except:
#             continue

#         # Predict cluster assignments
#         labels = gmm.predict(v)
#         counts = np.bincount(labels)
#         if counts.min() < 20:
#             continue  # skip channels with undersized cluster

#         # Mean and std for separation score
#         mu1, mu2 = gmm.means_.flatten()
#         std1 = np.sqrt(gmm.covariances_[0][0, 0])
#         std2 = np.sqrt(gmm.covariances_[1][0, 0])
#         sep = abs(mu1 - mu2) / np.sqrt(0.5 * (std1**2 + std2**2))

#         results.append((sep, c, t_peak, v.min(), v.max()))

#     results.sort(key=lambda x: -x[0])

#     # print(f"\nTop {n_top} channels by GMM bimodality score:")
#     # print("  chan  sample   sep      min     max")
#     # for sep, c, t, vmin, vmax in results[:n_top]:
#     #     if sep < min_sep:
#     #         break
#     #     print(f"  {c:4d}   {t:4d}   {sep:6.2f}   {vmin:7.1f}  {vmax:7.1f}")

#     return results

    
import numpy as np
import matplotlib.pyplot as plt

def compute_global_baseline_mean(ei,
                                 early=20,
                                 late=20,
                                 cap_val=50,
                                 make_plots=False,
                                 baseline_thresh=10):
    """
    Fast vectorised baseline-quality metric.
    Returns global_baseline_mean (float).
    """
    n_ch, n_t = ei.shape

    # ------------------------------------------------------------------
    # 1) channel-wise peak-to-peak and dominant-peak location
    # ------------------------------------------------------------------
    p2p         = np.ptp(ei, axis=1)

    pos_idx     = np.argmax(ei, axis=1)          # index of + peak
    neg_idx     = np.argmin(ei, axis=1)          # index of – peak
    pos_val_abs = ei[np.arange(n_ch), pos_idx]
    neg_val_abs = np.abs(ei[np.arange(n_ch), neg_idx])

    use_pos     = pos_val_abs > neg_val_abs
    peak_idx    = np.where(use_pos, pos_idx, neg_idx)   # (n_ch,)

    # ------------------------------------------------------------------
    # 2) baseline windows – vectorised
    # ------------------------------------------------------------------
    early_ok = peak_idx >= early
    late_ok  = peak_idx <  (n_t - late)

    # pre-compute means & maxima across the fixed windows
    early_block_mean = ei[:, :early].mean(axis=1)
    early_block_max  = ei[:, :early].max(axis=1)
    late_block_mean  = ei[:, -late:].mean(axis=1)
    late_block_max   = ei[:, -late:].max(axis=1)

    # keep only channels where the window is valid, else NaN
    early_mean = np.where(early_ok, early_block_mean, np.nan)
    early_max  = np.where(early_ok, early_block_max,  np.nan)
    late_mean  = np.where(late_ok,  late_block_mean,  np.nan)
    late_max   = np.where(late_ok,  late_block_max,   np.nan)

    # ------------------------------------------------------------------
    # 3) global metrics
    # ------------------------------------------------------------------
    abs_early_mean = np.abs(early_mean)
    abs_late_mean  = np.abs(late_mean)

    bad_baseline_ch = (abs_early_mean > baseline_thresh) | \
                      (abs_late_mean  > baseline_thresh)
    n_bad_channels  = np.count_nonzero(bad_baseline_ch)

    # combine valid baseline means & maxima
    baseline_means_all = np.concatenate([abs_early_mean[np.isfinite(early_mean)],
                                         abs_late_mean [np.isfinite(late_mean)]])
    baseline_max_all   = np.concatenate([np.abs(early_max[np.isfinite(early_max)]),
                                         np.abs(late_max [np.isfinite(late_max)])])

    global_baseline_mean     = baseline_means_all.mean()
    global_baseline_max_mean = baseline_max_all.mean()

    # ------------------------------------------------------------------
    # 4) optional plots (unchanged except for variable names)
    # ------------------------------------------------------------------
    if make_plots:
        fig, axs = plt.subplots(2, 2, figsize=(18, 6), sharex=True)
        axs = axs.ravel()

        axs[0].plot(p2p, color='black')
        axs[0].set_title('P2P amplitude')
        axs[0].set_ylabel('Amplitude')

        axs[1].plot(peak_idx, color='blue')
        axs[1].set_title('Location of dominant peak')
        axs[1].set_ylabel('Sample')

        axs[2].plot(early_mean, label='Early mean', color='green')
        axs[2].plot(late_mean,  label='Late mean',  color='orange')
        axs[2].set_title('Baseline mean')
        axs[2].set_ylim(-cap_val, cap_val)
        axs[2].legend()

        axs[3].plot(early_max, label='Early max', color='green')
        axs[3].plot(late_max,  label='Late max',  color='orange')
        axs[3].set_title('Baseline max')
        axs[3].set_ylim(-cap_val, cap_val)
        axs[3].legend()

        for ax in axs:
            ax.set_xlabel('Channel')
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    return global_baseline_mean


def ei_similarity(ei_a, ei_b):
    """
    Cosine similarity (or your existing compare_ei) – range [0..1].
    Higher ⇒ more alike.
    """
    a = ei_a.ravel(); b = ei_b.ravel()
    return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9)

import numpy as np

def median_ei_adaptive(snips, base=500):
    """
    Compute EI as the median of an adaptively sub-sampled set of spikes.

    snips : ndarray  [C, T, N]
        Raw snippets (channels × time × spikes)

    base  : int
        Use every spike when N ≤ base.
        For every additional `base` spikes, stride increases by 1.
        i.e. stride = 1 + (N-1)//base

    Returns
    -------
    ei_med : ndarray  [C, T]
        Median EI computed on the sub-sampled snippets.

    Notes
    -----
    * N =  500  → stride = 1   (all spikes)
    * N =  750  → stride = 2   (every 2nd spike)
    * N = 1500  → stride = 3   (every 3rd spike)
    """
    N = snips.shape[2]
    stride = 1 + (N - 1) // base          # adaptive stride
    ei_med = np.median(snips[:, :, ::stride], axis=2)
    return ei_med.astype(snips.dtype, copy=False)


# ---------------------------------------------------------------------
# ---  main recursive splitter  ---------------------------------------
# ---------------------------------------------------------------------
def verify_clusters_recursive(spike_times,
                              snips_raw,          # [C, T, N]
                              params,
                              target_chan,
                              abs_idx=None,          # ← NEW
                              depth=0):
    """
    Returns list of dicts with keys:
        'inds'  – indices into *spike_times*
        'ei'    – median EI [C, T]
    """
    import time

    if abs_idx is None:                      # top-level entry
        abs_idx = np.arange(spike_times.size)

    min_spikes     = params.get('min_spikes', 10)
    baseline_cut   = params.get('baseline_cut', 2.0)
    ei_sim_thr     = params.get('ei_sim_threshold', 0.9)
    k_start        = params.get('k_start', 6)
    k_refine       = params.get('k_refine', 3)

    N = spike_times.size
    if N < min_spikes:
        return []

    # -----------------------------------------------------------------
    # 1) parent EI + quick pruning
    # -----------------------------------------------------------------
    print(f"Depth: {depth}")

    start = time.perf_counter()
    ei_parent = median_ei_adaptive(snips_raw)
    elapsed = time.perf_counter() - start
    # print(f"median,   elapsed = {elapsed*1e3:.3f} ms")
    peak_ch   = np.argmin(ei_parent.min(axis=1))

    ei_parent = np.ascontiguousarray(ei_parent, dtype=np.float32)   # ensure RAM & contiguous

    gbm_parent = compute_global_baseline_mean(ei_parent)
    print(f"          parent gmb: {gbm_parent}")

    # if parent already fails basic checks, abort early
    if peak_ch != target_chan or gbm_parent >= baseline_cut:
        return []

    # -----------------------------------------------------------------
    # 2) split this cluster once
    # -----------------------------------------------------------------
    k_split = k_start if depth == 0 else k_refine
    k_split = max(2, min(k_split, N // min_spikes))  # guard rails
    if k_split < 2:
        # not enough spikes to split
        return [{'inds': abs_idx, 'ei': ei_parent}]

    # vectorise snippets for PCA
    C, T = snips_raw.shape[:2]
    p2p = ei_parent.ptp(axis=1)                    # C-vector in ADC counts
    p2p_thresh = params.get('p2p_thresh_adc', 50)

    chan_sel = np.where(p2p >= p2p_thresh)[0]     # channels above threshold
    if chan_sel.size > 80:
        # keep the 80 strongest by p2p
        chan_sel = chan_sel[np.argsort(p2p[chan_sel])[-80:]]
    elif chan_sel.size < 10:
        # ensure at least 10 channels (take strongest ones)
        chan_sel = np.argsort(p2p)[-10:]

    # slice snippets for PCA: [C_sel, T, N] → reshape to N × (C_sel·T)
    snips_sel = snips_raw[chan_sel, :, :]          # view, no copy
    C_sel = snips_sel.shape[0]
    X = snips_sel.reshape(C_sel*T, N).T            # N × (C_sel*T)


    n_comp = min(7, X.shape[1]-1)
    Xred = PCA(n_components=n_comp, svd_solver='randomized').fit_transform(X)

    labels = KMeans(k_split, n_init=5, random_state=depth).fit(Xred).labels_

    # -----------------------------------------------------------------
    # 3) child filtering (gbm & channel) -------------------------------
    # -----------------------------------------------------------------
    survivors = []
    for lab in np.unique(labels):
        mask = labels == lab
        if mask.sum() < min_spikes:
            continue

        snips_c = snips_raw[:, :, mask]
        ei_c    = median_ei_adaptive(snips_c)
        pk_c    = np.argmin(ei_c.min(axis=1))
        if pk_c != target_chan:
            continue

        gbm_c = compute_global_baseline_mean(ei_c)
        if gbm_c >= baseline_cut:
            continue

        print(f"            child gmb: {gbm_c}")

        survivors.append({'mask': mask,
                          'ei':   ei_c,
                          'gbm':  gbm_c})

    if not survivors:
        # nothing passed – keep parent as best we have
        return [{'inds': abs_idx, 'ei': ei_parent}]

    # -----------------------------------------------------------------
    # 4) merge still-similar children ----------------------------------
    # -----------------------------------------------------------------

    merged = []
    taken  = np.zeros(len(survivors), dtype=bool)

    for i, s_i in enumerate(survivors):
        if taken[i]:
            continue
        union_mask = s_i['mask'].copy()
        for j, s_j in enumerate(survivors[i+1:], start=i+1):
            if taken[j]:
                continue
            if ei_similarity(s_i['ei'], s_j['ei']) > ei_sim_thr:
                print(f"Merged some, sim {ei_similarity(s_i['ei'], s_j['ei']):0.2f}")
                union_mask |= s_j['mask']
                taken[j] = True

        snips_u = snips_raw[:, :, union_mask]
        merged.append({
            'mask': union_mask,
            'ei':   median_ei_adaptive(snips_u),
            'abs':  abs_idx[union_mask]          # ← keep absolute indices
        })


    # -----------------------------------------------------------------
    # 5) decide recurse vs. return ------------------------------------
    # -----------------------------------------------------------------
    if len(merged) == 1:
        return [{'inds': merged[0]['abs'],      # <-- use saved absolute indices
                'ei':   merged[0]['ei']}]

    # otherwise recurse on each surviving child
    out = []
    for child in merged:
        idxs   = np.where(child['mask'])[0]
        st_sub = spike_times[idxs]
        sr_sub = snips_raw[:, :, child['mask']]
        abs_sub = abs_idx[idxs]                      # ← keeps original indices
        out.extend(
            verify_clusters_recursive(st_sub,
                                      sr_sub,
                                      params,
                                      target_chan,
                                      abs_idx=child['abs'],          # pass absolute indices
                                      depth=depth+1)
        )
    return out





from scipy.signal import correlate, correlation_lags

def xcorr_spike_times(st1, st2, max_lag=None, total_len=None):
    """
    MATLAB-style cross-correlation of two spike time lists (binary event vectors).
    
    Parameters
    ----------
    st1, st2 : array-like
        Spike times (sample indices).
    max_lag : int or None
        If given, restrict lags to [-max_lag, max_lag].
    total_len : int or None
        Total duration of recording (vector length). 
        If None, inferred from max spike time + 1.

    Returns
    -------
    lags : ndarray
        Array of lag values.
    cc_norm : ndarray
        Normalized cross-correlation coefficients (like MATLAB xcorr(...,'coeff')).
    """

    st1 = np.asarray(st1, dtype=int)
    st2 = np.asarray(st2, dtype=int)

    if total_len is None:
        total_len = max(st1.max() if st1.size > 0 else 0,
                        st2.max() if st2.size > 0 else 0) + 1

    vec1 = np.zeros(total_len, dtype=np.float32)
    vec2 = np.zeros(total_len, dtype=np.float32)
    vec1[st1] = 1
    vec2[st2] = 1

    cc_full = correlate(vec2, vec1, mode='full')  # y,x like MATLAB
    lags = correlation_lags(len(vec2), len(vec1), mode='full')

    norm = np.sqrt(np.sum(vec1 ** 2) * np.sum(vec2 ** 2))
    cc_norm = cc_full / norm if norm > 0 else cc_full * 0

    if max_lag is not None:
        keep = np.abs(lags) <= max_lag
        lags = lags[keep]
        cc_norm = cc_norm[keep]

    return lags, cc_norm




# import numpy as np
# from collections import OrderedDict

# def _build_data_mask(raw_local, thresh_sigma=4.0):
#     """Boolean mask [C,T] that is *constant* for all tests."""
#     C, T = raw_local.shape
#     mask = np.zeros_like(raw_local, dtype=bool)
#     for c in range(C):
#         sigma = np.median(np.abs(raw_local[c])) / 0.6745
#         mask[c] = np.abs(raw_local[c]) > thresh_sigma * sigma
#     return mask

# def _peak_channel(ei):
#     return int(np.argmax(np.abs(ei).ptp(axis=1)))

# def _best_lag(raw_chan, ei_chan, peak_sample=40, max_lag=6):
#     """Return lag (int) giving best dot‐product alignment."""
#     search = range(-max_lag, max_lag + 1)
#     best, best_lag = -np.inf, 0
#     # centre EI at its peak (40 by convention)
#     ref = ei_chan.copy()
#     for L in search:
#         if L < 0:
#             seg_raw = raw_chan[peak_sample+L : peak_sample+L+len(ref)]
#         else:
#             seg_raw = raw_chan[peak_sample-L : peak_sample-L+len(ref)]
#         if seg_raw.shape != ref.shape:
#             continue
#         dot = np.dot(seg_raw, ref)
#         if dot > best:
#             best, best_lag = dot, L
#     return best_lag

# def _score_delta(active_dict, union_chans,
#                  raw_local, unit_info, p2p_all,
#                  mask, ei_positions, beta=0.5):
#     tmpl_sum = np.zeros_like(raw_local, dtype=np.float32)
#     for u, lag in active_dict.items():
#         ei = unit_info[u]["ei"]
#         shifted = np.array([np.roll(ei[c], lag) for c in union_chans])
#         tmpl_sum[union_chans,:] += shifted

#     delta = 0.0
#     for k, c in enumerate(union_chans):
#         m = mask[k]
#         if not m.any():
#             continue
#         rms_raw = np.sqrt((raw_local[k, m] ** 2).mean())
#         rms_res = np.sqrt(((raw_local[k, m] - tmpl_sum[k, m]) ** 2).mean())
#         # a_max = max(p2p_all[u][c] for u in active_dict)
#         # delta += (a_max ** beta) * (rms_raw - rms_res)
#         delta +=  (rms_raw - rms_res)

#     if delta>0:
#         from axolotl_utils_ram import plot_ei_waveforms

#         plt.figure(figsize=(25, 12))
#         plot_ei_waveforms(
#             [raw_local, tmpl_sum],
#             ei_positions,
#             scale=70.0,
#             box_height=1.0,
#             box_width=50.0,
#             colors=['gray', 'red', 'cyan']
#         )
#         plt.title(f"{active_dict}, delta {delta}")
#         plt.show()
#     return delta

# # ────────────────────────────────────────────────────────────────────
# def find_first_missing_unit(raw_local,
#                             union_chans,
#                             final_lags,     # {uid: lag}
#                             all_units,          # iterable of uid
#                             unit_info,
#                             p2p_all,
#                             ei_positions,
#                             beta=0.5,
#                             max_lag=6
#                             ):
#     """
#     Returns (uid, best_lag, gain) or None.
#     """
#     mask = _build_data_mask(raw_local)
#     base_score = _score_delta(final_lags, union_chans,
#                               raw_local, unit_info, p2p_all,
#                               mask, ei_positions, beta)

#     for uid in all_units:
#         if uid in final_lags:
#             continue

#         ei     = unit_info[uid]["ei"]
#         pch    = _peak_channel(ei)
#         c_idx  = union_chans.index(pch) if pch in union_chans else None
#         if c_idx is None:                      # peak channel not in window
#             continue


#         trace_rw = raw_local[pch]
#         trace_ei = ei[pch]

#         # raw lag from full x-corr
#         xcor   = correlate(trace_rw, trace_ei, mode="full")
#         lags   = np.arange(-len(trace_rw) + 1, len(trace_rw))
#         lag_raw = lags[np.argmax(xcor)]

#         snip_len = raw_local.shape[1]
#         # ----- asymmetric clipping so EI peak lands in 40-80 -----
#         p_idx     = np.argmax(np.abs(trace_ei))        # or store once per unit
#         base      = snip_len//2 - p_idx                # 19 when p_idx=41
#         lag_low   = base - max_lag                     # -1
#         lag_high  = base + max_lag                     # 39
#         lag      = int(np.clip(lag_raw, lag_low, lag_high))


#         # lag = _best_lag(raw_local[c_idx], ei[pch],
#         #                 peak_sample=peak_sample,
#         #                 max_lag=max_lag)

#         trial = OrderedDict(final_lags)
#         trial[uid] = lag

#         new_score = _score_delta(trial, union_chans,
#                                  raw_local, unit_info, p2p_all,
#                                  mask, ei_positions,beta)
#         gain = new_score - base_score
#         if gain > 0:          # improves ΔRMS
#             print(f">>> Candidate {uid}: lag={lag:+d},  gain={gain:.2f}")
#             return uid, lag, gain   # comment this 'return' to collect all

#     print("No missing unit improves the fit.")
#     return None



import numpy as np
import matplotlib.pyplot as plt

# ---------- helpers ----------
import numpy as np

def select_template_channels(
    ei, p2p_thr=50.0, max_n=80, min_n=10, force_include_main=True
):
    """
    Return (channels, p2p) where channels are sorted by descending p2p.
    Ensures at least min_n channels by adding sub-threshold channels with highest p2p.
    """
    ptp = ei.max(axis=1) - ei.min(axis=1)        # [C]
    C = ptp.size

    # strong channels (≥ thr), sorted by descending p2p
    strong = np.flatnonzero(ptp >= p2p_thr)
    strong = strong[np.argsort(ptp[strong])[::-1]]

    # optional: main negative-trough channel
    if force_include_main:
        ch_main = int(np.argmin(ei.min(axis=1)))
    else:
        ch_main = None

    # start with strong; ensure main present
    picked = list(strong)
    seen = set(picked)
    if ch_main is not None and ch_main not in seen:
        picked.append(ch_main); seen.add(ch_main)

    # top up to min_n using global top-p2p order (avoiding duplicates)
    order_all = np.argsort(ptp)[::-1]            # all channels high→low p2p
    for ch in order_all:
        if len(picked) >= min_n:
            break
        if ch not in seen:
            picked.append(ch); seen.add(ch)

    # cap at max_n and sort by p2p desc
    picked = sorted(picked, key=lambda c: ptp[c], reverse=True)[:max_n]

    # safety: if array unusually small, still return something coherent
    if len(picked) == 0:
        picked = order_all[:min(min_n, C)].tolist()

    return np.asarray(picked, dtype=int), ptp


def main_channel_and_neg_peak(ei):
    """Main channel = most negative trough; return (channel_index, t_neg)."""
    mins = ei.min(axis=1)                    # [C]
    ch_main = int(np.argmin(mins))
    t_neg = int(np.argmin(ei[ch_main]))
    return ch_main, t_neg

def roll_zero_2d(arr, shift):
    """
    Zero-padded shift along time axis for 2D [nch, T].
    Positive shift -> moves waveform to the right.
    """
    nch, T = arr.shape
    out = np.zeros_like(arr)
    if shift == 0:
        out[:] = arr
    elif shift > 0:
        out[:, shift:] = arr[:, :T-shift]
    else:
        s = -shift
        out[:, :T-s] = arr[:, s:]
    return out

# ---------- core computation ----------
def compute_harm_map_noamp(
    ei,                   # [C, T] template
    snips,                # [C, T, N] raw snippets for many spikes
    p2p_thr=30.0,
    max_channels=80,
    min_channels=10,
    lag_radius=4,         # ±3 samples
    weight_by_p2p=True,
    weight_beta=0.7,      # w_c ∝ (p2p)^beta
    channels_override=None,
    force_include_main=True,
):

    C, T = ei.shape
    assert snips.shape[:2] == (C, T), "snips must be [C,T,N]"
    N = snips.shape[2]

    # 1) channel selection
    # 1) channel selection
    if channels_override is not None:
        # accept the provided channel list as-is (order preserved, de-duped, in-bounds)
        ch_arr = np.array(list(channels_override), dtype=int).ravel()
        C = ei.shape[0]
        ch_arr = ch_arr[(ch_arr >= 0) & (ch_arr < C)]

        # de-duplicate while preserving order
        seen = set()
        chans_list = []
        for c in ch_arr:
            if int(c) not in seen:
                seen.add(int(c))
                chans_list.append(int(c))
        # ensure main channel present if requested
        ch_main, t_neg = main_channel_and_neg_peak(ei)
        if force_include_main and (ch_main not in seen):
            chans_list.append(int(ch_main))

        chans = np.asarray(chans_list, dtype=int)
        # full-array p2p (used later for weights and for 'channel_ptp' in output)
        ptp = (ei.max(axis=1) - ei.min(axis=1)).astype(ei.dtype, copy=False)
    else:
        # default behavior: P2P-based selection with optional force-include
        chans, ptp = select_template_channels(ei, p2p_thr, max_channels, min_channels)
        if force_include_main:
            ch_main, t_neg = main_channel_and_neg_peak(ei)
            if ch_main not in chans:
                # ensure main channel is present; replace last channel
                chans = np.concatenate([chans[:-1], [ch_main]])
                # keep unique and re-order by p2p descending
                chans = np.array(sorted(set(chans), key=lambda c: ptp[c], reverse=True), dtype=int)
        else:
            ch_main, t_neg = main_channel_and_neg_peak(ei)


    ei_sel = ei[chans]                 # [nch, T]
    raw_sel = snips[chans]             # [nch, T, N]
    nch = ei_sel.shape[0]

    # 2) lag bank
    lags = np.arange(-lag_radius, lag_radius + 1, dtype=int)   # [L]
    L = lags.size

    # 3) precompute RMS(raw) per channel/spike (no mask)
    rms_raw = np.sqrt((raw_sel ** 2).mean(axis=1))             # [nch, N]

    E_raw = rms_raw ** 2                                       # [nch, N]
    _eps = 1e-12                                               # avoid divide-by-zero

    # 4) p2p weights (optionally tempered)
    if weight_by_p2p:
        w = (ptp[chans] ** weight_beta).astype(np.float32)
    else:
        w = np.ones(nch, dtype=np.float32)
    w /= w.sum()

    # 5) scan lags (vectorized over all spikes)
    deltas = np.empty((L, nch, N), dtype=np.float32)           # ΔRMS per lag
    ratios = np.empty((L, nch, N), dtype=np.float32)           # E_res / E_raw per lag (unitless)

    for i, d in enumerate(lags):
        shifted = roll_zero_2d(ei_sel, d)                      # [nch, T]
        resid   = raw_sel - shifted[:, :, None]                # [nch, T, N]
        E_res   = (resid ** 2).mean(axis=1)                    # [nch, N]
        rms_res = np.sqrt(E_res)                               # [nch, N]
        deltas[i] = rms_res - rms_raw                          # [nch, N]  (kept for backward compat)
        ratios[i] = E_res / np.maximum(E_raw, _eps)            # [nch, N]  (unitless)


    # 6) pick best lag per spike by (weighted) mean ΔRMS across channels
    mean_deltas = (deltas * w[:, None]).sum(axis=1)            # [L, N] weighted by channels
    best_i = np.argmin(mean_deltas, axis=0)                    # [N]
    best_lags = lags[best_i]                                   # [N]

    # 7) assemble harm matrix at chosen lag per spike (channels × spikes), vectorized
    # Ensure deltas is (L, nch, N); if you accidentally built (L, N, nch), fix it:
    if deltas.shape == (L, N, nch):
        deltas = deltas.transpose(0, 2, 1)
    elif deltas.shape != (L, nch, N):
        raise AssertionError(f"Unexpected deltas shape: {deltas.shape}")
    
    # Ensure ratios is (L, nch, N) too
    if ratios.shape == (L, N, nch):
        ratios = ratios.transpose(0, 2, 1)
    elif ratios.shape != (L, nch, N):
        raise AssertionError(f"Unexpected ratios shape: {ratios.shape}")


    # Gather per-spike best lag in one shot → harm: (nch, N)
    harm = np.take_along_axis(deltas, best_i[None, None, :], axis=0).squeeze(0)

    # Normalized energy ratio at chosen lag, and R²-like improvement
    harm_ratio = np.take_along_axis(ratios, best_i[None, None, :], axis=0).squeeze(0)  # [nch, N]
    harm_r2    = 1.0 - harm_ratio                                                       # [nch, N]
    mean_r2_unweighted = harm_r2.mean(axis=0)                                           # [N]
    mean_r2_weighted   = (harm_r2 * w[:, None]).sum(axis=0)                             # [N]


    # Per-spike summaries
    mean_delta_unweighted = harm.mean(axis=0)                  # [N]
    mean_delta_weighted   = (harm * w[:, None]).sum(axis=0)    # [N]
    p2p_sel = ptp[chans]                                       # [nch]

    out = {
        "selected_channels": chans,
        "channel_ptp": p2p_sel,
        "main_channel": ch_main,
        "neg_peak_index": t_neg,
        "lags": lags,
        "best_lag_per_spike": best_lags,
        "harm_matrix": harm,                        # [nch, N]
        "mean_delta_unweighted": mean_delta_unweighted,
        "mean_delta_weighted": mean_delta_weighted,
        "harm_matrix_r2": harm_r2,                      # [nch, N], unitless, upper-bounded by 1
        "mean_r2_unweighted": mean_r2_unweighted,       # [N]
        "mean_r2_weighted":   mean_r2_weighted,         # [N]
        "selection_metric":   "delta_rms",              # documents how best_i was chosen

    }
    return out


def scan_continuous_harm_regions(
    raw_data: np.ndarray,
    final_ei_full: np.ndarray,
    *,
    start_sample: int = 0,
    stop_sample: int = None,      # exclusive
    channels=None,
    mean_thr: float = -2.0,
    chan_thr: float = 10.0,
    ref_thr: float = -5.0,
    center_index: int = 40,
    region_max_len: int = 20,
    amp_ratio_min: float = None,   # NEW: e.g., 0.5 → keep if ≥ 0.5× template
    amp_ratio_max: float = None,   # NEW: e.g., 1.5 → keep if ≤ 1.5× template
):
    """
    Continuous harm-map scan on intact raw data with a fixed EI (no amplitude fit).

    Scans raw samples in [start_sample, stop_sample) and finds contiguous regions
    (max length = region_max_len) where per-sample ΔRMS gates pass. For each region,
    returns edges (absolute raw samples), the single best sample, and the ΔRMS vector
    across the selected channels at that best sample.

    Parameters
    ----------
    raw_data : (C, L)
        Continuous raw voltage (μV). NOT mutated.
    final_ei_full : (C, T)
        EI to test; may be off-centered. We re-center so the most negative trough on the
        strongest (most negative) channel is at sample `center_index`.
    start_sample : int
        Absolute raw sample at which to begin scanning (inclusive).
    stop_sample : int or None
        Absolute raw sample at which to stop scanning (**exclusive**).
        If None, scans to the maximal valid sample given the EI length and center_index.
    channels : sequence[int] or None
        If provided, restrict to these channels. Otherwise auto-select:
        {ref} ∪ {ch: EI_RMS ≥ chan_thr}.
    mean_thr, chan_thr, ref_thr : float
        Gates per raw sample:
          • mean ΔRMS across selected channels ≤ mean_thr
          • any channel ΔRMS > chan_thr → reject
          • ref-channel ΔRMS < ref_thr
    center_index : int
        Index within the EI that is aligned to the tested raw sample (default 40).
    region_max_len : int
        Split any continuous pass run into chunks of at most this many samples.
    amp_ratio_min / amp_ratio_max : float or None
        Optional ref-channel amplitude ratio gate. If provided, keep only regions where
        (raw trough @ ref_channel) / (template trough @ ref_channel) ∈ [min, max].

    Returns
    -------
    dict with:
      - regions: list of {start, end, best_idx, mean_delta, ref_delta, amp_ratio_ref?}
                 (absolute raw samples)
      - harm_at_best: (n_selected_channels, n_regions)  ΔRMS at each region’s best sample
      - selected_channels: (n_sel,) channel indices used
      - ref_channel: int
      - center_index: int
    """
    C, T = final_ei_full.shape
    assert raw_data.shape[0] == C, "raw_data and EI must share channel count"

    # 0) Re-center EI (zero-padded), ref = most negative trough channel
    trough_vals_all = final_ei_full.min(axis=1)
    ref_channel = int(np.argmin(trough_vals_all))
    trough_idx  = int(np.argmin(final_ei_full[ref_channel]))
    shift = int(center_index - trough_idx)
    ei = roll_zero_all(final_ei_full, shift)  # [C, T], zero-padded roll

    trough_vals = ei.min(axis=1)
    ref_channel = int(np.argmin(trough_vals))
    ref_wave    = ei[ref_channel]
    E_T_ref     = float(np.sum(ref_wave**2))
    ei_rms      = np.sqrt(np.mean(ei**2, axis=1))  # per-channel EI RMS

    # Template trough magnitude at ref channel (for amplitude ratio)
    tpl_ref_trough = float(np.abs(np.min(ref_wave))) + 1e-12

    # 1) Valid raw-sample range for testing
    L = raw_data.shape[1]
    min_valid = center_index
    max_valid = L - (T - center_index) - 1  # inclusive
    if max_valid < min_valid:
        return dict(regions=[], harm_at_best=np.zeros((0,0), dtype=float),
                    selected_channels=np.zeros((0,), dtype=int),
                    ref_channel=ref_channel, center_index=center_index)

    s0 = max(int(start_sample), int(min_valid))
    sN = (max_valid + 1) if (stop_sample is None) else min(int(stop_sample), max_valid + 1)
    if sN <= s0:
        return dict(regions=[], harm_at_best=np.zeros((0,0), dtype=float),
                    selected_channels=np.zeros((0,), dtype=int),
                    ref_channel=ref_channel, center_index=center_index)

    # These are the absolute raw samples we test
    test_samples = np.arange(s0, sN, dtype=np.int64)
    n_tests = test_samples.size

    # Slice raw once to cover all needed snippet spans implicitly
    snippet_starts = test_samples - center_index
    base_start = int(snippet_starts[0])
    raw_slice = raw_data[:, base_start : base_start + n_tests + T - 1]  # [C, n_tests+T-1]

    # 2) Stage 1: ref-channel ΔRMS across full test span (no snippets)
    x_ref = raw_slice[ref_channel].astype(np.float64, copy=False)
    x2 = x_ref * x_ref
    csum = np.empty(x2.size + 1, dtype=np.float64); csum[0] = 0.0; np.cumsum(x2, out=csum[1:])
    rel_starts = np.arange(0, n_tests, dtype=np.int64)
    E_x_ref = csum[rel_starts + T] - csum[rel_starts]                # [n_tests]
    r_ref   = np.correlate(x_ref, ref_wave, mode='valid')            # [n_tests]
    sqrt_Ex = np.sqrt(E_x_ref / T)
    delta_ref = np.sqrt((E_x_ref + E_T_ref - 2.0 * r_ref) / T) - sqrt_Ex

    pass_ref = (delta_ref < ref_thr)
    if not np.any(pass_ref):
        return dict(regions=[], harm_at_best=np.zeros((0,0), dtype=float),
                    selected_channels=np.zeros((0,), dtype=int),
                    ref_channel=ref_channel, center_index=center_index)

    def _runs_to_regions(mask: np.ndarray):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts_local = np.r_[idx[0], idx[breaks+1]]
        ends_local   = np.r_[idx[breaks], idx[-1]]
        regions = []
        for a, b in zip(starts_local, ends_local):
            length = int(b - a + 1)
            if length <= region_max_len:
                regions.append((int(a), int(b)))
            else:
                k = (length + region_max_len - 1) // region_max_len
                for j in range(k):
                    s = int(a + j * region_max_len)
                    e = int(min(a + (j+1) * region_max_len - 1, b))
                    regions.append((s, e))
        return regions

    proto_regions_local = _runs_to_regions(pass_ref)
    if not proto_regions_local:
        return dict(regions=[], harm_at_best=np.zeros((0,0), dtype=float),
                    selected_channels=np.zeros((0,), dtype=int),
                    ref_channel=ref_channel, center_index=center_index)

    # 3) Stage 2: choose channels and prune inside proto-regions only
    if channels is not None:
        sel_ch = np.array(sorted(set(int(c) for c in channels) | {ref_channel}), dtype=int)
    else:
        sel_mask = (ei_rms >= chan_thr)
        sel_mask[ref_channel] = True
        sel_ch = np.flatnonzero(sel_mask).astype(int)

    n_sel = sel_ch.size
    E_T   = np.sum(ei[sel_ch]**2, axis=1)                  # [n_sel]

    final_regions = []
    harm_cols = []

    for (a_loc, b_loc) in proto_regions_local:
        loc_idx = np.arange(a_loc, b_loc+1, dtype=np.int64)     # indices into relative arrays
        abs_idx = test_samples[loc_idx]                         # absolute raw samples
        rel_st  = loc_idx                                       # relative starts in raw_slice
        m = rel_st.size
        if m == 0:
            continue

        # Compute energies/correlations locally for this proto-region only
        local_start = int(rel_st[0])
        local_end   = int(rel_st[-1] + T)
        raw_local = raw_slice[sel_ch, local_start:local_end].astype(np.float64, copy=False)  # [n_sel, region_len+T-1]

        # prefix sums of x^2 for E_x at each start in this region
        cs = np.empty((n_sel, raw_local.shape[1] + 1), dtype=np.float64)
        cs[:, 0] = 0.0
        np.cumsum(raw_local**2, axis=1, out=cs[:, 1:])
        rel0 = rel_st - local_start
        E_x = cs[:, rel0 + T] - cs[:, rel0]   # [n_sel, m]

        # correlations r for each selected channel at each start
        r = np.empty((n_sel, m), dtype=np.float64)
        for i in range(n_sel):
            ch = sel_ch[i]
            # small loop is fine here
            for j, s in enumerate(rel0):
                r[i, j] = raw_local[i, s:s+T] @ ei[ch]

        sqrt_Ex   = np.sqrt(E_x / T)
        delta     = np.sqrt((E_x + E_T[:, None] - 2.0 * r) / T) - sqrt_Ex # [n_sel, m]

        ref_pos   = int(np.flatnonzero(sel_ch == ref_channel)[0])
        keep      = (delta[ref_pos] < ref_thr)
        keep     &= np.all(delta <= chan_thr, axis=0)
        mean_d    = delta.mean(axis=0)
        keep     &= (mean_d <= mean_thr)
        if not np.any(keep):
            continue

        loc_pass  = np.flatnonzero(keep)
        breaks    = np.flatnonzero(np.diff(loc_pass) > 1)
        sub_st    = np.r_[loc_pass[0], loc_pass[breaks+1]]
        sub_en    = np.r_[loc_pass[breaks], loc_pass[-1]]

        for ss, ee in zip(sub_st, sub_en):
            length = int(ee - ss + 1)
            splits = []
            if length <= region_max_len:
                splits = [(int(ss), int(ee))]
            else:
                k = (length + region_max_len - 1) // region_max_len
                for j in range(k):
                    s2 = int(ss + j * region_max_len)
                    e2 = int(min(ss + (j+1) * region_max_len - 1, ee))
                    splits.append((s2, e2))

            for s2, e2 in splits:
                seg = slice(s2, e2+1)
                seg_mean = mean_d[seg]
                best_local = int(s2 + int(np.argmin(seg_mean)))
                best_abs = int(abs_idx[best_local])
                harm_vec = delta[:, best_local].astype(np.float32)
                final_regions.append(dict(
                    start=int(abs_idx[s2]),
                    end=int(abs_idx[e2]),
                    best_idx=best_abs,
                    mean_delta=float(seg_mean.min()),
                    ref_delta=float(delta[ref_pos, best_local]),
                ))
                harm_cols.append(harm_vec)

    if not final_regions:
        return dict(regions=[], harm_at_best=np.zeros((0,0), dtype=float),
                    selected_channels=sel_ch, ref_channel=ref_channel, center_index=center_index)

    harm_at_best = np.stack(harm_cols, axis=1)  # [n_sel, n_regions]

    # --- NEW: optional amplitude-ratio filter on ref channel ---
    if (amp_ratio_min is not None) or (amp_ratio_max is not None):
        C_tot, L_total = raw_data.shape
        Twin = ei.shape[1]

        kept_regions = []
        kept_cols = []

        for k, r in enumerate(final_regions):
            s = int(r["best_idx"])  # absolute sample of best alignment
            a0 = s - int(center_index)
            a1 = a0 + Twin
            # bounds guard
            if a0 < 0 or a1 > L_total:
                continue

            # ref-channel window aligned to the template
            ref_seg = raw_data[ref_channel, a0:a1]
            raw_ref_trough = float(np.abs(np.min(ref_seg))) + 1e-12
            ratio = raw_ref_trough / tpl_ref_trough

            # decide
            if (amp_ratio_min is not None) and (ratio < amp_ratio_min):
                continue
            if (amp_ratio_max is not None) and (ratio > amp_ratio_max):
                continue

            rr = dict(r)
            rr["amp_ratio_ref"] = ratio
            kept_regions.append(rr)
            kept_cols.append(k)

        # apply selection to harm_at_best and regions
        if len(kept_cols) == 0:
            harm_at_best = np.zeros((harm_at_best.shape[0], 0), dtype=harm_at_best.dtype)
            final_regions = []
        else:
            kept_cols = np.asarray(kept_cols, dtype=int)
            harm_at_best = harm_at_best[:, kept_cols]
            final_regions = kept_regions
    # --- end NEW ---

    return dict(
        regions=final_regions,
        harm_at_best=harm_at_best,
        selected_channels=sel_ch,
        ref_channel=ref_channel,
        center_index=center_index,
    )


# ---------- quick plotting (optional) ----------
def plot_harm_heatmap(result, field="harm_matrix", sort_by_ptp=True, spike_order=None, vclip=None, title=None,
                      vline_at=None, vline_kwargs=None):
    H   = result[field]           # [nch, N]
    ptp = result["channel_ptp"]

    chan_order  = np.argsort(-ptp) if sort_by_ptp else np.arange(H.shape[0])
    if spike_order is None:
        spike_order = np.arange(H.shape[1])
    else:
        spike_order = np.asarray(spike_order, dtype=int)

    Hs = H[chan_order][:, spike_order]

    # Fix label and vclip range for R²
    if field == "harm_matrix_r2":
        if vclip is None:
            vmax = 1.0
        else:
            vmax = float(vclip)
        vmin = -1.0  # Allow some red for poor-fitting spikes
        colorbar_label = "Explained variance (R²)  [1 = perfect fit]"
    else:
        if vclip is None:
            vmax = np.percentile(np.abs(Hs), 98)
        else:
            vmax = float(vclip)
        vmin = -vmax
        colorbar_label = "ΔRMS (res − raw)  [neg = help]"


    # if vclip is None:
    #     vmax = np.percentile(np.abs(Hs), 98)
    # else:
    #     vmax = float(vclip)

    plt.figure(figsize=(20, 4))
    im = plt.imshow(Hs, aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=colorbar_label)

    plt.xlabel("Spike index (ordered)")
    plt.ylabel("Channels (sorted by EI p2p)" if sort_by_ptp else "Channels")
    if title:
        plt.title(title)

    # --- optional vertical line(s) highlighting split(s) ---
    if vline_at is not None:
        ax = plt.gca()
        opts = dict(color='k', linestyle='--', linewidth=1.5, alpha=0.9)
        if vline_kwargs:
            opts.update(vline_kwargs)

        def _to_xpos(x):
            # integers are treated as a column boundary after x-th spike → x-0.5
            if isinstance(x, (int, np.integer)):
                return float(x) - 0.5
            return float(x)

        if isinstance(vline_at, (list, tuple, np.ndarray)):
            for x in vline_at:
                ax.axvline(_to_xpos(x), **opts)
        else:
            ax.axvline(_to_xpos(vline_at), **opts)

    plt.tight_layout()
    plt.show()




def plot_spike_delta_summary(result, base="delta", weighted=True, bins=60, title=None, vline=None):
    """
    Plot histogram of mean ΔRMS or mean R² per spike.

    Parameters:
    - result:  dict containing summary arrays
    - base:    'delta' or 'r2'
    - weighted: True = weighted version; False = unweighted
    - bins:    histogram bins
    - title:   plot title
    - vline:   optional vertical line(s); defaults based on metric
    """
    assert base in ("delta", "r2"), f"Invalid base: {base}"
    field = f"mean_{base}_{'weighted' if weighted else 'unweighted'}"
    d = result.get(field, None)
    if d is None:
        raise ValueError(f"Field '{field}' not found in result.")

    plt.figure(figsize=(6, 2))
    plt.hist(d, bins=bins, color='gray', edgecolor='black')

    # Auto vline
    if vline is None:
        if base == "delta":
            vline = [-2]
        elif base == "r2":
            vline = [0.0]
    if vline is not None:
        if not isinstance(vline, (list, tuple, np.ndarray)):
            vline = [vline]
        for x in vline:
            plt.axvline(x, linestyle='--', color='red', linewidth=1)

    # Axis labels
    if base == "delta":
        plt.xlabel(f"Mean ΔRMS per spike ({'weighted' if weighted else 'unweighted'})")
    elif base == "r2":
        plt.xlabel(f"Mean R² per spike ({'weighted' if weighted else 'unweighted'})")

    plt.ylabel("Count")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def _help_harm_by_spike(res, thr=0.0, spike_order=None, weighted=True, weight_beta=0.5):
    """
    Returns per-spike counts and (weighted) averages for 'help' (Δ<thr) and 'harm' (Δ>thr).
    """
    H   = res["harm_matrix"]        # [nch, N], ΔRMS (res - raw)
    ptp = res["channel_ptp"]        # [nch], EI p2p on selected channels
    nch, N = H.shape

    if spike_order is None:
        order = np.arange(N)
    else:
        order = np.asarray(spike_order, dtype=int)
    H = H[:, order]

    # channel weights (same as in your harm map; re-normalized within each group)
    if weighted:
        w = (ptp.astype(float) ** weight_beta)
    else:
        w = np.ones_like(ptp, dtype=float)

    help_mask = (H <  thr)   # True where subtracting EI helped
    harm_mask = (H >  thr)   # True where it hurt

    n_help = help_mask.sum(axis=0)
    n_harm = harm_mask.sum(axis=0)

    # weighted means within each subgroup, NaN if subgroup empty for that spike
    w_col_help = w[:, None] * help_mask
    w_col_harm = w[:, None] * harm_mask
    denom_help = w_col_help.sum(axis=0)
    denom_harm = w_col_harm.sum(axis=0)
    mean_help = (H * w[:, None] * help_mask).sum(axis=0) / np.where(denom_help > 0, denom_help, np.nan)
    mean_harm = (H * w[:, None] * harm_mask).sum(axis=0) / np.where(denom_harm > 0, denom_harm, np.nan)

    return {
        "order": order,
        "n_help": n_help, "n_harm": n_harm,
        "mean_help": mean_help, "mean_harm": mean_harm
    }

def plot_help_harm_lines(res, thr=0.0, spike_order=None, weighted=True, weight_beta=0.5, title=None):
    m = _help_harm_by_spike(res, thr, spike_order, weighted, weight_beta)
    x = np.arange(m["n_help"].size)

    fig, ax = plt.subplots(1, 2, figsize=(12, 2), sharex=True)
    # counts
    ax[0].plot(x, m["n_help"], label=f"help (Δ<{thr:g})")
    ax[0].plot(x, m["n_harm"], label=f"harm (Δ>{thr:g})")
    ax[0].set_ylabel("# channels")
    ax[0].set_xlabel("Spike index (ordered)")
    ax[0].legend()
    ax[0].grid(True, alpha=0.25)

    # group-wise means
    ax[1].plot(x, m["mean_help"], label="mean Δ (help)")
    ax[1].plot(x, m["mean_harm"], label="mean Δ (harm)")
    ax[1].axhline(0, ls="--", lw=1)
    ax[1].set_ylabel("ΔRMS (res − raw)")
    ax[1].set_xlabel("Spike index (ordered)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.25)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return m  # hand back metrics if you want to reuse them

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors

def compute_spike_gate(
    res,
    *,
    thr_global=-2.0,
    thr_channel=0.0,
    min_good_frac=0.5,
    max_bad_delta=5.0,
    weighted=True,
    weight_beta=0.5,
    # existing (optional) exceed rule args...
    ideal=None,
    exceed_thresh=None,
    exceed_ignore_if_delta_below=None,   # <-- if you added the previous mod
    # NEW:
    force_accept_if_global_below=None    # e.g. -15.0
):

    """
    Returns:
      accept_mask        : bool [N]
      global_mean        : float [N]  (weighted mean Δ across all selected channels)
      n_good, n_bad      : ints [N]
      frac_good          : float [N]
      max_harm_delta     : float [N]  (max Δ among harmful channels; -inf if none)
      n_exceed_max_gap   : int   [N]  #(channels with (Δ - max_trusted) > exceed_thresh, after ignore)
      n_exceed_ignored   : int   [N]  #(channels ignored by exceed_ignore_if_delta_below)  # NEW
    """
    H   = res["harm_matrix"]    # [nch, N], Δ = RMS(res) - RMS(raw)
    ptp = res["channel_ptp"]    # [nch]
    nch, N = H.shape

    good = (H < thr_channel)
    harm = (H > thr_channel)

    n_good = good.sum(axis=0)
    n_bad  = harm.sum(axis=0)
    frac_good = n_good / max(1, nch)

    if weighted:
        w = (ptp.astype(float) ** weight_beta)
    else:
        w = np.ones_like(ptp, float)
    w /= w.sum()

    global_mean = (H * w[:, None]).sum(axis=0)   # [N]

    # per-spike max harmful Δ; if a spike has no harmful channels, it’s -inf
    max_harm_delta = np.where(harm, H, -np.inf).max(axis=0)
    harm_cap_ok = (max_harm_delta <= max_bad_delta) | ~np.isfinite(max_harm_delta)

    # NEW: exceed trusted max rule (with optional ignore for very-helpful channels)
    if (ideal is not None) and (exceed_thresh is not None):
        mxt = np.asarray(ideal["max_trusted"], dtype=float)   # [nch]
        if mxt.shape[0] != nch:
            raise ValueError(f"ideal['max_trusted'] has nch={mxt.shape[0]} but res has nch={nch}")

        delta_minus_max = H - mxt[:, None]                    # [nch, N]
        exceed_mat = delta_minus_max > float(exceed_thresh)   # raw exceeders (per-channel)

        # --- ignore channels that are already very helpful (Δ < ignore threshold) ---
        if exceed_ignore_if_delta_below is not None:
            helpful_big = H < float(exceed_ignore_if_delta_below)   # [nch, N]
            ignored_mat = exceed_mat & helpful_big                  # channels we will ignore
            exceed_mat  = exceed_mat & ~helpful_big                 # keep only non-ignored exceeders
            n_exceed_ignored = ignored_mat.sum(axis=0).astype(int)
        else:
            n_exceed_ignored = np.zeros(N, dtype=int)

        n_exceed_max_gap = exceed_mat.sum(axis=0).astype(int)
        exceed_ok = (n_exceed_max_gap == 0)
    else:
        n_exceed_max_gap = np.zeros(N, dtype=int)
        n_exceed_ignored = np.zeros(N, dtype=int)
        exceed_ok = np.ones(N, dtype=bool)


    accept = (
        (global_mean < thr_global) &
        (frac_good   >= min_good_frac) &
        harm_cap_ok &
        exceed_ok    # NEW veto
    )

     # --- force-accept override: if global mean ΔRMS is very negative, accept regardless ---
    if force_accept_if_global_below is not None:
        forced = (global_mean < float(force_accept_if_global_below))
        accept = accept | forced
    else:
        forced = np.zeros_like(global_mean, dtype=bool)


    return {
        "accept_mask": accept,
        "global_mean": global_mean,
        "n_good": n_good, "n_bad": n_bad,
        "frac_good": frac_good,
        "max_harm_delta": max_harm_delta,
        "n_exceed_max_gap": n_exceed_max_gap,   # expose for plotting/debug
        "n_exceed_ignored": n_exceed_ignored,   # NEW
        "forced_accept_mask": forced,
    }


def plot_harm_scatter_rms_vs_r2(result, idx_in=None, idx_out=None, title=None,
                                 weighted=True, alpha=0.6, s=18):
    """
    Scatter plot: mean ΔRMS vs mean R² for each spike.
    Accepted = blue, rejected = red.

    Parameters:
    - result:   dict containing harm_matrix, harm_matrix_r2, and weight info
    - idx_in:   list/array of accepted spike indices
    - idx_out:  list/array of rejected spike indices
    - weighted: use channel-weighted means
    - alpha:    transparency
    - s:        marker size
    """
    base_x = "mean_delta_weighted" if weighted else "mean_delta_unweighted"
    base_y = "mean_r2_weighted" if weighted else "mean_r2_unweighted"
    x = result.get(base_x, None)
    y = result.get(base_y, None)
    if x is None or y is None:
        raise ValueError(f"Missing required field(s): {base_x}, {base_y}")

    if idx_in is None or idx_out is None:
        # Default full split
        N = len(x)
        mask = np.zeros(N, dtype=bool)
        acc = result.get("accept_mask")
        if acc is not None and acc.size == N:
            mask = acc
        idx_in = np.where(mask)[0]
        idx_out = np.where(~mask)[0]

    plt.figure(figsize=(6, 5))
    if len(idx_out):
        plt.scatter(x[idx_out], y[idx_out], s=s, alpha=alpha, c='red', label='Rejected')
    if len(idx_in):
        plt.scatter(x[idx_in], y[idx_in], s=s, alpha=alpha, c='blue', label='Accepted')

    plt.axvline(0, color='gray', lw=1, linestyle='--', alpha=0.5)
    plt.axhline(0, color='gray', lw=1, linestyle='--', alpha=0.5)

    plt.xlabel("Mean ΔRMS per spike")
    plt.ylabel("Mean R² per spike")
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()


import numpy as np
import matplotlib.pyplot as plt

# ---------- 1) Score every spike vs every template ----------
def compute_template_scores_from_harm(
    snips_raw,               # [C,T,N] float32
    accepted_eis,            # list of dicts with 'ei' (and ideally 'selected_channels' but not required)
    *,
    p2p_thr=50.0,
    max_channels=80,
    min_channels=10,
    lag_radius=3,
    weight_by_p2p=True,
    weight_beta=0.5,         # MUST match what you want to use in the harm-map
    use_weighted=True        # return both weighted+unweighted; this selects the default for plotting
):
    """
    For each template in accepted_eis, run harm-map on the SAME snips_raw and
    collect per-spike scores:
        - delta_weighted / delta_unweighted
        - r2_weighted    / r2_unweighted
    Returns:
        dict with 4 matrices of shape [M_templates, N_spikes]
        plus per-template metadata used internally.
    """
    M = len(accepted_eis)
    C, T, N = snips_raw.shape
    delta_w = np.empty((M, N), dtype=np.float32)
    delta_u = np.empty((M, N), dtype=np.float32)
    r2_w    = np.empty((M, N), dtype=np.float32)
    r2_u    = np.empty((M, N), dtype=np.float32)

    meta = []  # stash per-template res bits we use to build R2

    for i, tpl in enumerate(accepted_eis):
        ei = tpl["ei"].astype(np.float32, copy=False)

        res = compute_harm_map_noamp(
            ei, snips_raw,
            p2p_thr=p2p_thr, max_channels=max_channels, min_channels=min_channels,
            lag_radius=lag_radius, weight_by_p2p=weight_by_p2p, weight_beta=weight_beta
        )
        # ΔRMS summaries (already provided by your harm-map)
        delta_w[i, :] = np.asarray(res["mean_delta_weighted"],   dtype=np.float32)
        delta_u[i, :] = np.asarray(res["mean_delta_unweighted"], dtype=np.float32)

        # For R² we need per-channel ΔRMS + RMS_raw on the same channel set
        sel  = np.asarray(res["selected_channels"], dtype=int)
        harm = np.asarray(res["harm_matrix"], dtype=np.float32)      # [nch, N] = RMS_res - RMS_raw
        # RMS_raw per channel/spike from snips_raw
        raw_sel = snips_raw[sel, :, :]                                # [nch, T, N]
        rms_raw = np.sqrt(np.mean(raw_sel**2, axis=1, dtype=np.float64)).astype(np.float32)  # [nch, N]

        # R² per channel/spike
        rms_res = rms_raw + harm
        denom   = np.maximum(rms_raw**2, 1e-12)
        r2_ch   = 1.0 - (rms_res**2) / denom                           # [nch, N]

        # weights (same rule the harm-map used)
        ptp_sel = np.asarray(res["channel_ptp"], dtype=np.float32)     # [nch]
        if weight_by_p2p:
            w = (ptp_sel ** weight_beta).astype(np.float32)
        else:
            w = np.ones_like(ptp_sel, dtype=np.float32)
        s = float(w.sum()) if w.size else 1.0
        w = w / s

        # aggregate across channels
        r2_w[i, :] = (r2_ch * w[:, None]).sum(axis=0)
        r2_u[i, :] = r2_ch.mean(axis=0)

        meta.append({"sel": sel, "w": w})

    return {
        "delta_weighted":   delta_w,
        "delta_unweighted": delta_u,
        "r2_weighted":      r2_w,
        "r2_unweighted":    r2_u,
        "weight_beta":      weight_beta,
        "use_weighted":     use_weighted,
        "meta":             meta
    }

# ---------- 2) Best/second-best picking ----------
def _top2_per_spike(score_matrix, *, better="lower"):
    """
    Given [M,N] per-template-per-spike scores, pick best and second per spike.
    better='lower' for ΔRMS, 'higher' for R².
    Returns: best, second, best_idx, second_idx
    """
    assert better in ("lower", "higher")
    if better == "lower":
        order = np.argsort(score_matrix, axis=0)        # ascending
    else:
        order = np.argsort(-score_matrix, axis=0)       # descending
    idx1 = order[0, :]
    idx2 = order[1, :] if score_matrix.shape[0] >= 2 else np.full(score_matrix.shape[1], -1, int)
    cols = np.arange(score_matrix.shape[1])
    best   = score_matrix[idx1, cols]
    second = score_matrix[idx2, cols] if score_matrix.shape[0] >= 2 else np.full(score_matrix.shape[1], np.nan)
    return best, second, idx1, idx2

# ---------- 3) Plots: best vs second for ΔRMS and R² ----------
def plot_best_vs_second_from_scores(scores, *, weighted=True, s=14, alpha=0.55, title_prefix=""):
    """
    Uses score matrices from compute_template_scores_from_harm(..)
    Produces two scatter plots (ΔRMS and R²): X=best, Y=second best.
    """
    # pick matrices
    D = scores["delta_weighted"] if weighted else scores["delta_unweighted"]
    R = scores["r2_weighted"]    if weighted else scores["r2_unweighted"]

    # ΔRMS: lower is better
    best_d, second_d, _, _ = _top2_per_spike(D, better="lower")
    plt.figure(figsize=(5.5, 5))
    plt.scatter(best_d, second_d, s=s, alpha=alpha, c='gray', edgecolor='none')
    # diagonal
    mn = np.nanmin([best_d.min(), second_d.min()])
    mx = np.nanmax([best_d.max(), second_d.max()])
    plt.plot([mn, mx], [mn, mx], ls='--', lw=1, color='k', alpha=0.4)
    plt.xlabel("Best ΔRMS per spike (most negative)")
    plt.ylabel("Second-best ΔRMS per spike")
    plt.title(f"{title_prefix}Best vs Second (ΔRMS, {'weighted' if weighted else 'unweighted'})")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    # R²: higher is better
    best_r2, second_r2, _, _ = _top2_per_spike(R, better="higher")
    plt.figure(figsize=(5.5, 5))
    plt.scatter(best_r2, second_r2, s=s, alpha=alpha, c='gray', edgecolor='none')
    mn = np.nanmin([best_r2.min(), second_r2.min()])
    mx = np.nanmax([best_r2.max(), second_r2.max()])
    plt.plot([mn, mx], [mn, mx], ls='--', lw=1, color='k', alpha=0.4)
    plt.xlabel("Best R² per spike (largest)")
    plt.ylabel("Second-best R² per spike")
    plt.title(f"{title_prefix}Best vs Second (R², {'weighted' if weighted else 'unweighted'})")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()




def plot_help_harm_scatter_swapped(
    res, thr=0.0, spike_order=None, weighted=True, weight_beta=0.5,
    title="Scatter: mean Δ vs #channels (color = |opposite mean Δ|)",
    cmap="YlGn_r", big_mask=None, big_thresh=None, s_small=14, s_big=48
):
    # reuse helper from before
    m = _help_harm_by_spike(res, thr, spike_order, weighted, weight_beta)
    order = m["order"]
    N = order.size

    # global weighted mean (for optional threshold mode)
    global_mean = np.asarray(res["mean_delta_weighted"])[order]

    # sizes: prefer external gate mask; else threshold; else all small
    if big_mask is not None:
        big_mask = np.asarray(big_mask, dtype=bool)[order]
        sizes = np.where(big_mask, s_big, s_small).astype(float)
    elif big_thresh is not None:
        sizes = np.where(global_mean < big_thresh, s_big, s_small).astype(float)
    else:
        sizes = np.full(N, s_small, float)

    n_help, n_harm = m["n_help"], m["n_harm"]
    mean_help, mean_harm = m["mean_help"], m["mean_harm"]

    # color by |opposite mean Δ| (min–max per opposite group)
    def norm_colors(vals):
        v = np.abs(vals).copy()
        if np.all(np.isnan(v)): v[:] = 0.0
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        if not np.isfinite(vmax - vmin) or (vmax - vmin) == 0: vmax = vmin + 1.0
        return cm.get_cmap(cmap)(mcolors.Normalize(vmin=vmin, vmax=vmax)(v))

    colors_help = norm_colors(mean_harm)  # help colored by opposite (harm) |meanΔ|
    colors_harm = norm_colors(mean_help)  # harm colored by opposite (help) |meanΔ|

    plt.figure(figsize=(6, 3))
    mh = ~np.isnan(mean_help)
    mb = ~np.isnan(mean_harm)

    plt.scatter(mean_help[mh], n_help[mh], c=colors_help[mh], s=sizes[mh],
                alpha=0.85, label=f"help (Δ<{thr:g})")
    plt.scatter(mean_harm[mb], n_harm[mb], c=colors_harm[mb], s=sizes[mb],
                alpha=0.85, label=f"harm (Δ>{thr:g})")

    plt.axvline(0, ls="--", lw=1)
    plt.xlabel("Group mean ΔRMS (res − raw)")
    plt.ylabel("# channels in group")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

    return {
        "order": order,
        "sizes": sizes,
        "global_mean": global_mean,
        "n_help": n_help, "n_harm": n_harm,
        "mean_help": mean_help, "mean_harm": mean_harm
    }


def plot_ei_quick(ei, ei_positions, title, channel_of_interest, scale=90):
    plt.figure(figsize=(18, 8))
    plot_ei_waveforms.plot_ei_waveforms(
        ei, ei_positions,
        ref_channel=channel_of_interest,
        scale=scale, box_height=1, box_width=50,
        colors='black', aspect=0.5)
    plt.title(title)
    plt.show()


def bimodality_probe(snips_cand, c_best, t_best, SHOW_DIAGNOSTICS, offsets=(-5,-3,-1,1,3,5,7,9,11)):
    T, N = snips_cand.shape[1], snips_cand.shape[2]
    reports = []
    for d in offsets:
        t = t_best + d
        if t < 0 or t >= T: 
            continue
        v = snips_cand[c_best, t, :].astype(float)
        X = v.reshape(-1,1)
        g1 = GaussianMixture(1, reg_covar=1e-6, random_state=0).fit(X)
        g2 = GaussianMixture(2, reg_covar=1e-6, random_state=0).fit(X)
        bic1 = float(g1.bic(X)); bic2 = float(g2.bic(X)); db = bic1 - bic2
        idx = np.argsort(g2.means_.ravel())
        mu  = g2.means_.ravel()[idx]
        sd  = np.sqrt(g2.covariances_.reshape(-1))[idx]
        wt  = g2.weights_[idx]
        dpr = float(abs(mu[1]-mu[0]) / (np.sqrt(0.5*(sd[0]**2+sd[1]**2))+1e-12))
        reports.append(dict(offset=d, sample=int(t), N=int(N),
                            bic1=bic1, bic2=bic2, delta_bic=float(db),
                            dprime=dpr, means=mu.astype(float),
                            stds=sd.astype(float), weights=wt.astype(float),
                            sizes=(wt*N).round().astype(int)))
    if not reports:
        print(f"\nBIMODALITY PROBE: c={c_best}, t_best={t_best} → no valid offsets in {list(offsets)}")
        return None
    best = max(reports, key=lambda r: (r['delta_bic'], r['dprime']))
    if SHOW_DIAGNOSTICS:
        mu = ", ".join(f"{m:.2f}" for m in best['means'])
        sd = ", ".join(f"{s:.2f}" for s in best['stds'])
        wt = ", ".join(f"{w:.2f}" for w in best['weights'])
        sz = ", ".join(str(int(s)) for s in best['sizes'])
        print("\nBIMODALITY PROBE (best channel, off-peak)")
        print(f"  c={c_best}  t_best={t_best}  tested={list(offsets)}")
        print(f"  selected t={best['sample']} (off {best['offset']}), N={best['N']}")
        print(f"  BIC1={best['bic1']:.2f}  BIC2={best['bic2']:.2f}  ΔBIC={best['delta_bic']:.2f}  d′={best['dprime']:.2f}")
        print(f"  means=[{mu}]  stds=[{sd}]  weights=[{wt}]  sizes=[{sz}]")
    return best


def build_delta_prototype(
    res, *,
    subset=None,           # optional global spike indices to learn from (e.g., idx_in)
    good_thresh=-5.0,      # spikes with mean Δ ≤ this are eligible
    top_frac=0.25,         # among eligible, take this fraction with most-negative mean Δ
    core_k=None,           # if None, use core_frac; else fixed count
    core_frac=0.5,         # fraction of channels as "core" when core_k is None
    min_periph=5,          # ensure ≥ this many periphery channels when possible
    beta=0.7               # p2p^beta weights for consistency with the rest
):
    """
    Learn per-channel expected Δ profile (mu_c) and robust scale (mad_c) from very-good spikes.
    Returns dict with mu_c, mad_c, core_idx, periph_idx, weights, ptp, idx_good (global).
    """
    H   = res["harm_matrix"]          # [nch, N]
    ptp = res["channel_ptp"]          # [nch]
    mdw = res["mean_delta_weighted"]  # [N]
    nch, N = H.shape

    # ----- restrict to subset if provided -----
    if subset is not None:
        subset = np.asarray(subset, dtype=int)
        pool_indices = subset                                 # global indices we’re considering
        mdw_pool = mdw[subset]
    else:
        pool_indices = np.arange(N, dtype=int)
        mdw_pool = mdw

    # ----- pick trusted spikes -----
    good_pool_local = np.where(mdw_pool <= good_thresh)[0]     # LOCAL indices w.r.t. pool_indices
    if good_pool_local.size == 0:
        # fallback: take top_frac most negative from the whole pool
        order = np.argsort(mdw_pool)                           # ascending (most negative first)
        q = max(1, int(np.floor(order.size * top_frac)))
        good_local = order[:q]
    else:
        q = max(1, int(np.floor(good_pool_local.size * top_frac)))
        good_local = good_pool_local[np.argsort(mdw_pool[good_pool_local])[:q]]

    idx_good_global = pool_indices[good_local]                 # map to GLOBAL spike indices

    # ----- per-channel robust center and spread from trusted spikes -----
    G = H[:, idx_good_global]                                  # [nch, q]
    mu_c = np.median(G, axis=1)
    mad_c = np.median(np.abs(G - mu_c[:, None]), axis=1)
    mad_c = np.where(mad_c > 1e-6, mad_c, 1.0)                 # avoid div-by-zero later

    # ----- core/periphery split by EI strength (ensure periphery exists) -----
    if core_k is None:
        k = int(round(nch * core_frac))
    else:
        k = int(core_k)
    # clamp so we leave at least min_periph channels in periphery when possible
    k = max(1, min(k, max(1, nch - min_periph)))

    order_by_ptp = np.argsort(ptp)[::-1]
    core_idx   = order_by_ptp[:k]
    periph_idx = order_by_ptp[k:]                                # may be empty if nch ≤ k

    # ----- channel weights (consistent with your other plots) -----
    w = (ptp.astype(float) ** beta)
    w_sum = w.sum()
    w = w / w_sum if w_sum > 0 else np.full_like(w, 1.0 / max(1, w.size), dtype=float)

    return {
        "mu_c": mu_c, "mad_c": mad_c,
        "core_idx": core_idx, "periph_idx": periph_idx,
        "weights": w, "ptp": ptp,
        "idx_good": idx_good_global,
    }



def reassign_leftovers_to_existing_templates(accepted_eis, snips_pool, spike_times_pool, pool_ids, SHOW_DIAGNOSTICS, cfg: SimpleNamespace):
    """
    Try to place leftover spikes onto already-accepted templates.
    Uses the same harm/gate as elsewhere. Does NOT recalc EIs.
    Returns updated (snips_pool, spike_times_pool, pool_ids, placed_total).
    """
    placed_total = 0
    if (snips_pool.shape[2] == 0) or (len(accepted_eis) == 0):
        return snips_pool, spike_times_pool, pool_ids, placed_total

    while True:
        N = snips_pool.shape[2]
        if N == 0: break
        avail = np.ones(N, dtype=bool)
        placed_this_pass = 0

        for ti, tpl in enumerate(accepted_eis):
            ei_t = tpl['ei']
            # evaluate only on currently available spikes
            res = compute_harm_map_noamp(
                ei_t, snips_pool,
                p2p_thr=cfg.HARM_P2P_THR, max_channels=cfg.HARM_MAX_CHANNELS, min_channels=cfg.HARM_MIN_CHANNELS,
                lag_radius=cfg.HARM_LAG_RADIUS, weight_by_p2p=cfg.WEIGHT_BY_P2P, weight_beta=cfg.WEIGHT_BETA
            )
            gate = compute_spike_gate(
                res,
                thr_global=cfg.GATE_THR_GLOBAL, thr_channel=0.0,
                min_good_frac=cfg.GATE_MIN_GOOD_FRAC, max_bad_delta=cfg.GATE_MAX_BAD_DELTA,
                weighted=cfg.WEIGHT_BY_P2P, weight_beta=cfg.WEIGHT_BETA
            )
            acc = (gate["accept_mask"] & avail)
            n_acc = int(acc.sum())
            if n_acc > 0:
                # append spike times, no EI recompute
                accepted_eis[ti]['spike_times'] = np.concatenate([accepted_eis[ti]['spike_times'],
                                                                   spike_times_pool[acc]])
                # mark as consumed
                avail[acc] = False
                placed_this_pass += n_acc
                if SHOW_DIAGNOSTICS:
                    print(f"Reassigned {n_acc} leftover spikes to template {ti}.")

        if placed_this_pass == 0:
            break

        # drop consumed spikes from pool
        snips_pool       = snips_pool[:, :, avail]
        spike_times_pool = spike_times_pool[avail]
        pool_ids         = pool_ids[avail]
        placed_total    += placed_this_pass

    if SHOW_DIAGNOSTICS:
        print(f"Reassignment placed {placed_total} spikes into existing templates.")
    return snips_pool, spike_times_pool, pool_ids, placed_total


def score_against_prototype(res, proto, *, harm_cap=10.0, tau_core=1.0, tau_periph=1.5):
    """
    For each spike, compute S_core/S_periph from z-shortfalls vs the prototype,
    plus max harmful Δ for your cap. Labels: {'good','near_miss','bad'}.
    """
    H = res["harm_matrix"]            # [nch, N]
    mu, mad = proto["mu_c"], proto["mad_c"]
    core, periph = proto["core_idx"], proto["periph_idx"]

    shortfall = np.maximum(0.0, H - mu[:, None])          # only worse-than-expected counts
    zshort = shortfall / mad[:, None]

    def safe_mean(arr, axis=0):
        return np.nanmean(np.where(np.isfinite(arr), arr, np.nan), axis=axis)

    S_core   = safe_mean(zshort[core, :], axis=0)
    S_periph = safe_mean(zshort[periph, :], axis=0) if periph.size > 0 else np.zeros(H.shape[1])

    harm_mask = (H > 0)
    max_harm = np.where(harm_mask, H, -np.inf).max(axis=0)
    cap_ok   = (max_harm <= harm_cap) | ~np.isfinite(max_harm)

    labels = np.empty(H.shape[1], dtype=object); labels[:] = "bad"
    good_core = (S_core <= tau_core) & cap_ok
    near_miss = good_core & (S_periph > tau_periph)
    good_core = good_core & ~near_miss
    labels[good_core] = "good"; labels[near_miss] = "near_miss"

    return {
        "S_core": S_core, "S_periph": S_periph,
        "max_harm": max_harm, "labels": labels,
        "good_mask": good_core, "near_miss_mask": near_miss
    }

def _sanitize_order(order, N):
    if order is None:
        return np.arange(N, dtype=int)
    o = np.asarray(order, dtype=int).ravel()
    o = o[(o >= 0) & (o < N)]
    # de-duplicate, keep first occurrence
    seen = set(); safe = []
    for i in o:
        if i not in seen:
            seen.add(i); safe.append(i)
    return np.asarray(safe if len(safe)>0 else np.arange(N, dtype=int))


def build_ideal_delta(res, *, subset=None, good_thresh=-5.0, top_frac=0.25):
    """
    Learn per-channel 'ideal' Δ profile from a trusted subset of spikes.
    Returns per-channel:
      mu_c          : median Δ over trusted spikes
      var_c         : variance of Δ over trusted spikes (ddof=0)
      max_trusted   : max Δ over trusted spikes  (more positive = worse)
      trusted_idx   : global spike indices used to build the profile
    """
    H   = res["harm_matrix"]          # [nch, N]  (Δ = RMS(res) - RMS(raw))
    mdw = res["mean_delta_weighted"]  # [N]
    nch, N = H.shape

    pool = np.asarray(subset, dtype=int) if subset is not None else np.arange(N, dtype=int)
    mdw_pool = mdw[pool]

    # Eligible = mean Δ ≤ good_thresh; fallback to most negative if none eligible
    elig_local = np.where(mdw_pool <= good_thresh)[0]
    if elig_local.size == 0:
        order = np.argsort(mdw_pool)                           # most negative first
        q = max(1, int(np.floor(order.size * top_frac)))
        good_local = order[:q]
    else:
        q = max(1, int(np.floor(elig_local.size * top_frac)))
        good_local = elig_local[np.argsort(mdw_pool[elig_local])[:q]]

    trusted = pool[good_local]                                  # global indices
    G = H[:, trusted]                                           # [nch, q]

    mu_c        = np.median(G, axis=1)                          # [nch]
    var_c       = np.var(G, axis=1)                             # [nch], ddof=0 to avoid NaNs at q=1
    max_trusted = np.max(G, axis=1)                             # [nch]

    return {
        "mu_c": mu_c.astype(float),
        "var_c": var_c.astype(float),
        "max_trusted": max_trusted.astype(float),
        "trusted_idx": trusted
    }



def compute_profile_deviation_metrics(res, ideal, *, var_threshold=3.0, exceed_thresh=25.0):

    """
    From res and per-channel 'ideal' stats, compute per-spike diagnostics:

    Returns dict of 1D arrays length N:
      n_gt_mu_plus_varthr   : (1) #channels where Δ > mu_c + var_threshold * var_c
      n_gt_half_max         : (2) #channels where Δ > 0.5 * max_trusted
      n_pos_when_trusted_lt0: (3) #channels where (max_trusted < 0) and (Δ > 0)
      max_delta_minus_max   : (4) max over channels of (Δ - max_trusted)
      mean_pos_delta_gap    : (5) mean(Δ - max_trusted) over channels with Δ>0 (0 if none)
      n_exceed_max_gap : #(channels with Δ - max_trusted > exceed_thresh)
    """
    H   = res["harm_matrix"]            # [nch, N]
    mu  = ideal["mu_c"]                 # [nch]
    var = ideal["var_c"]                # [nch]
    mxt = ideal["max_trusted"]          # [nch]
    nch, N = H.shape

    # (1)
    thr = mu[:, None] + (var_threshold * var)[:, None]
    n_gt_mu_plus_varthr = (H > thr).sum(axis=0).astype(int)

    # (2)
    n_gt_half_max = (H > (0.5 * mxt)[:, None]).sum(axis=0).astype(int)

    # (3)
    trusted_help = (mxt < 0.0)[:, None]
    n_pos_when_trusted_lt0 = (trusted_help & (H > 0.0)).sum(axis=0).astype(int)

    # (4)
    delta_minus_max = H - mxt[:, None]
    max_delta_minus_max = delta_minus_max.max(axis=0)

    # (5)
    pos = (H > 0.0)
    sum_pos = (H * pos).sum(axis=0)
    cnt_pos = pos.sum(axis=0)
    mean_pos_delta = np.divide(sum_pos, np.maximum(cnt_pos, 1), where=(cnt_pos>0))
    sum_mxt_on_pos = (mxt[:, None] * pos).sum(axis=0)
    mean_mxt_on_pos = np.divide(sum_mxt_on_pos, np.maximum(cnt_pos, 1), where=(cnt_pos>0))
    mean_pos_delta_gap = np.where(cnt_pos>0, (mean_pos_delta - mean_mxt_on_pos), 0.0)

    # (6) NEW: count channels exceeding trusted max by a fixed margin
    n_exceed_max_gap = (delta_minus_max > float(exceed_thresh)).sum(axis=0).astype(int)

    return dict(
        n_gt_mu_plus_varthr=n_gt_mu_plus_varthr,
        n_gt_half_max=n_gt_half_max,
        n_pos_when_trusted_lt0=n_pos_when_trusted_lt0,
        max_delta_minus_max=max_delta_minus_max,
        mean_pos_delta_gap=mean_pos_delta_gap,
        n_exceed_max_gap=n_exceed_max_gap,          # (6)
    )


def plot_deviation_lines(res, ideal, *, var_threshold=3.0, exceed_thresh=25.0, spike_order=None, title=None):
    """
    Panels:
      [0] counts per spike: #(Δ > μ + thr·var), #(Δ > 0.5·max_trusted), #(Δ>0 where max_trusted<0)
      [1] gap magnitudes:  max(Δ − max_trusted), mean(Δ − max_trusted) on Δ>0
      [2] NEW count:       #(Δ − max_trusted > exceed_thresh)
    """
    N = res["harm_matrix"].shape[1]
    order = _sanitize_order(spike_order, N)

    metrics = compute_profile_deviation_metrics(res, ideal, var_threshold=var_threshold, exceed_thresh=exceed_thresh)

    x = np.arange(order.size)
    fig, ax = plt.subplots(1, 3, figsize=(20, 2), sharex=True)

    # [0] Counts
    ax[0].plot(x, metrics["n_gt_mu_plus_varthr"][order], label=f"#(Δ > μ + {var_threshold}·var)")
    ax[0].plot(x, metrics["n_gt_half_max"][order],         label="#(Δ > 0.5·max_trusted)")
    ax[0].plot(x, metrics["n_pos_when_trusted_lt0"][order],label="#(Δ>0 where max_trusted<0)")
    ax[0].set_ylabel("# channels"); ax[0].set_xlabel("Spike index"); ax[0].grid(True, alpha=0.25)
    ax[0].legend(loc="upper right", fontsize=8)

    # [1] Gaps
    ax[1].plot(x, metrics["max_delta_minus_max"][order],   label="max(Δ − max_trusted)")
    ax[1].plot(x, metrics["mean_pos_delta_gap"][order],    label="mean(Δ − max_trusted) on Δ>0")
    ax[1].axhline(0, ls="--", lw=1)
    ax[1].set_ylabel("Δ gap"); ax[1].set_xlabel("Spike index"); ax[1].grid(True, alpha=0.25)
    ax[1].legend(loc="upper right", fontsize=8)

    # [2] NEW exceed count
    ax[2].plot(x, metrics["n_exceed_max_gap"][order], label=f"#(Δ − max_trusted > {exceed_thresh:g})")
    ax[2].set_ylabel("# channels"); ax[2].set_xlabel("Spike index"); ax[2].grid(True, alpha=0.25)
    ax[2].legend(loc="upper right", fontsize=8)

    if title: fig.suptitle(title)
    fig.tight_layout()

    return metrics


def compute_deviation_signals(res, mu_c, *, top_k=5):
    """
    dev_matrix = actualΔ − idealΔ (positive = worse than ideal)
    Returns per-spike:
      dev_mean_all: mean over all channels
      dev_mean_topk_bad: mean of largest 'top_k' positive deviations (only >0; if <k positives, average over existing)
      pos_counts: number of channels with positive deviation
    """
    H = res["harm_matrix"]               # [nch, N]
    D = H - mu_c[:, None]                # [nch, N]  dev_from_ideal (positive => worse)
    dev_mean_all = D.mean(axis=0)        # [N], unweighted mean over channels

    # top-k positive-only mean (per spike)
    Dpos = np.maximum(0.0, D)                                       # clip negatives
    # sort descending along channels to take the largest positives
    Dpos_sorted = np.sort(Dpos, axis=0)[::-1, :]                    # [nch, N]
    k = min(top_k, Dpos_sorted.shape[0])
    topk = Dpos_sorted[:k, :]                                       # [k, N]
    pos_counts = (Dpos > 0).sum(axis=0)                             # [N]
    denom = np.clip(np.minimum(pos_counts, k), 1, None).astype(float)
    dev_mean_topk_bad = topk.sum(axis=0) / denom                    # [N]; 0 if no positives

    return {
        "dev_matrix": D,
        "dev_mean_all": dev_mean_all,
        "dev_mean_topk_bad": dev_mean_topk_bad,
        "pos_counts": pos_counts
    }



def recursive_bimodal_split(
    snips_pool, cand_mask, *, 
    channel_of_interest, c_best, t_best,
    cfg, SHOW_DIAGNOSTICS,
    max_splits=4,             # safety to avoid runaway loops
    size_mult=3,              # keep your 3× MIN_CLUSTER_SIZE gate
    bic_thr=300.0, dprime_thr=5.0
):
    """
    Repeatedly check bimodality and split the picked cluster on one channel/time,
    until it is no longer bimodal (on both probes) or the selected subcluster is too small.

    Probes (in order): (channel_of_interest, ~40±offsets), and (c_best, ~t_best±offsets) if distinct.

    Returns:
        cand_mask   : updated boolean mask on the CURRENT pool (global indices)
        ei_cand     : median EI of the final picked cluster
        n_splits    : number of successful splits performed
        last_probe  : ('c','t','metric_val') tuple of the split that happened last (or None)
    """
    n_splits = 0
    last_probe = None

    # Build probe list: ensure uniqueness but preserve priority
    probes = [(int(channel_of_interest), 40)]
    if int(c_best) != int(channel_of_interest):
        probes.append((int(c_best), int(t_best)))

    while True:
        # Stop if too small to be meaningful at all
        snips_cand = snips_pool[:, :, cand_mask]
        n_cur = snips_cand.shape[2]
        if n_cur < cfg.MIN_CLUSTER_SIZE:
            if SHOW_DIAGNOSTICS:
                print(f"RECURSION STOP: cluster too small for any further work (n={n_cur} < {cfg.MIN_CLUSTER_SIZE})")
            break

        # Try each probe; split on the first that passes the gate
        split_happened = False
        for (c_probe, t_probe) in probes:
            best = bimodality_probe(snips_cand, c_probe, t_probe, SHOW_DIAGNOSTICS)
            if best is None:
                continue
            if (best['delta_bic'] <= bic_thr) and (best['dprime'] <= dprime_thr):
                # Not clearly bimodal on this probe
                continue

            t_sel = int(best['sample'])
            v = snips_cand[c_probe, t_sel, :].astype(float).reshape(-1, 1)

            g2 = GaussianMixture(n_components=2, reg_covar=1e-6, random_state=0).fit(v)
            mu = g2.means_.ravel()
            labels = g2.predict(v)

            # choose by descending |μ|, like your current logic
            try_order = np.argsort(np.abs(mu))[::-1]
            idx_cand_loc = np.where(cand_mask)[0]
            picked = None
            thr = size_mult * cfg.MIN_CLUSTER_SIZE

            for k in try_order:
                submask_local = (labels == k)
                n_sub = int(submask_local.sum())
                if n_sub >= thr:
                    new_mask = np.zeros_like(cand_mask, dtype=bool)
                    new_mask[idx_cand_loc[submask_local]] = True
                    if SHOW_DIAGNOSTICS:
                        print(
                            f"SPLIT-GATE TRIGGERED (recursive) → "
                            f"c={c_probe}, t={t_sel}, picked comp {k} |μ|={abs(mu[k]):.2f}, "
                            f"size={n_sub}/{idx_cand_loc.size} (thr={thr})"
                        )
                    cand_mask = new_mask
                    n_splits += 1
                    last_probe = (c_probe, t_sel, float(best['delta_bic']))
                    split_happened = True
                    break
                else:
                    if SHOW_DIAGNOSTICS:
                        print(f"  split candidate comp {k} too small: {n_sub} < {thr} (skipping)")

            if split_happened:
                break  # go re-evaluate bimodality from the top on the new cluster

        # No probe triggered a split → stop
        if not split_happened:
            if SHOW_DIAGNOSTICS:
                print("RECURSION STOP: no probe shows strong bimodality or subclusters too small.")
            break

        # Safety clamp on # of splits
        if n_splits >= max_splits:
            if SHOW_DIAGNOSTICS:
                print(f"RECURSION STOP: reached max_splits={max_splits}.")
            break

    # Final EI for the (possibly reduced) cluster
    ei_cand = median_ei_adaptive(snips_pool[:, :, cand_mask])
    return cand_mask, ei_cand, n_splits, last_probe



def _wilson_lower_bound(k, n, z=1.96):
    """Wilson score lower bound for a proportion, good for small N."""
    if n <= 0:
        return np.nan
    phat = k / n
    denom = 1.0 + (z*z)/n
    center = phat + (z*z)/(2*n)
    spread = z * np.sqrt((phat*(1.0 - phat) + (z*z)/(4*n)) / n)
    return (center - spread) / denom

def lag_metrics_from_res(res, mask=None, central_band=1, z=1.96):
    """
    Compute lag metrics from compute_harm_map_noamp output.
      central_LB: Wilson lower bound of P(|lag|<=central_band)
      edge_frac : fraction at min/max lag of the bank
      mad       : median absolute deviation (around median lag)
    """
    best = np.asarray(res["best_lag_per_spike"])
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        best = best[mask]
    N = best.size
    if N == 0:
        return {"N": 0, "central_LB": np.nan, "edge_frac": np.nan, "mad": np.nan}

    lags = np.asarray(res["lags"])
    lmin, lmax = lags.min(), lags.max()

    # central mass within ±central_band samples
    central = np.abs(best) <= central_band
    central_LB = _wilson_lower_bound(int(central.sum()), N, z=z)

    # edge hits
    edge_frac = np.mean((best == lmin) | (best == lmax))

    # robust spread
    med = np.median(best)
    mad = np.median(np.abs(best - med))

    return {"N": int(N), "central_LB": float(central_LB), "edge_frac": float(edge_frac), "mad": float(mad)}


def main_channel_traces(snips, ch, idx, title):
    traces = snips[ch, :, idx]
    plt.figure(figsize=(12, 2))
    for tr in traces: plt.plot(tr, color='red', alpha=0.25)
    plt.plot(np.median(traces, axis=0), color='blue', lw=2)
    plt.title(title)
    plt.grid(True); plt.show()

def format_lag_metrics(metrics, central_band=1):
    """
    Compact string for titles, e.g.:
    'Lag: MAD=0.8, central±1 LB=0.72, edge=5.3% (N=446)'
    """
    if metrics["N"] == 0:
        return "Lag: N=0"
    return (f"Lag: MAD={metrics['mad']:.2f}, "
            f"central±{central_band} LB={metrics['central_LB']:.2f}, "
            f"edge={100*metrics['edge_frac']:.1f}% (N={metrics['N']})")



# ==== Global K-means discovery helpers =======================================
import numpy as np
from typing import Optional, Dict, Any, Sequence, Tuple
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Uses existing helpers defined in this module:
# - median_ei_adaptive(snips)          [C,T,N] -> [C,T]
# - main_channel_and_neg_peak(ei)      -> (c_best, t_best)

def _flatten_snips(snips: np.ndarray) -> np.ndarray:
    """
    snips: [C,T,N] -> X: [N, C*T] (copy)
    """
    C, T, N = snips.shape
    return snips.transpose(2, 0, 1).reshape(N, C*T).copy()

def build_global_pcs(snips: np.ndarray, n_pcs: int, *, center: bool = True,
                     random_state: Optional[int] = 42) -> Tuple[np.ndarray, PCA]:
    """
    Flatten [C,T,N] spikes, (optionally) column-center, then randomized PCA.
    Returns (pcs [N, n_pcs], fitted PCA object).
    """
    X = _flatten_snips(snips)  # [N, C*T]
    if center:
        X -= X.mean(axis=0, keepdims=True)
    n_pcs = int(min(n_pcs, X.shape[0]-1, X.shape[1]))
    if n_pcs < 2:
        raise ValueError(f"n_pcs too small for PCA: {n_pcs}")
    pca = PCA(n_components=n_pcs, svd_solver='randomized', random_state=random_state)
    pcs = pca.fit_transform(X)
    return pcs, pca

def kmeans_labels(pcs: np.ndarray, k: int, *, n_init: int = 10,
                  random_state: Optional[int] = 42) -> Tuple[np.ndarray, KMeans]:
    """
    Run KMeans on pcs [N, d]. Returns (labels [N], model).
    """
    k = int(max(2, min(k, pcs.shape[0])))  # guard rails
    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(pcs)
    return labels, km

def _cluster_snr_from_ei(ei: np.ndarray) -> float:
    """SNR proxy: max(p2p) / median(p2p)."""
    p2p = (ei.max(axis=1) - ei.min(axis=1)).astype(np.float32)
    med = float(np.median(p2p)) if p2p.size else 1e-9
    return float(p2p.max() / max(med, 1e-9))

def choose_cluster_by_amp_snr(
    snips: np.ndarray,
    labels: np.ndarray,
    *,
    channel_of_interest: int,
    sample_idx: int = 40,
    ampl_thr: float = -100.0,
    min_cluster_size: int = 10,
    time_ok: Tuple[int, int] = (20, 100)
) -> Optional[Dict[str, Any]]:
    """
    Among clusters with size ≥ min_cluster_size and median amp@c*,sample ≤ ampl_thr,
    pick argmax of score = (−median_amp@c*,sample) × SNR(EI).
    Returns dict with:
      cand_mask, ei_cand, cid, snr, median_amp40, c_best, t_best
    or None if no cluster passes the gates.
    """
    N = snips.shape[2]
    C, T = snips.shape[0], snips.shape[1]
    unique, counts = np.unique(labels, return_counts=True)
    best = None

    for cid, cnt in zip(unique.tolist(), counts.tolist()):
        if cnt < min_cluster_size:
            continue
        cmask = (labels == cid)
        # median amplitude on the detection channel at the reference sample
        vals = snips[channel_of_interest, sample_idx, cmask]
        med_amp = float(np.median(vals)) if vals.size else np.inf
        if med_amp > ampl_thr:
            continue

        # EI + SNR
        ei = median_ei_adaptive(snips[:, :, cmask])
        snr = _cluster_snr_from_ei(ei)

        # derive (c_best, t_best) from EI itself; keep time in allowed band
        c_best, t_best = main_channel_and_neg_peak(ei)
        if (t_best < time_ok[0]) or (t_best > time_ok[1]):
            continue

        score = (-med_amp) * snr
        candidate = dict(
            cand_mask=cmask, ei_cand=ei, cid=int(cid),
            snr=float(snr), median_amp40=float(med_amp),
            c_best=int(c_best), t_best=int(t_best)
        )
        if (best is None) or (score > best[0]):
            best = (score, candidate)

    return best[1] if best is not None else None

def kmeans_discover_global(
    snips: np.ndarray,
    *,
    channel_of_interest: int,
    cfg,
    seeds: Sequence[int] = (42, 7, 113),
    k_list: Optional[Sequence[int]] = None,
    pc_list: Sequence[int] = (80, 64, 100),
    verbose: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Wrapper that tries (pc_dim × seed × k) in order until a valid cluster is found.
    Returns dict with keys:
      'cand_mask','ei_cand','labels','pcs','cid','k','pc_dim','seed','snr','median_amp40','c_best','t_best'
    or None if no candidate passes the gates.
    """
    N = snips.shape[2]
    min_k_by_size = max(3, N // max(cfg.MIN_CLUSTER_SIZE, 1))
    default_k = int(min(cfg.KM_K_DEFAULT, min_k_by_size))
    k_list = list(k_list) if k_list is not None else [default_k, max(3, default_k-2), default_k+2, default_k+6]

    for pc_dim in pc_list:
        try:
            pcs, pca = build_global_pcs(snips, n_pcs=int(pc_dim), center=True, random_state=seeds[0] if len(seeds) else 42)
        except Exception as e:
            if verbose:
                print(f"[kmeans_discover] skipping PCs={pc_dim}: {e}")
            continue

        for seed in seeds:
            for k in k_list:
                try:
                    labels, km = kmeans_labels(pcs, k=int(k), n_init=int(cfg.KM_N_INIT), random_state=int(seed))
                except Exception as e:
                    if verbose:
                        print(f"[kmeans_discover] KMeans failed (k={k}, seed={seed}): {e}")
                    continue

                chosen = choose_cluster_by_amp_snr(
                    snips, labels,
                    channel_of_interest=channel_of_interest,
                    sample_idx=int(cfg.AMP_POST_SAMPLE),
                    ampl_thr=float(cfg.AMPL_THR),
                    min_cluster_size=int(cfg.MIN_CLUSTER_SIZE),
                    time_ok=(20, 100)
                )
                if chosen is None:
                    continue

                # success
                out = dict(chosen)
                out.update({
                    "labels": labels,
                    "pcs": pcs,
                    "k": int(k),
                    "pc_dim": int(pc_dim),
                    "seed": int(seed)
                })
                if verbose:
                    cid = out["cid"]; n = int(np.sum(out["cand_mask"]))
                    print(f"[kmeans_discover] success: PCs={pc_dim}, k={k}, seed={seed}, cid={cid}, size={n}, "
                          f"SNR={out['snr']:.1f}, med_amp40={out['median_amp40']:.1f}")
                return out

        if verbose:
            print(f"[kmeans_discover] no valid cluster with PCs={pc_dim}; trying next PC setting.")

    if verbose:
        print("[kmeans_discover] no acceptable cluster found across all PCs/seeds/k.")
    return None


