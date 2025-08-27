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

try:
    import torch
    from collision_cuda_clean import (
        build_rolled_bank, quick_unit_filter_t, scan_unit_lags_t,
        score_active_set_t, MAX_LAG, TOP_K
    )
    USE_CUDA = torch.cuda.is_available()
except ImportError:
    USE_CUDA = False     # fallback to old numpy helpers


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


# ================================================================
# unified ΔRMS scorer — switches between CPU (NumPy) and CUDA
# ================================================================
def make_scorer(use_cuda: bool,
                rolled_EI_t=None, raw_snip_t=None, uid2row=None):
    """
    Returns a function scorer(active_dict, union_chans, raw_local, unit_info, p2p_all)
    that computes Σ(RMS_raw - RMS_res) with CPU or CUDA automatically.
    """
    if not use_cuda:
        from collision_utils import score_active_set as score_cpu
        def cpu_sc(active_dict, union_chans, raw_local, unit_info, p2p_all):
            return score_cpu(active_dict, union_chans, raw_local,
                             unit_info, p2p_all)
        return cpu_sc

    # ------------- CUDA branch ----------------
    from collision_cuda_clean import score_active_set_t

    def cuda_sc(active_dict, union_chans, _raw_local, unit_info, _p2p):
        # build boolean mask [1,U,41]   (U = rolled_EI_t.shape[0])
        mask = torch.zeros(rolled_EI_t.shape[0], 41,
                           device=rolled_EI_t.device, dtype=torch.bool)
        for uid, lag in active_dict.items():
            mask[uid2row[uid], lag] = True
        mask = mask.unsqueeze(0)                 # [1,U,41]
        return score_active_set_t(mask, rolled_EI_t, raw_snip_t)[0].item()
    return cuda_sc


# ------------------------------------------------------------------
# 0.  FAST PEAK‑CHANNEL FILTER (new)
# ------------------------------------------------------------------
def quick_unit_filter(
    unit_ids,
    raw_snippet: np.ndarray,
    unit_info: dict,
    *,
    delta_thr: float = 0.0,
    max_lag: int = 60,
):
    """Fast x-corr gate; behaviour now identical to the original inline code."""
    rows = []

    snip_len = raw_snippet.shape[1]

    for uid in unit_ids:
        ei       = unit_info[uid]['ei']
        peak_ch  = unit_info[uid]['peak_channel']
        sel_ch   = unit_info[uid]['selected_channels']

        if len(sel_ch) == 0:
            # print(f"Skipping {uid}: no selected_channels")
            continue

        trace_rw = raw_snippet[peak_ch]
        trace_ei = ei[peak_ch]

        # raw lag from full x-corr
        xcor   = correlate(trace_rw, trace_ei, mode="full")
        lags   = np.arange(-len(trace_rw) + 1, len(trace_rw))
        lag_raw = lags[np.argmax(xcor)]


        # ----- asymmetric clipping so EI peak lands in 40-80 -----
        p_idx     = np.argmax(np.abs(trace_ei))        # or store once per unit
        base      = snip_len//2 - p_idx                # 19 when p_idx=41
        lag_low   = base - max_lag                     # -1
        lag_high  = base + max_lag                     # 39
        lag0      = int(np.clip(lag_raw, lag_low, lag_high))
        # ---------------------------------------------------------

        aligned  = np.roll(ei[sel_ch], lag0, axis=1)

        # RMS on *all* samples of selected channels  (no mask)
        raw_sel = raw_snippet[sel_ch]
        weights = aligned.max(axis=1) - aligned.min(axis=1)
        weights[weights>200] = 200 # clips weights

        delta = delta_rms(raw_sel, aligned, weights)

        # rms_pre  = sum(np.sqrt(np.mean(raw_sel**2, axis=1)))
        # rms_post = sum(np.sqrt(np.mean((raw_sel - aligned)**2, axis=1)))
        # delta    = rms_post - rms_pre

        # if uid=='unit_582':
        #     print(f"delta for 582: {delta}, =lag {lag0}")


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

def build_channel_index(good_df: pd.DataFrame, unit_info: dict):
    """Return {channel: [uids]} for the surviving units."""

    from collections import defaultdict
    ch_map = defaultdict(list)
    for uid in good_df['uid']:
        for ch in unit_info[uid]['selected_channels']:
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
    peak_idx  = 40 # EI center - convention

    SNIP_LEN = raw_snippet.shape[1]
    mid   = SNIP_LEN // 2          # 60
    base  = mid - peak_idx         # lag that centres the EI peak at 60
    # legal lag window so spike peak ∈ [40,80]
    lag_min = base - max_lag       # 60-peak_idx-max_lag
    lag_max = base + max_lag       # 60-peak_idx+max_lag
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

        delta = delta_rms(raw_sel, shifted_ei, weights)

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
                mask_thr=mask_thr, max_lag=max_lag
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
    *,
    mask_thr: float = 5.0,
    mask_array: np.ndarray = None,        # optional [C_union, T] boolean array
    beta: float = 0.5,
) -> float:
    """
    Return weighted ΔRMS for the given unit set.
    """
    if not active_dict:
        return 0.0

    # def per_channel_deltas(unit_set: Dict[int, int]) -> list[tuple[int, float, float, float]]:
    #     """Return [(channel, delta, rms_raw, rms_res)]"""
    #     tmpl_sum = np.zeros_like(raw_local, dtype=np.float32)
    #     for u, lag in unit_set.items():
    #         shifted = np.array(
    #             [roll_zero(unit_info[u]["ei"][c], lag) for c in union_chans]
    #         )
    #         tmpl_sum += shifted

    #     results = []
    #     for k, c in enumerate(union_chans):
    #         if mask_array is not None:
    #             m = mask_array[k]
    #         else:
    #             m = np.abs(tmpl_sum[k]) > mask_thr

    #         if not m.any():
    #             continue
    #         rms_raw = np.sqrt((raw_local[k, m] ** 2).mean())
    #         rms_res = np.sqrt(((raw_local[k, m] - tmpl_sum[k, m]) ** 2).mean())

    #         # NOTE: `u` is not defined inside this loop anymore
    #         # If you want to use `a_u`, clarify: which unit's P2P to use?
    #         # For now, fallback to max over contributing units:
    #         a_u = max(p2p_all[u][c] for u in unit_set)
    #         delta = (a_u ** beta) * (rms_raw - rms_res)
    #         results.append((c, delta, rms_raw, rms_res))
    #     return results
    


    tmpl_sum = np.zeros_like(raw_local, dtype=np.float32)
    for u, lag in active_dict.items():
        shifted = np.array(
            [roll_zero(unit_info[u]["ei"][c], lag) for c in union_chans]
        )
        tmpl_sum += shifted

    # RMS on *all* samples of selected channels  (no mask)
    # weights = tmpl_sum.max(axis=1) - tmpl_sum.min(axis=1)
    # weights[weights>200] = 200 # clips weights
    weights = np.zeros(len(union_chans), dtype=np.float32)
    for k, c in enumerate(union_chans):
        a_u = max(p2p_all[u][c] for u in active_dict)
        weights[k] = a_u ** beta
    delta = delta_rms(raw_local, tmpl_sum, weights)

    score = -delta
    # deltas_base = per_channel_deltas(active_dict)
    # score       = sum(d for _, d, _, _ in deltas_base)


    return score


# ------------------------------------------------------------------
# STEP 2 – GREEDY BEAM SEARCH  (no additive assumption needed)
# ------------------------------------------------------------------
# =============================================================
def beam_combo_search(units, lag_dict, union_chans, raw_local,
                      unit_info, p2p_all, beta, beam, scorer_fn):
    """
    Greedy beam search over {unit × 3 lags}.
    Returns {'lags': {uid:lag}, 'score': ΔRMS}
    """
    combos = [({}, 0.0)]        # list of (lag_dict, score)
    for uid in units:
        new_combos = []
        for l in lag_dict[uid]:
            for base, sc in combos:
                trial = dict(base); trial[uid] = l
                score = scorer_fn(trial, union_chans,
                                  raw_local, unit_info, p2p_all)
                new_combos.append((trial, score))
        # keep best `beam` combos
        new_combos.sort(key=lambda x: x[1], reverse=True)
        combos = new_combos[:beam]
    best = max(combos, key=lambda x: x[1])
    return {'lags': best[0], 'score': best[1]}

# =============================================================
def prune_combo(lag_map, union_chans, raw_local,
                unit_info, p2p_all, scorer_fn):
    """
    Greedy backward elimination: drop units that don’t hurt the score.
    """
    changed = True
    while changed and lag_map:
        changed = False
        base = scorer_fn(lag_map, union_chans, raw_local, unit_info, p2p_all)
        for uid in list(lag_map):
            trial = dict(lag_map); trial.pop(uid)
            if scorer_fn(trial, union_chans,
                         raw_local, unit_info, p2p_all) >= base:
                lag_map.pop(uid); changed = True; break
    return lag_map
# =============================================================

# ----------------------------------------------------------------------
# Beam-search version – returns TWO objects to match resolve_snippet()
# ----------------------------------------------------------------------
def evaluate_local_group(c0, working_units, raw_snippet, unit_info, lag_dict,
                         p2p_all, amp_thr, beta,
                         scorer_fn,
                         use_cuda=False,
                         rolled_EI_t=None, raw_snip_t=None, uid2row=None):
    """
    c0              : anchor channel
    working_units   : list[str]   candidate units on this anchor
    scorer_fn       : callable returned by make_scorer()
    Returns
    -------
    best_combo      : {'lags': {uid:lag}, 'score': ΔRMS}
    per_unit_delta  : {uid: ΔRMS vs empty set}
    """
    # ---------- union of “strong” channels ---------------------------
    union = {c0}
    for u in working_units:
        union.update(np.where(p2p_all[u] >= amp_thr)[0])
    union_chans = sorted(union)
    raw_local   = raw_snippet[union_chans]

    # ---------- beam search -----------------------------------------
    best_combo = beam_combo_search(
        units       = working_units,
        lag_dict    = lag_dict,
        union_chans = union_chans,
        raw_local   = raw_local,
        unit_info   = unit_info,
        p2p_all     = p2p_all,
        beta        = beta,
        beam        = 4,
        scorer_fn   = scorer_fn,        # ← only change
    )

    # ---------- marginal prune --------------------------------------
    pruned = prune_combo(best_combo['lags'], union_chans,
                         raw_local, unit_info, p2p_all,
                         scorer_fn=scorer_fn)

    best_combo['lags']  = pruned
    best_combo['score'] = scorer_fn(pruned, union_chans,
                                    raw_local, unit_info, p2p_all)

    # ---------- per-unit marginal ΔRMS ------------------------------
    full_score = best_combo['score']
    per_unit_delta = {u: (full_score if u in pruned else 0.0)
                      for u in working_units}
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
# ───── replace the whole old function with this one ──────────
# ───────────────── resolve_snippet (clean) ──────────────────
def resolve_snippet_t(
        raw_snippet: np.ndarray,
        good_units: Sequence[str],
        channel_to_units: Dict[int, List[str]],
        lag_dict: Dict[str, List[int]],
        unit_info: Dict[str, Dict],
        p2p_all: Dict[str, np.ndarray],
        *,
        amp_thr: float = 25.0,
        beta: float = 0.5,
        use_cuda: bool = False,
        rolled_t=None,            # [U,41,C,T] torch
        raw_snip_t=None,          # [C,T]      torch
        uid2row: Dict[str, int] = None
):
    """
    Decide which candidate units + lags best explain one 121-sample snippet.
    CUDA heavy work is used when `use_cuda=True`.
    Returns
    -------
    best_combo_global : {"lags": {uid:lag}, "score": ΔRMS}
    per_unit_delta    : {uid: marginal ΔRMS}
    combo_history     : list of best_combo after each anchor iteration
    """
    from collections import defaultdict
    import math, numpy as np

    # ---------- unified scorer (CPU / CUDA) --------------------------
    scorer = make_scorer(use_cuda, rolled_t, raw_snip_t, uid2row)

    # ---------- bookkeeping -----------------------------------------
    raw_ptp = raw_snippet.ptp(axis=1)
    unresolved = {ch for uid in good_units
                     for ch in unit_info[uid]["selected_channels"]}

    unit_log = defaultdict(lambda: {"deltas": [], "lags": []})
    combo_history = []

    # ---------- unresolved-channel loop ------------------------------
    while unresolved:
        c0 = max(unresolved, key=lambda c: raw_ptp[c])
        W_full = channel_to_units[c0]

        W = [u for u in W_full if not is_certain(u, unit_log)]
        if len(W) > MAX_W_UNITS:
            W.sort(key=lambda u: p2p_all[u][c0], reverse=True)
            W = W[:MAX_W_UNITS]
        if not W:
            W = [max(W_full, key=lambda u: p2p_all[u][c0])]

        # ---- beam search on this anchor -----------------------------
        best_combo, per_delta_anchor = evaluate_local_group(
                c0, W, raw_snippet, unit_info, lag_dict, p2p_all,
                amp_thr, beta,
                scorer_fn = scorer,
                use_cuda  = use_cuda,
                rolled_EI_t = rolled_t,
                raw_snip_t  = raw_snip_t,
                uid2row     = uid2row)
        combo_history.append(best_combo)

        for uid, d in per_delta_anchor.items():
            unit_log[uid]["deltas"].append(d)
            unit_log[uid]["lags"].append(best_combo["lags"].get(uid, math.nan))

        # ---- update unresolved-channel list -------------------------
        Wset = set(W)
        for ch in list(unresolved):
            remaining = [u for u in channel_to_units[ch]
                           if not is_certain(u, unit_log)]
            if set(remaining).issubset(Wset):
                unresolved.remove(ch)

    # ---------- aggregate lags per unit ------------------------------
    agg = {u: float(np.nanmedian(rec["lags"]))
           for u, rec in unit_log.items()
           if not np.all(np.isnan(rec["lags"]))}
    active = {u: int(round(lag)) for u, lag in agg.items()
              if not np.isnan(lag)}

    # ---------- build local scorer tied to this union of channels ----
    union_chans = sorted({c for u in active
                            for c in unit_info[u]["selected_channels"]})
    raw_local   = raw_snippet[union_chans]

    def local_score(lag_map):
        return scorer(lag_map, union_chans,
                      raw_local, unit_info, p2p_all)

    # ---------- marginal prune --------------------------------------
    def marginal_prune(lag_map):
        changed = True
        while changed and lag_map:
            changed = False
            base = local_score(lag_map)
            for uid in list(lag_map):
                trial = dict(lag_map); trial.pop(uid)
                if local_score(trial) >= base:
                    lag_map.pop(uid); changed = True; break
        return lag_map

    pruned = marginal_prune(active)

    # ---------- final score & per-unit gains -------------------------
    score_full = local_score(pruned)
    per_unit_delta = {}
    for uid in pruned:
        trial = dict(pruned); trial.pop(uid)
        per_unit_delta[uid] = score_full - local_score(trial)

    return {"lags": pruned, "score": score_full}, per_unit_delta, combo_history
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────


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
    pos_min: float = 200.0,
    net_min: float = 50.0,
    h_max: float = 0.3,
    lag_mad_max: float = 2,
) -> Tuple[List[int], List[int]]:
    """Return (accepted_uids, rejected_uids) based on hard thresholds."""
    accepted = []
    for uid, s in stats.items():
        P = s["delta_pos"]
        N = s["delta_neg"]
        H = abs(N) / P if P else np.inf
        net = P + N
        if P >= pos_min and net >= net_min and H <= h_max and s["lag_mad"] <= lag_mad_max:
            accepted.append(uid)
    rejected = [u for u in stats if u not in accepted]
    return accepted, rejected



def micro_align_units(
    accepted: Sequence[int],
    stats: Dict[int, Dict],
    unit_info: Dict[int, Dict],
    raw_snippet: np.ndarray,
    p2p_all: Dict[int, np.ndarray],
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
            score = score_active_set({uid: lag}, sel_ch, raw_snippet[sel_ch], unit_info, p2p_all, mask_thr=mask_thr, beta=beta)

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

