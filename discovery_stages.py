"""
Discovery stages

Linear, stage-based pipeline with a single shared Core Template Evaluation (CTE).
This file contains concrete implementations based on the agreed design.

STAGES
- Stage 1: REUSE       → iterate existing global templates that are strong on this channel
- Stage 2: DISCOVERY   → yield proto-templates from your bimodality/GMM candidate engine
- Stage 3: AUXILIARY   → try a small number of residual-based protos (e.g., median EI)

SHARED BLOCK
- Core Template Evaluation (CTE): steps 3–12 + 15 from the stage table
  (align→select channels→harm-map→ideal→per-spike gate→rebuild EI→per-template gate
   → subtract & trim → dedup → optional plotting)

DATA SHAPES (conventions)
- ei:            np.ndarray, shape [C, T]
- snips_pool:    np.ndarray, shape [C, T, N]
- spike_times:   np.ndarray, shape [N], dtype=int64 (aligned to detect_channel)
- selected_channels: np.ndarray, shape [n_sel], dtype=int
- GLOBAL_TEMPLATES: list[template_dict]
- template_dict keys (minimal):
    {
      "ei": [C, T],
      "spike_times": int64[],          # aligned to detect_channel for this record
      "detect_channel": int,           # where spikes are centered at center_sample
      "peak_channel": int,             # channel with max p2p
      "selected_channels": int[],      # channels used for ΔRMS scoring
      "p2p": float[],                  # per-channel p2p of EI
      "gbm": float,
      "snr": float,
      # optional: "detect_channels_log": list[(channel, n_peeled)]
    }

PARAMETERS (expected keys in `params` dict)
- General:    center_sample, show_plots, verbose, ei_positions (optional for plotting)
- Channel sel: p2p_thr, max_channels, min_channels
- Harm-map:   lag_radius, weight_by_p2p, weight_beta
- Per-spike gate: thr_global, thr_channel, min_good_frac, max_bad_delta,
                  trusted_good_thresh, trusted_top_frac, exceed_thresh
- Per-template gate: min_snr, min_n_reuse, min_n_discovery, min_n_aux,
                     lag_mad_max, lag_central_lb_min, lag_edge_frac_max,
                     lag_central_band (±samples), lag_wilson_z
- Residual trim: amp_post_sample, amp_post_thr
- Reuse: reuse_p2p_thr
- Aux:   max_aux_rounds
- Dedup: ei_sim_thr, ei_sim_max_lag
- Jitter (per-channel timing coherence):
    jitter_search_radius (default 3),
    jitter_central_band (default 1),
    jitter_p2p_thr (default 100.0), jitter_min_channels (default 30), jitter_include_proto (default True),
    jitter_edge_any_max (default 0.30),
    jitter_edge_many_max (default 0.15), jitter_edge_many_count (default 3),
    jitter_central_any_min (default 0.50),
    jitter_central_many_min (default 0.70), jitter_central_many_count (default 3)

All plotting happens only *after* acceptance inside CTE and is toggled by params['show_plots'].
"""
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
import numpy as np

# -----------------------------
# Stage-specific candidate providers
# -----------------------------

def iter_reuse_protos(
    global_templates: List[Dict[str, Any]],
    channel_of_interest: int,
    params: Dict[str, Any],
) -> Iterator[Dict[str, Any]]:
    """Yield proto-templates for Stage 1 (REUSE).

    Filters `global_templates` by per-channel p2p on `channel_of_interest` and yields
    strongest-first. Does not mutate any inputs.
    """
    thr = float(params.get("reuse_p2p_thr", 100.0))
    if not global_templates:
        return iter(())
    # Rank by strength on this channel
    scored: List[Tuple[float, int]] = []
    for i, rec in enumerate(global_templates):
        p2p = rec.get("p2p", None)
        if p2p is None or len(p2p) <= channel_of_interest:
            continue
        val = float(p2p[channel_of_interest])
        if val >= thr:
            scored.append((val, i))
    scored.sort(reverse=True)

    def _gen() -> Iterator[Dict[str, Any]]:
        for _val, idx in scored:
            rec = global_templates[idx]
            yield {
                "stage": "reuse",
                "ei": rec["ei"],
                "source_index": idx,
                "detect_channel_hint": channel_of_interest,
            }
    return _gen()


def next_discovery_proto(
    snips_pool: np.ndarray,
    candidate_state: Dict[str, Any],
    params: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Produce the next proto-template for Stage 2 (DISCOVERY).

    This is a thin adapter so you can plug in *your* candidate engine without rewriting it.
    Supported patterns in `candidate_state`:
      1) a callable under key 'get_next_candidate' with signature
         get_next_candidate(snips_pool: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray [C,T]]
         (returns a proto EI or None)
      2) a precomputed list under key 'candidates' (list of EIs) plus an integer 'cursor'.

    Returns a proto dict {'stage': 'discovery', 'ei': EI, 'meta': {...}} or None if exhausted.
    """
    if snips_pool is None or snips_pool.size == 0:
        return None

    # Pattern 1: user-supplied callable
    fn = candidate_state.get("get_next_candidate", None)
    if callable(fn):
        ei = fn(snips_pool, params)
        if ei is None:
            return None
        return {"stage": "discovery", "ei": ei, "meta": {"source": "callback"}}

    # Pattern 2: precomputed list + cursor
    cand_list: Optional[List[np.ndarray]] = candidate_state.get("candidates")
    if cand_list is not None:
        cur = int(candidate_state.get("cursor", 0))
        while cur < len(cand_list) and (cand_list[cur] is None):
            cur += 1
        if cur >= len(cand_list):
            return None
        ei = cand_list[cur]
        candidate_state["cursor"] = cur + 1
        return {"stage": "discovery", "ei": ei, "meta": {"index": cur}}

    # Otherwise we don't know how to generate a discovery proto
    return None


def build_aux_proto(
    snips_pool: np.ndarray,
    params: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build a proto-template for Stage 3 (AUXILIARY).

    Strategy: if the pool is large enough, take median EI of either the full pool
    or the top-K by detect-channel amplitude (optional future tweak).
    """
    if snips_pool is None or snips_pool.size == 0:
        return None
    C, T, N = snips_pool.shape
    min_n_aux = int(params.get("min_n_aux", 12))
    if N < max(8, min_n_aux):
        return None
    try:
        from collision_utils import median_ei_adaptive
    except Exception:
        # Fallback: plain median
        ei = np.median(snips_pool, axis=2)
    else:
        ei = median_ei_adaptive(snips_pool)
    return {"stage": "aux", "ei": ei, "meta": {"N": int(N)}}


# -----------------------------
# Core Template Evaluation (shared steps 3–12 + 15)
# -----------------------------

def core_template_evaluation(
    proto_ei: np.ndarray,
    channel_of_interest: int,
    snips_pool: np.ndarray,
    spike_times_pool: np.ndarray,
    params: Dict[str, Any],
    global_templates: List[Dict[str, Any]],
    stage: str,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Shared evaluation pipeline for a single proto-template.
    See module header for detailed behavior.
    """
    result = {
        "decision": "rejected",
        "reason": None,
        "accept_mask": None,
        "ei_final": None,
        "diag": {"snr": np.nan, "n_spikes": 0, "global_mean_delta": None,
                  "lag": {"mad": np.nan, "central_LB": np.nan, "edge_frac": np.nan},
                  "channel_selection": {}},
        "dedup_action": None,
        "accepted_record": None,
    }

    # Quick exits
    if snips_pool is None or snips_pool.size == 0:
        result["reason"] = "empty_pool"
        return result, snips_pool, spike_times_pool
    if proto_ei is None or proto_ei.size == 0:
        result["reason"] = "empty_proto"
        return result, snips_pool, spike_times_pool

    C, T, N = snips_pool.shape
    center = int(params.get("center_sample", 40))

    # ---- helpers (local) ----
    def roll_zero_1d(a: np.ndarray, s: int) -> np.ndarray:
        if s == 0:
            return a
        out = np.zeros_like(a)
        if s > 0:
            out[s:] = a[:-s]
        else:
            out[:s] = a[-s:]
        return out

    def roll_zero_all(ei2d: np.ndarray, shift: int) -> np.ndarray:
        return np.vstack([roll_zero_1d(ei2d[c], shift) for c in range(ei2d.shape[0])])

    # ---- 1) Align proto EI to this detect channel ----
    t_peak = int(np.argmin(proto_ei[channel_of_interest]))
    shift = center - t_peak
    ei_aligned = roll_zero_all(proto_ei, shift)

    # ---- 2) Harm-map, channels & lags ----
    from collision_utils import compute_harm_map_noamp, build_ideal_delta, compute_spike_gate, median_ei_adaptive, compute_global_baseline_mean
    try:
        res = compute_harm_map_noamp(
            ei_aligned,
            snips_pool,
            p2p_thr=float(params.get("p2p_thr", 50.0)),
            max_channels=int(params.get("max_channels", 80)),
            min_channels=int(params.get("min_channels", 10)),
            lag_radius=int(params.get("lag_radius", 3)),
            weight_by_p2p=bool(params.get("weight_by_p2p", True)),
            weight_beta=float(params.get("weight_beta", 0.7)),
        )
    except TypeError:
        # Backward-compat in case signature differs
        res = compute_harm_map_noamp(ei_aligned, snips_pool)

    H = res.get("harm_matrix")  # [nch, N]
    if H is None or H.size == 0:
        result["reason"] = "harm_map_failed"
        return result, snips_pool, spike_times_pool

    # ---- 3) Ideal per-channel Δ from trusted spikes ----
    ideal = build_ideal_delta(
        res,
        good_thresh=float(params.get("trusted_good_thresh", -5.0)),
        top_frac=float(params.get("trusted_top_frac", 0.25)),
    )

    # ---- 4) Per-spike gate → accept_mask ----
    gate_kwargs = dict(
        thr_global=float(params.get("thr_global", -2.0)),
        thr_channel=float(params.get("thr_channel", 0.0)),
        min_good_frac=float(params.get("min_good_frac", 0.45)),
        max_bad_delta=float(params.get("max_bad_delta", 10.0)),
        weighted=True,
        weight_beta=float(params.get("weight_beta", 0.7)),
    )
    # Try newer signature with ideal/exceed
    try:
        gate = compute_spike_gate(
            res,
            ideal=ideal,
            exceed_thresh=float(params.get("exceed_thresh", 20.0)),
            **gate_kwargs,
        )
    except TypeError:
        gate = compute_spike_gate(res, **gate_kwargs)

    acc = np.asarray(gate.get("accept_mask"), bool)
    n_acc = int(acc.sum()) if acc.size else 0

    # Minimal N by stage
    min_n_map = {
        "reuse": int(params.get("min_n_reuse", 20)),
        "discovery": int(params.get("min_n_discovery", 10)),
        "aux": int(params.get("min_n_aux", 12)),
    }
    if n_acc < min_n_map.get(stage, 10):
        result["reason"] = f"too_few_spikes:{n_acc}"
        return result, snips_pool, spike_times_pool

    # ---- 5) Rebuild EI on accepted spikes ----
    snips_cand = snips_pool[:, :, acc]
    ei_final = median_ei_adaptive(snips_cand)

    # Diagnostics
    p2p = ei_final.max(axis=1) - ei_final.min(axis=1)
    snr = float(p2p.max() / max(np.median(p2p), 1e-9))
    peak_ch = int(np.argmax(p2p))
    gbm = float(compute_global_baseline_mean(ei_final))

    # Recompute channel selection on the FINAL EI (supersedes proto selection)
    final_sel = select_channels_for_scoring(ei_final, params)

    # ---- 6) Per-template gate ----
    min_snr = float(params.get("min_snr", 8.0))
    if snr < min_snr:
        result["reason"] = f"snr_low:{snr:.2f}"
        return result, snips_pool, spike_times_pool

    # Lag health
    lag_metrics = compute_lag_health(
        np.asarray(res.get("best_lag_per_spike")),
        np.asarray(res.get("lags")),
        acc,
        params,
    )
    if (
        (lag_metrics["mad"] >= float(params.get("lag_mad_max", 1.5)))
        or (lag_metrics["central_LB"] < float(params.get("lag_central_lb_min", 0.5)))
        or (lag_metrics["edge_frac"] > float(params.get("lag_edge_frac_max", 0.25)))
    ):
        result["reason"] = "lag_health_fail"
        return result, snips_pool, spike_times_pool

    # ---- 6b) Per-channel timing coherence (jitter) gate on strong channels ----
    jit_p2p_thr = float(params.get("jitter_p2p_thr", 100.0))
    # Base set: channels strong in FINAL EI
    ch_strong = np.where(p2p > jit_p2p_thr)[0]
    # Optionally union with proto-selected channels (captures informative B-channels that shrank post-rebuild)
    if bool(params.get("jitter_include_proto", True)):
        ch_proto = np.asarray(res.get("selected_channels", []), dtype=int)
        ch_proto = ch_proto[(ch_proto >= 0) & (ch_proto < C)] if ch_proto.size else ch_proto
        if ch_proto.size:
            ch_strong = np.union1d(ch_strong, ch_proto)
    # Ensure we test at least K channels by topping up with top-p2p
    minK = int(params.get("jitter_min_channels", 10))
    if ch_strong.size < minK:
        topk = np.argsort(-p2p)[:minK]
        ch_strong = np.union1d(ch_strong, topk)

    if ch_strong.size:
        acc_idx = np.where(acc)[0]
        # best_lags_all = np.asarray(res.get("best_lag_per_spike"), int)
        # For jitter, anchor to detect-channel alignment (parity with post-hoc)
        best_lags_acc = np.zeros(acc_idx.size, dtype=int)

        jitter = compute_channel_jitter_stats(
            snips_cand, ei_final, best_lags_acc, ch_strong,
            center_sample=int(params.get("center_sample", 40)),
            search_radius=int(params.get("jitter_search_radius", 3)),
            central_band=int(params.get("jitter_central_band", 1)),
        )
        cf = jitter["central_frac"]  # [n_ch]
        ef = jitter["edge_frac"]

        # Thresholds
        edge_any_max    = float(params.get("jitter_edge_any_max", 0.30))
        edge_many_max   = float(params.get("jitter_edge_many_max", 0.15))
        edge_many_count = int(params.get("jitter_edge_many_count", 3))
        cent_any_min    = float(params.get("jitter_central_any_min", 0.50))
        cent_many_min   = float(params.get("jitter_central_many_min", 0.70))
        cent_many_count = int(params.get("jitter_central_many_count", 3))
        if (
            (np.any(ef > edge_any_max))
            or (np.sum(ef > edge_many_max) >= edge_many_count)
            or (np.any(cf < cent_any_min))
            or (np.sum(cf < cent_many_min) >= cent_many_count)
        ):
            result["reason"] = "jitter_reject"
            result["diag"]["jitter"] = {
                "channels": ch_strong.tolist(),
                "central_frac": cf.tolist(),
                "edge_frac": ef.tolist(),
            }
            return result, snips_pool, spike_times_pool


    # ---- 7) Accept: subtract with per-spike lags; trim by post-amp ----

    # print("Central frac:", np.round(cf, 2))
    # print("Edge frac:",    np.round(ef, 2))
    # print(f"channels: {ch_strong}")
    # print(f"lags: {best_lags_acc}")


    # Cache pre-subtraction traces for plotting on detect channel and peak channel
    acc_idx = np.where(acc)[0]
    trace_detect_pre = None
    trace_peak_pre = None
    if acc_idx.size > 0:
        try:
            trace_detect_pre = snips_pool[channel_of_interest, :, acc_idx].copy()  # [T, n_acc]
            trace_peak_pre   = snips_pool[peak_ch, :, acc_idx].copy()              # [T, n_acc]
        except Exception:
            trace_detect_pre = None
            trace_peak_pre = None

    
    best_lags = np.asarray(res.get("best_lag_per_spike"), int)
    acc_idx = np.where(acc)[0]
    for j in acc_idx:
        lag = int(best_lags[j]) if best_lags.size == N else 0
        shifted = np.vstack([roll_zero_1d(ei_final[c], lag) for c in range(C)])
        snips_pool[:, :, j] = snips_pool[:, :, j] - shifted

    # Trim well-explained spikes
    amp_sample = int(params.get("amp_post_sample", 40))
    keep_thr = float(params.get("amp_post_thr", -100.0))
    amp_post = snips_pool[channel_of_interest, amp_sample, :]
    keep_mask = amp_post < keep_thr

    new_snips = snips_pool[:, :, keep_mask]
    new_times = spike_times_pool[keep_mask]

    # ---- 8) Dedup into global set ----
    candidate_record = {
        "ei": ei_final,
        "spike_times": spike_times_pool[acc].copy(),
        "detect_channel": int(channel_of_interest),
        "peak_channel": int(peak_ch),
        "selected_channels": final_sel,
        "p2p": p2p,
        "gbm": gbm,
        "snr": snr,
    }

    action, canonical = dedup_template(candidate_record, global_templates, params)

    # ---- 9) Plots (optional, after acceptance) ----
    if params.get("show_plots", False):
        try:
            from collision_utils import plot_harm_heatmap, plot_spike_delta_summary, plot_help_harm_scatter_swapped
            import matplotlib.pyplot as plt
            try:
                import plot_ei_waveforms as pew
            except Exception:
                pew = None
            mdw = res.get("mean_delta_weighted")
            if mdw is not None and mdw.size:
                idx_in = np.where(acc)[0]; idx_out = np.where(~acc)[0]
                order = np.r_[idx_in[np.argsort(mdw[idx_in])], idx_out[np.argsort(mdw[idx_out])]] if idx_in.size and idx_out.size else np.arange(N)
            else:
                order = np.arange(N)
            vline_at = int(acc.sum())
            plot_harm_heatmap(res, spike_order=order, title=f"{stage.upper()} harm-map", vline_at=vline_at)
            plot_spike_delta_summary(res, weighted=True, title="Per-spike mean ΔRMS (weighted)")
            plot_help_harm_scatter_swapped(res, thr=0.0, spike_order=order, weighted=True,
                                           weight_beta=float(params.get("weight_beta", 0.7)),
                                           big_mask=acc, s_small=14, s_big=64,
                                           title="Mean Δ vs #channels; big markers = accepted")
            # Plot accepted waveforms on detect channel and peak channel (pre-subtraction)
            try:
                if trace_detect_pre is not None:
                    traces = trace_detect_pre  # [n_acc, T]
                    plt.figure(figsize=(12, 3))
                    for tr in traces:
                        plt.plot(tr, color='red', alpha=0.25)
                    plt.plot(np.median(traces, axis=0), color='blue', lw=2)
                    plt.title(f"Detect-ch {channel_of_interest} traces (n={traces.shape[0]})")
                    plt.grid(True); plt.show()
                if trace_peak_pre is not None:
                    traces = trace_peak_pre
                    plt.figure(figsize=(12, 3))
                    for tr in traces:
                        plt.plot(tr, color='red', alpha=0.25)
                    plt.plot(np.median(traces, axis=0), color='blue', lw=2)
                    plt.title(f"Peak-ch {peak_ch} traces (n={traces.shape[0]})")
                    plt.grid(True); plt.show()
            except Exception:
                pass
            if pew is not None:
                plt.figure(figsize=(10, 6))
                pew.plot_ei_waveforms(canonical["ei"], params.get("ei_positions", None),
                                      ref_channel=channel_of_interest, scale=90,
                                      box_height=1, box_width=50, colors='black')
                if action == "kept_new":
                    plt.title(f"Accepted EI ({stage}) SNR {snr:.1f}")
                else:
                    plt.title(f"Merged EI ({stage}) SNR {snr:.1f}")
                plt.show()
        except Exception:
            pass

    if action == "kept_new":
        result.update({
            "decision": "accepted",
            "accept_mask": acc,          # peel spikes from pool
            "ei_final": ei_final,
            "diag": {
                "snr": snr,
                "n_spikes": int(acc.sum()),
                "global_mean_delta": float(np.nanmean(gate.get("global_mean", np.array([np.nan])))),
                "lag": lag_metrics,
                "channel_selection": {
                    "n_channels": int(res.get("harm_matrix").shape[0]) if res.get("harm_matrix") is not None else None,
                },
            },
            "dedup_action": action,
            "accepted_record": canonical # caller may append this
        })
    else:
        # Treat as reject for this round; spikes still peeled, but nothing to append
        result.update({
            "decision": "rejected_merged",
            "accept_mask": acc,          # still peel from pool to avoid rediscovery churn
            "ei_final": ei_final,
            "diag": {
                "snr": snr,
                "n_spikes": int(acc.sum()),
                "global_mean_delta": float(np.nanmean(gate.get("global_mean", np.array([np.nan])))),
                "lag": lag_metrics,
                "channel_selection": {
                    "n_channels": int(res.get("harm_matrix").shape[0]) if res.get("harm_matrix") is not None else None,
                },
            },
            "dedup_action": action
            # NOTE: no "accepted_record" key here
        })
        
    # # ---- Finalize result ----
    # result.update({
    #     "decision": "accepted",
    #     "accept_mask": acc,
    #     "ei_final": ei_final,
    #     "diag": {
    #         "snr": snr,
    #         "n_spikes": int(acc.sum()),
    #         "global_mean_delta": float(np.nanmean(gate.get("global_mean", np.array([np.nan])))),
    #         "lag": lag_metrics,
    #         "channel_selection": {
    #             "n_channels": int(res.get("harm_matrix").shape[0]) if res.get("harm_matrix") is not None else None,
    #         },
    #     },
    #     "dedup_action": action,
    #     "accepted_record": canonical,
    # })


    return result, new_snips, new_times


# -----------------------------
# Deduplication (shared step inside CTE after acceptance)
# -----------------------------

def dedup_template(
    candidate_record: Dict[str, Any],
    global_templates: List[Dict[str, Any]],
    params: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """
    Decide whether the newly accepted `candidate_record` (Y) is a duplicate of an
    existing global template (X). If duplicate, pick a winner (more spikes, then higher SNR),
    merge unique spikes from the loser into the winner (single integer lag), and update
    the global list accordingly.

    Returns:
        (action, canonical)
        action ∈ {"kept_new", "updated_old", "discarded_new"}
            kept_new     -> appended Y as a brand new global entry
            updated_old  -> replaced existing X with Y (Y won), after merging X's unique spikes
            discarded_new-> kept existing X (X won), merged Y's unique spikes into X
        canonical: the winner record (object stored in global_templates)
    """
    import numpy as _np

    # ---- parameters (with sane defaults) ----
    top_ratio      = float(params.get("dedup_top_ratio", 0.90))      # 90% of top p2p
    lag_window     = int(params.get("dedup_lag_window", 40))         # ±samples for alignment
    cos1_thr       = float(params.get("dedup_cos_top_thr", 0.90))    # top-channel cosine
    p2p_corr_thr   = float(params.get("dedup_p2p_corr_thr", 0.90))   # union p2p Pearson r
    cos_full_thr   = float(params.get("dedup_cos_full_thr", 0.90))   # union EI cosine
    match_tol      = int(params.get("dedup_match_tolerance", 5))     # ±samples for spike matching
    do_plot_merge  = bool(params.get("plot_merge_overlays", True))

    # ---- helpers ----
    def _p2p(ei: _np.ndarray) -> _np.ndarray:
        return (ei.max(axis=1) - ei.min(axis=1)).astype(_np.float32)

    def _top_set(p2p: _np.ndarray, ratio: float) -> _np.ndarray:
        mx = float(p2p.max(initial=0.0))
        if mx <= 0.0:
            return _np.array([], dtype=_np.int32)
        return _np.where(p2p >= ratio * mx)[0].astype(_np.int32)

    def _best_lag_1d(a: _np.ndarray, b: _np.ndarray, win: int) -> int:
        # maximize cosine(a, roll(b, lag)) over lag∈[-win,win]
        if a.size != b.size:
            L = min(a.size, b.size)
            a = a[:L]; b = b[:L]
        best_lag = 0
        best_val = -_np.inf
        denom = (float(_np.linalg.norm(a)) * float(_np.linalg.norm(b))) + 1e-12
        if denom <= 0.0:
            return 0
        for lag in range(-win, win+1):
            if lag == 0:
                v = float(_np.dot(a, b)) / denom
            elif lag > 0:
                v = float(_np.dot(a[lag:], b[:-lag])) / denom
            else:
                L = -lag
                v = float(_np.dot(a[:-L], b[L:])) / denom
            if v > best_val:
                best_val = v; best_lag = lag
        return int(best_lag)

    def _cosine(u: _np.ndarray, v: _np.ndarray) -> float:
        u = u.astype(_np.float32, copy=False).ravel()
        v = v.astype(_np.float32, copy=False).ravel()
        nu = float(_np.linalg.norm(u)); nv = float(_np.linalg.norm(v))
        if nu == 0.0 or nv == 0.0: return -_np.inf
        return float(_np.dot(u, v) / (nu * nv))

    def _pearson(x: _np.ndarray, y: _np.ndarray) -> float:
        x = x.astype(_np.float32, copy=False).ravel()
        y = y.astype(_np.float32, copy=False).ravel()
        if x.size < 2 or y.size < 2: return -_np.inf
        sx = float(x.std()); sy = float(y.std())
        if sx == 0.0 or sy == 0.0: return -_np.inf
        return float((_np.cov(x, y, bias=True)[0,1]) / (sx * sy))

    def _merge_spike_times(times_w: _np.ndarray, times_l: _np.ndarray, lag: int, tol: int) -> _np.ndarray:
        # times_w, times_l are int64 sorted absolute times
        if times_w.size == 0:
            return times_l.astype(_np.int64, copy=False)
        tw = times_w.astype(_np.int64, copy=False)
        tl = (times_l.astype(_np.int64, copy=False) + int(lag))
        i = j = 0
        out = []
        while i < tw.size and j < tl.size:
            dt = int(tl[j] - tw[i])
            if abs(dt) <= tol:
                out.append(int(tw[i])); i += 1; j += 1
            elif dt < -tol:
                out.append(int(tl[j])); j += 1
            else:
                out.append(int(tw[i])); i += 1
        while i < tw.size: out.append(int(tw[i])); i += 1
        while j < tl.size: out.append(int(tl[j])); j += 1
        out = _np.array(sorted(set(out)), dtype=_np.int64)
        return out

    # ---- pull fields from candidate (Y) ----
    Y = candidate_record
    eiY = _np.asarray(Y["ei"])
    p2pY = _p2p(eiY) 
    selY = _np.asarray(Y.get("selected_channels", _np.where(p2pY > 0)[0]), dtype=_np.int32)
    timesY = _np.asarray(Y.get("spike_times", _np.array([], dtype=_np.int64)), dtype=_np.int64)
    topY = int(Y.get("peak_channel", int(p2pY.argmax())))
    topsetY = _top_set(p2pY, top_ratio)

    # ---- shortlist X by simple rule: X.peak_channel ∈ topset(Y) ----
    shortlist = []
    for idx, X in enumerate(global_templates):
        px = int(X.get("peak_channel", int(_np.asarray(X.get("p2p", _p2p(X["ei"]))).argmax())))
        if px in topsetY:
            shortlist.append((idx, X))

    if not shortlist:
        global_templates.append(Y)
        return "kept_new", Y

    # ---- evaluate candidates ----
    best = None
    best_metrics = None
    for idx, X in shortlist:
        eiX = _np.asarray(X["ei"])
        selX = _np.asarray(X.get("selected_channels", _np.where(_p2p(eiX) > 0)[0]), dtype=_np.int32)

        # Pick alignment channel: prefer Y's top channel; else best overlap
        align_ch = topY if topY in selX else None
        if align_ch is None:
            inter = _np.intersect1d(topsetY, selX, assume_unique=False)
            if inter.size:
                align_ch = int(inter[_np.argmax(p2pY[inter])])
            else:
                continue

        # lag by maximizing cosine on single channel within ±lag_window
        lag = _best_lag_1d(eiY[align_ch], eiX[align_ch], lag_window)

        # top-channel cosine after alignment
        cos_top = _cosine(_np.roll(eiY[topY], lag), eiX[topY] if topY in selX else eiX[align_ch])
        if cos_top < cos1_thr:
            continue

        # Union-of-selected P2P correlation (no channel cap)
        U = _np.union1d(selY, selX)
        p2pX = _p2p(eiX)
        r = _pearson(p2pY[U], p2pX[U])
        if r < p2p_corr_thr:
            continue

        # Cosine on concatenated EI over union channels
        eiY_U = _np.roll(eiY[U], lag, axis=1).reshape(U.size, -1)
        eiX_U = eiX[U].reshape(U.size, -1)
        cos_full = _cosine(eiY_U, eiX_U)
        if cos_full < cos_full_thr:
            continue

        key = (cos_top, cos_full)
        if (best is None) or (key > best_metrics):
            best = (idx, X, lag, U, cos_top, r, cos_full)
            best_metrics = key

    if best is None:
        global_templates.append(Y)
        return "kept_new", Y

    # ---- duplicate found → choose winner and merge spikes ----
    idxX, X, lagXY, U, cos_top, rU, cos_full = best
    eiX = np.asarray(X["ei"])       # winner’s EI (correct template for plotting)

    n_old = int(_np.asarray(X.get("spike_times", [])).size)
    n_new = int(timesY.size)
    snr_old = float(X.get("snr", 0.0))
    snr_new = float(Y.get("snr", 0.0))

    new_wins = (n_new > n_old) or (n_new == n_old and snr_new > snr_old)

    # optional overlay for inspection
    if do_plot_merge:
        try:
            import matplotlib.pyplot as _plt
            try:
                import plot_ei_waveforms as pew
            except Exception:
                pew = None
            if pew is not None:
                _plt.figure(figsize=(10,6))
                pew.plot_ei_waveforms([eiX, _np.roll(eiY, lagXY, axis=1)],
                                      params.get("ei_positions", None),
                                      ref_channel=int(Y.get("detect_channel", topY)),
                                      scale=90, box_height=1, box_width=50,
                                      colors=['black','red'])
                _plt.title(f"DEDUP MERGE: cos_top={cos_top:.2f}, r_p2p={rU:.2f}, cos_full={cos_full:.2f}  "
                           f"(lag={lagXY:+d})")
                _plt.show()
        except Exception:
            pass

    def _ensure_fields(rec: Dict[str, Any]) -> None:
        if "merged_from" not in rec: rec["merged_from"] = []
        if "ei_recompute_pending" not in rec: rec["ei_recompute_pending"] = False

    if new_wins:
        # merge unique spikes from X into Y, then replace X in globals
        merged_times = _merge_spike_times(times_w=timesY,
                                          times_l=_np.asarray(X.get("spike_times", []), dtype=_np.int64),
                                          lag=-lagXY, tol=match_tol)
        Y["spike_times"] = merged_times
        _ensure_fields(Y)
        n_added = int(merged_times.size) - int(timesY.size)
        if (int(merged_times.size) <= 100) and (n_added >= 10):
            Y["ei_recompute_pending"] = True
        Y["merged_from"].append({
            "src": int(idxX), "delta": int(lagXY), "n_added": int(n_added),
            "cos_top": float(cos_top), "r_p2p": float(rU), "cos_full": float(cos_full)
        })
        global_templates[idxX] = Y
        return "updated_old", Y
    else:
        # keep X, merge unique spikes from Y into X
        merged_times = _merge_spike_times(times_w=_np.asarray(X.get("spike_times", []), dtype=_np.int64),
                                          times_l=timesY, lag=lagXY, tol=match_tol)
        X["spike_times"] = merged_times
        _ensure_fields(X)
        n_added = int(merged_times.size) - int(n_old)
        if (int(merged_times.size) <= 100) and (n_added >= 10):
            X["ei_recompute_pending"] = True
        X["merged_from"].append({
            "src": "candidate", "delta": int(lagXY), "n_added": int(n_added),
            "cos_top": float(cos_top), "r_p2p": float(rU), "cos_full": float(cos_full)
        })
        return "discarded_new", X



# -----------------------------
# Bookkeeping (tiny, stage-specific)
# -----------------------------

def record_reuse_peel(
    global_templates: List[Dict[str, Any]],
    source_index: int,
    channel_of_interest: int,
    n_peeled: int,
) -> None:
    """Log reuse peel stats inside the source template."""
    if not (0 <= source_index < len(global_templates)):
        return
    rec = global_templates[source_index]
    log = rec.get("detect_channels_log", [])
    log.append((int(channel_of_interest), int(n_peeled)))
    rec["detect_channels_log"] = log


def record_discovery_accept(
    accepted_eis_channel: List[Dict[str, Any]],
    accepted_record: Dict[str, Any],
) -> None:
    """Append accepted record produced in discovery to this channel's list."""
    accepted_eis_channel.append(accepted_record)


def record_aux_accept(
    accepted_eis_channel: List[Dict[str, Any]],
    accepted_record: Dict[str, Any],
) -> None:
    """Append accepted record produced in auxiliary stage to this channel's list."""
    accepted_eis_channel.append(accepted_record)


# -----------------------------
# Optional small helpers
# -----------------------------

def _roll_zero_1d(a: np.ndarray, s: int) -> np.ndarray:
    if s == 0:
        return a
    out = np.zeros_like(a)
    if s > 0:
        out[s:] = a[:-s]
    else:
        out[:s] = a[-s:]
    return out

def _local_peak_offset(w: np.ndarray, t0: int, R: int, prefer_sign: int) -> int:
    lo = max(0, t0 - R); hi = min(w.size - 1, t0 + R)
    seg = w[lo:hi+1]
    j = int(np.argmin(seg)) if prefer_sign < 0 else int(np.argmax(seg))
    return (lo + j) - t0

def compute_channel_jitter_stats(
    snips_cand: np.ndarray,
    ei_final: np.ndarray,
    best_lags_acc: np.ndarray,
    channels: np.ndarray,
    center_sample: int,
    search_radius: int,
    central_band: int,
) -> Dict[str, np.ndarray]:
    """Per-channel timing coherence on accepted spikes.
    Returns dict with keys 'central_frac' and 'edge_frac' arrays over `channels`.
    """
    C, T, Nacc = snips_cand.shape
    chans = np.asarray(channels, int)
    if Nacc == 0 or chans.size == 0:
        return {"central_frac": np.array([], float), "edge_frac": np.array([], float)}

    # EI per-channel peak location and sign
    tpk = np.array([int(np.argmax(np.abs(ei_final[c]))) for c in chans], dtype=int)
    signs = np.sign(ei_final[chans, tpk])
    signs[signs == 0] = -1

    deltas = np.zeros((chans.size, Nacc), dtype=int)
    for s in range(Nacc):
        lag = int(best_lags_acc[s]) if best_lags_acc.size == Nacc else 0
        for k, c in enumerate(chans):
            w = _roll_zero_1d(snips_cand[c, :, s], lag)
            deltas[k, s] = _local_peak_offset(w, int(tpk[k]), search_radius, int(signs[k]))

    central_frac = np.mean(np.abs(deltas) <= central_band, axis=1)
    edge_frac    = np.mean(np.abs(deltas) >= search_radius, axis=1)
    return {"central_frac": central_frac, "edge_frac": edge_frac}

def select_channels_for_scoring(ei: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Return channel indices to use for ΔRMS scoring (p2p>thr, capped at max, floored at min)."""
    p2p = ei.max(axis=1) - ei.min(axis=1)
    thr = float(params.get("p2p_thr", 50.0))
    maxc = int(params.get("max_channels", 80))
    minc = int(params.get("min_channels", 10))
    idx = np.where(p2p > thr)[0]
    if idx.size < minc:
        idx = np.argsort(-p2p)[:minc]
    if idx.size > maxc:
        order = np.argsort(-p2p[idx])
        idx = idx[order[:maxc]]
    return np.asarray(idx, int)


def compute_lag_health(
    best_lag_per_spike: np.ndarray,
    lags: np.ndarray,
    mask: Optional[np.ndarray],
    params: Dict[str, Any],
) -> Dict[str, float]:
    """Compute lag jitter metrics: MAD, central-band Wilson lower bound, edge fraction."""
    if best_lag_per_spike is None or best_lag_per_spike.size == 0:
        return {"mad": np.nan, "central_LB": np.nan, "edge_frac": np.nan}
    b = np.asarray(best_lag_per_spike)
    if mask is not None:
        m = np.asarray(mask, bool)
        if m.size == b.size:
            b = b[m]
    if b.size == 0:
        return {"mad": np.nan, "central_LB": np.nan, "edge_frac": np.nan}

    lags = np.asarray(lags)
    if lags.size:
        lmin, lmax = int(lags.min()), int(lags.max())
    else:
        lmin, lmax = int(b.min()), int(b.max())

    # MAD
    med = float(np.median(b))
    mad = float(np.median(np.abs(b - med)))

    # central band LB using Wilson for p = P(|lag| ≤ central_band)
    central_band = int(params.get("lag_central_band", 1))
    z = float(params.get("lag_wilson_z", 1.96))
    N = b.size
    phat = float(np.mean(np.abs(b) <= central_band))
    denom = 1.0 + (z * z) / N
    center = phat + (z * z) / (2 * N)
    spread = z * np.sqrt((phat * (1.0 - phat) + (z * z) / (4 * N)) / N)
    central_LB = (center - spread) / denom

    # edge fraction: mass at extreme lags
    edge_frac = float(np.mean((b == lmin) | (b == lmax)))

    return {"mad": mad, "central_LB": float(central_LB), "edge_frac": edge_frac}



# --- Post-hoc shard collapsing (consolidate-by-harm, no per-spike LOO) ---
# Place this block near the bottom of discovery_stages.py (helpers + public API function)

from typing import Any, Dict, List, Tuple, Optional
import numpy as np

import axolotl_utils_ram
from collision_utils import median_ei_adaptive, compute_harm_map_noamp

try:
    import matplotlib.pyplot as plt
    try:
        import plot_ei_waveforms as pew
    except Exception:
        pew = None
except Exception:
    plt = None
    pew = None

# -----------------------
# Helpers
# -----------------------

def _neighbors_within_um(ei_positions, dc: int, radius_um: float = 100.0) -> np.ndarray:
    """Return channel IDs within radius_um (µm) of detect channel dc (includes dc)."""
    if ei_positions is None:
        return np.array([dc], dtype=int)
    pos = np.asarray(ei_positions)
    if pos.ndim != 2 or pos.shape[1] < 2:
        return np.array([dc], dtype=int)
    p0 = pos[int(dc), :2]
    d = np.sqrt(((pos[:, :2] - p0) ** 2).sum(axis=1))
    return np.where(d < float(radius_um))[0].astype(int)


def _concat_roll_subset(ei: np.ndarray, chans: np.ndarray, shift: int) -> np.ndarray:
    """Roll selected channels by a single time shift (zero-pad), then flatten."""
    chans = np.asarray(chans, int)
    sub = ei[chans]
    out = np.zeros_like(sub)
    if shift == 0:
        out[:] = sub
    elif shift > 0:
        out[:, shift:] = sub[:, :-shift]
    else:
        out[:, :shift] = sub[:, -shift:]
    return out.reshape(-1).astype(np.float32)


def _concat_cos_sim_multichan(eiA: np.ndarray, eiB: np.ndarray, chans: np.ndarray, max_lag: int) -> float:
    """Cosine similarity between concatenated neighborhoods with a global lag search."""
    vB = _concat_roll_subset(eiB, chans, 0)
    nB = float(np.linalg.norm(vB)) + 1e-12
    best = -1.0
    for s in range(-max_lag, max_lag + 1):
        vA = _concat_roll_subset(eiA, chans, s)
        nA = float(np.linalg.norm(vA)) + 1e-12
        best = max(best, float(np.dot(vA, vB) / (nA * nB)))
    return best


def _form_cohorts_concat(templates: List[Tuple[int, Dict[str, Any]]], detect_channel: int,
                         ei_positions, radius_um: float, sim_thr: float, max_lag: int) -> List[List[int]]:
    """Build cohorts among templates that share detect_channel using cosine over a spatial neighborhood."""
    idxs = [i for i, (_, rec) in enumerate(templates) if int(rec["detect_channel"]) == int(detect_channel)]
    if not idxs:
        return []

    neigh = _neighbors_within_um(ei_positions, int(detect_channel), radius_um)
    m = len(idxs)
    W = np.zeros((m, m), dtype=bool)

    for a in range(m):
        eiA = templates[idxs[a]][1]["ei"]
        for b in range(a, m):
            eiB = templates[idxs[b]][1]["ei"]
            sim = _concat_cos_sim_multichan(eiA, eiB, neigh, max_lag)
            ok = (sim >= sim_thr)
            W[a, b] = W[b, a] = ok

    # connected components
    visited = np.zeros(m, dtype=bool)
    cohorts = []
    for k in range(m):
        if visited[k]:
            continue
        comp = []
        stack = [k]
        visited[k] = True
        while stack:
            u = stack.pop()
            comp.append(idxs[u])
            for v in np.where(W[u])[0]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        cohorts.append(comp)
    return cohorts


def _select_strong_channels_from_final(ei: np.ndarray, p2p_thr: float, minc: int) -> np.ndarray:
    p2p = ei.max(axis=1) - ei.min(axis=1)
    idx = np.where(p2p >= p2p_thr)[0]
    if idx.size < minc:
        order = np.argsort(-p2p)
        idx = order[:minc]
    return idx.astype(int)


def _chan_peak_index(ei_c: np.ndarray) -> int:
    return int(np.argmax(np.abs(ei_c)))


def _local_peak_offset(w: np.ndarray, t0: int, R: int, prefer_sign: int) -> int:
    lo = max(0, t0 - R); hi = min(len(w)-1, t0 + R)
    seg = w[lo:hi+1]
    j = int(np.argmin(seg)) if prefer_sign < 0 else int(np.argmax(seg))
    return (lo + j) - t0


def _compute_jitter_for_template(rec: Dict[str, Any], raw_data: np.ndarray,
                                 snippet_window: Tuple[int,int],
                                 p2p_thr: float, minc: int,
                                 search_radius: int, central_band: int) -> Dict[str, Any]:
    ei   = rec["ei"]
    times = np.asarray(rec["spike_times"], dtype=np.int64)
    if times.size == 0:
        return {"channels": np.array([], int), "central_frac": np.array([], float), "edge_frac": np.array([], float), "n_spikes": 0}

    C = ei.shape[0]
    snips, vtimes = axolotl_utils_ram.extract_snippets_fast_ram(
        raw_data=raw_data, spike_times=times,
        selected_channels=np.arange(C, dtype=int), window=snippet_window,
    )
    if vtimes.size == 0:
        return {"channels": np.array([], int), "central_frac": np.array([], float), "edge_frac": np.array([], float), "n_spikes": 0}

    chans = _select_strong_channels_from_final(ei, p2p_thr, minc)
    tpk  = np.array([_chan_peak_index(ei[c]) for c in chans], dtype=int)
    sign = np.sign(ei[chans, tpk]); sign[sign == 0] = -1

    Nsp = snips.shape[2]
    deltas = np.zeros((chans.size, Nsp), dtype=int)
    for k, c in enumerate(chans):
        t0 = int(tpk[k]); sgn = int(sign[k])
        for s in range(Nsp):
            w = snips[c, :, s]
            deltas[k, s] = _local_peak_offset(w, t0, search_radius, sgn)

    central_frac = np.mean(np.abs(deltas) <= central_band, axis=1)
    edge_frac    = np.mean(np.abs(deltas) >= search_radius, axis=1)
    return {"channels": chans, "central_frac": central_frac, "edge_frac": edge_frac, "n_spikes": int(Nsp)}


def _harm_score_mdw(ei: np.ndarray, snips: np.ndarray) -> Optional[np.ndarray]:
    res = compute_harm_map_noamp(
        ei, snips, p2p_thr=50.0, max_channels=80, min_channels=10,
        lag_radius=3, weight_by_p2p=True, weight_beta=0.7
    )
    mdw = res.get("mean_delta_weighted") if isinstance(res, dict) else None
    if mdw is None:
        return None
    return np.asarray(mdw, float)


# -----------------------
# Public API
# -----------------------

def posthoc_collapse_shards_consolidate(
    accepted_eis: List[Dict[str, Any]],
    raw_data: np.ndarray,
    global_templates: List[Dict[str, Any]],
    *,
    ei_positions=None,
    params: Optional[Dict[str, Any]] = None,
    prune_accepted_eis: bool = True,
) -> Dict[str, Any]:
    """
    Consolidate shard-like templates **within each detect channel cohort** by iteratively
    extracting spike clusters using harm-map mdw against composite EIs.

    Behavior:
      - Single-template cohort: skipped
      - Multi-template cohort, all pass jitter: skipped
      - Otherwise: treat the failing (shard-like) templates as a shard pool.
        Build EI on **all spikes**, score by mdw, accept those ≤ mdw_thr. Remove accepted and repeat
        until leftover < leftover_stop or last accepted < min_assign. Produce 1..K composites.

    Side effects:
      - Updates `global_templates` in place: removes fully-covered shard templates, appends new composites.
      - Optionally updates `accepted_eis` in place the same way (prune + append) so it mirrors globals.
    """
    if params is None:
        params = {}

    # Tunables
    cohort_sim_thr   = float(params.get("cohort_sim_thr", 0.90))
    cohort_max_lag   = int(params.get("cohort_max_lag", 2))
    radius_um        = float(params.get("cohort_radius_um", 100.0))

    snippet_window   = tuple(params.get("snippet_window", (-40, 80)))

    jit_p2p_thr      = float(params.get("jit_p2p_thr", 100.0))
    jit_min_channels = int(params.get("jit_min_channels", 5))
    jit_search_radius= int(params.get("jit_search_radius", 3))
    jit_central_band = int(params.get("jit_central_band", 1))

    # mdw decisioning
    mdw_thr          = float(params.get("assign_mdw_thr", -2.0))
    leftover_stop    = int(params.get("leftover_stop", 20))   # stop when < this many remain
    min_assign       = int(params.get("min_assign", 10))      # stop if < this many accepted in an iter

    show_plots       = bool(params.get("show_plots", False))

    templates = list(enumerate(accepted_eis))
    if not templates:
        return {"status": "empty", "message": "accepted_eis is empty."}

    # map identity to globals index for in-place updates
    id2gidx = {id(gt): i for i, gt in enumerate(global_templates)}

    detect_channels = sorted({int(rec["detect_channel"]) for _, rec in templates})

    summary = {"n_detect_channels": len(detect_channels), "cohorts": [], "new_composites": 0, "shards_removed": 0}

    for dc in detect_channels:
        cohorts = _form_cohorts_concat(templates, dc, ei_positions, radius_um, cohort_sim_thr, cohort_max_lag)
        if not cohorts:
            continue

        for ci, comp in enumerate(cohorts, 1):
            cohort_ids  = [templates[i][0] for i in comp]
            cohort_recs = [templates[i][1] for i in comp]
            print(f"[Cohort] DC {dc} | cohort {ci} | template IDs: {cohort_ids}")
            snr_stopped = False

            if len(cohort_recs) == 1:
                summary["cohorts"].append({
                    "detect_channel": dc, "cohort_id": ci, "action": "single_skip",
                    "template_ids": [int(x) for x in cohort_ids],
                })

                continue

            # Jitter per template
            jit_list = [
                _compute_jitter_for_template(
                    rec, raw_data,
                    snippet_window=snippet_window,
                    p2p_thr=jit_p2p_thr, minc=jit_min_channels,
                    search_radius=jit_search_radius, central_band=jit_central_band,
                ) for rec in cohort_recs
            ]

            # Shard-like flags per your original thresholds
            edge_any_max    = float(params.get("edge_any_max", 0.30))
            edge_many_max   = float(params.get("edge_many_max", 0.15))
            edge_many_count = int(params.get("edge_many_count", 3))
            cent_any_min    = float(params.get("cent_any_min", 0.50))
            cent_many_min   = float(params.get("cent_many_min", 0.70))
            cent_many_count = int(params.get("cent_many_count", 3))

            def _is_shard_like(jres: Dict[str, Any]) -> bool:
                cf = jres.get("central_frac", np.array([], float))
                ef = jres.get("edge_frac",    np.array([], float))
                if cf.size == 0:
                    return True
                flags = (
                    np.any(ef > edge_any_max) or
                    (np.sum(ef > edge_many_max) >= edge_many_count) or
                    np.any(cf < cent_any_min) or
                    (np.sum(cf < cent_many_min) >= cent_many_count)
                )
                return bool(flags)

            shard_mask = np.array([_is_shard_like(j) for j in jit_list], dtype=bool)
            shard_ids = [int(tid) for tid, m in zip(cohort_ids, shard_mask) if m]
            kept_ids  = [int(tid) for tid, m in zip(cohort_ids, shard_mask) if not m]
            print(f"         shard-like: {shard_ids} | pass-jitter: {kept_ids}")

            if not shard_mask.any():
                summary["cohorts"].append({
                    "detect_channel": dc, "cohort_id": ci, "action": "all_pass_jitter",
                    "template_ids": [int(x) for x in cohort_ids],
                    "shard_ids": [], "kept_ids": [int(x) for x in cohort_ids],
                })

                continue

            # Build shard pool = union of spikes from shard-like templates
            shard_recs = [rec for rec, m in zip(cohort_recs, shard_mask) if m]
            shard_ids  = [tid for tid, m in zip(cohort_ids, shard_mask) if m]

            pool_times = np.unique(np.concatenate([np.asarray(r["spike_times"], np.int64) for r in shard_recs]))
            if pool_times.size == 0:
                summary["cohorts"].append({
                    "detect_channel": dc, "cohort_id": ci, "action": "empty_pool",
                    "template_ids": [int(x) for x in cohort_ids],
                    "shard_ids": shard_ids, "kept_ids": kept_ids,
                })

                continue

            # Extract snippets for the shard pool once (use channels count from first rec)
            C = shard_recs[0]["ei"].shape[0]
            snips_pool, times_pool = axolotl_utils_ram.extract_snippets_fast_ram(
                raw_data=raw_data, spike_times=pool_times,
                selected_channels=np.arange(C, dtype=int), window=snippet_window,
            )
            if times_pool.size == 0:
                summary["cohorts"].append({
                    "detect_channel": dc, "cohort_id": ci, "action": "no_valid_snips",
                    "template_ids": [int(x) for x in cohort_ids],
                    "shard_ids": shard_ids, "kept_ids": kept_ids,
                })

                continue

            # Iteratively extract composites until stop
            clusters: List[np.ndarray] = []
            snips_cur = snips_pool
            times_cur = times_pool
            iter_idx = 0
            while True:
                iter_idx += 1
                # Composite EI from current pool
                ei_comp = median_ei_adaptive(snips_cur)
                mdw = _harm_score_mdw(ei_comp, snips_cur)  # vector over current pool
                if mdw is None or mdw.size == 0:
                    break
                accept = (mdw <= mdw_thr)
                n_acc = int(np.sum(accept))
                n_left = int(snips_cur.shape[2] - n_acc)
                if n_acc < min_assign:
                    break
                clusters.append(times_cur[accept])
                # peel accepted and continue if enough remain
                if n_left < leftover_stop:
                    break
                keep = ~accept
                snips_cur = snips_cur[:, :, keep]
                times_cur = times_cur[keep]

            if not clusters:
                summary["cohorts"].append({
                    "detect_channel": dc, "cohort_id": ci, "action": "no_cluster",
                    "template_ids": [int(x) for x in cohort_ids],
                    "shard_ids": shard_ids, "kept_ids": kept_ids,
                })

                continue

            # Build new composite template(s), update globals and accepted_eis
            new_templates = []
            for cl_times in clusters:
                snips_final, times_final = axolotl_utils_ram.extract_snippets_fast_ram(
                    raw_data=raw_data, spike_times=np.asarray(cl_times, np.int64),
                    selected_channels=np.arange(C, dtype=int), window=snippet_window,
                )
                if times_final.size == 0:
                    continue
                ei_final = median_ei_adaptive(snips_final)

                # SNR gate: if too low, do not add; stop producing further composites in this cohort
                min_snr = float(params.get("min_snr", 8.0))
                p2p = (ei_final.max(axis=1) - ei_final.min(axis=1)).astype(float)
                snr = float(p2p.max() / max(np.median(p2p), 1e-9))
                if snr < min_snr:
                    snr_stopped = True  # flag; we’ll reflect it in the final cohort summary
                    break

                # minimal metadata
                rec_new: Dict[str, Any] = {
                    "ei": ei_final,
                    "spike_times": times_final,
                    "detect_channel": int(dc),
                    "p2p": p2p,
                    "snr": snr,
                    "selected_channels": np.where(p2p >= 0.7 * p2p.max())[0].astype(int),
                    "origin": "posthoc_shard_composite",
                }
                new_templates.append(rec_new)

            # Unconditional removal of all shard participants (ignore any unassigned/orphan spikes)
            removed = 0
            for rec_sh in shard_recs:
                # remove from globals
                gidx = id2gidx.get(id(rec_sh), None)
                if gidx is not None and 0 <= gidx < len(global_templates):
                    global_templates.pop(gidx)
                    id2gidx = {id(gt): i for i, gt in enumerate(global_templates)}
                # remove from accepted_eis
                if prune_accepted_eis:
                    for k in range(len(accepted_eis)):
                        if accepted_eis[k] is rec_sh:
                            accepted_eis.pop(k)
                            break
                removed += 1

            # Append new composites
            for rec_new in new_templates:
                global_templates.append(rec_new)
                if prune_accepted_eis:
                    accepted_eis.append(rec_new)
            summary["new_composites"] += len(new_templates)
            summary["shards_removed"] += removed
            summary["cohorts"].append({
                "detect_channel": dc, "cohort_id": ci, "action": "consolidated",
                "removed": int(removed), "new": int(len(new_templates)),
                "snr_stop": bool(snr_stopped),
                "template_ids": [int(x) for x in cohort_ids],
                "shard_ids": shard_ids, "kept_ids": kept_ids,
            })

            # Optional plotting of resulting composites
            if show_plots and pew is not None and len(new_templates) > 0 and plt is not None:
                cols = 2; rows = (len(new_templates) + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(12, 6*rows))
                axes = np.atleast_1d(axes).ravel()
                for ax, recn in zip(axes, new_templates):
                    pew.plot_ei_waveforms(
                        recn["ei"], ei_positions,
                        ref_channel=int(dc), scale=90,
                        box_height=1, box_width=50, colors='black', ax=ax
                    )
                    ax.set_title(f"DC {dc} cohort {ci} — new composite (n={len(recn['spike_times'])})")
                for ax in axes[len(new_templates):]:
                    ax.axis('off')
                plt.tight_layout(); plt.show()

    summary["status"] = "ok"
    return summary
