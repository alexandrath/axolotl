# === Global templates: dedup + merge (periodic maintenance) ===
import numpy as np
import matplotlib.pyplot as plt

try:
    import plot_ei_waveforms as pew
except Exception:
    pew = None

from collision_utils import select_template_channels, median_ei_adaptive
import axolotl_utils_ram  # needed only if EI recompute is triggered

# ------------------
# helpers (local)
# ------------------
def _p2p(ei): return ei.max(axis=1) - ei.min(axis=1)

def _alive_idx(globals_list, obj):
    """Return current index of obj in globals_list by identity, or -1 if not present."""
    for k, g in enumerate(globals_list):
        if g is obj:
            return k
    return -1

def _in_globals(globals_list, obj):
    return _alive_idx(globals_list, obj) != -1


def _roll0_1d(a, s):
    if s == 0: return a
    out = np.zeros_like(a)
    if s > 0: out[s:] = a[:-s]
    else:     out[:s] = a[-s:]
    return out

def _cos1d(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def _best_cosine_on_channel(eiX, eiY, ch, max_lag):
    """Return (best_cos, best_shift) aligning Y to X on single channel ch."""
    x = eiX[ch].astype(np.float32)
    best = -1.0; best_s = 0
    for s in range(-max_lag, max_lag+1):
        y = _roll0_1d(eiY[ch], s).astype(np.float32)
        c = _cos1d(x, y)
        if c > best:
            best = c; best_s = s
    return best, best_s

def _concat_subset(ei, chans, shift=0):
    sub = ei[chans]
    if shift != 0:
        sub = np.vstack([_roll0_1d(sub[k], shift) for k in range(sub.shape[0])])
    return sub.reshape(-1).astype(np.float32)

def _pearson(x, y):
    x = x.astype(np.float64); y = y.astype(np.float64)
    if x.size < 2 or y.size < 2: return 0.0
    vx = x - x.mean(); vy = y - y.mean()
    den = np.linalg.norm(vx) * np.linalg.norm(vy)
    return 0.0 if den == 0 else float(np.dot(vx, vy) / den)

def _selected_channels(rec, params):
    if "selected_channels" in rec and rec["selected_channels"] is not None and len(rec["selected_channels"]) > 0:
        return np.asarray(rec["selected_channels"], dtype=int)
    p2p_thr = float(params.get("p2p_thr", 50.0))
    max_channels = int(params.get("max_channels", 80))
    min_channels = int(params.get("min_channels", 10))
    chans, _ = select_template_channels(rec["ei"], p2p_thr=p2p_thr, max_n=max_channels, min_n=min_channels)
    return chans

def _unique_new_times(times_w, times_l_shifted, tol=5):
    """
    Return mask over loser times: True if NOT matched to any winner time within ±tol.
    Both inputs are 1D int64 arrays (sorted or not); O(n log n).
    """
    tw = np.sort(times_w.astype(np.int64))
    tl = np.sort(times_l_shifted.astype(np.int64))
    j = 0; m = tw.size
    new_mask = np.ones(tl.size, dtype=bool)
    for i in range(tl.size):
        t = tl[i]
        # advance j until tw[j] >= t - tol
        while j < m and tw[j] < t - tol:
            j += 1
        # check match window
        k = j
        matched = False
        while k < m and tw[k] <= t + tol:
            if abs(int(tw[k]) - int(t)) <= tol:
                matched = True
                break
            k += 1
        new_mask[i] = not matched
    return new_mask

# ------------------
# main API (cell)
# ------------------
def dedup_merge_globals(
    global_templates,
    positions,
    params,
    *,
    raw_data=None,                 # needed only if EI recompute is triggered
    accepted_eis=None,             # optional: keep in sync
    show_plots=True,
    plot_max=12,
    verbose=True
):
    """
    Periodic global deduplication & merge.
    - Shortlist by top-channel neighborhood (≥90% of top p2p).
    - Align on top channel (±max_lag), require cos_top >= thr.
    - Union-of-selected p2p Pearson r >= thr.
    - Full cosine on concatenated union >= thr.
    - If duplicate: winner = more spikes (tie-break by SNR). Merge loser's unique spikes
      (after shifting by found lag; ±5-sample jitter for matching). If winner_n<=100 and added>10,
      optionally recompute EI from raw_data.
    - Remove loser from globals (and accepted_eis if provided). Plot overlay for audit.

    Returns a summary dict.
    """
    if not global_templates:
        print("GLOBAL_TEMPLATES is empty.")
        return {"n_pairs": 0, "n_merges": 0}

    # thresholds (same style as CTE)
    max_lag       = int(params.get("ei_sim_max_lag", 40))
    sim_top_thr   = float(params.get("ei_sim_top_thr", 0.90))
    p2p_corr_thr  = float(params.get("ei_p2p_corr_thr", 0.90))
    sim_full_thr  = float(params.get("ei_sim_full_thr", 0.90))
    top_neigh_pct = float(params.get("ei_top_neigh_pct", 0.90))
    jitter_tol    = int(params.get("merge_jitter_tol", 5))
    recalc_min_add= int(params.get("merge_recalc_min_add", 10))
    recalc_n_max  = int(params.get("merge_recalc_winner_n_max", 100))
    snr_floor     = float(params.get("min_snr", 0.0))  # not gating merges, only for metadata

    # build infos snapshot (indices must be stable across mutations, so we refresh per pass)
    def _build_infos():
        infos = []
        for idx, rec in enumerate(global_templates):
            ei = rec["ei"]; p2p = _p2p(ei)
            peak_ch = int(np.argmax(p2p)) if p2p.size else 0
            thr = top_neigh_pct * float(p2p[peak_ch]) if p2p.size else np.inf
            neigh = np.where(p2p >= thr)[0]
            infos.append({
                "idx": idx,
                "rec": rec,
                "ei": ei,
                "p2p": p2p,
                "peak_ch": peak_ch,
                "neigh": neigh,
                "sel": _selected_channels(rec, params),
                "dc": int(rec.get("detect_channel", peak_ch)),
                "n_spikes": int(np.asarray(rec.get("spike_times", [])).size),
                "snr": float(rec.get("snr", 0.0)),
            })
        return infos

    # iterative passes until no merges
    n_merges_total = 0
    plots_done = 0
    pass_idx = 0

    while True:
        pass_idx += 1
        infos = _build_infos()
        K = len(infos)
        merged_any = False

        if verbose:
            print(f"\n[global-dedup] pass {pass_idx}: scanning {K} templates")

        # build candidate pairs (shortlist)
        pairs = []
        for i in range(K):
            Xi = infos[i]
            cand_js = [j for j in range(i+1, K) if infos[j]["peak_ch"] in Xi["neigh"]]
            for j in cand_js:
                Xj = infos[j]

                # align on Xi's top channel
                cos_top, lag = _best_cosine_on_channel(Xi["ei"], Xj["ei"], Xi["peak_ch"], max_lag)
                if cos_top < sim_top_thr:
                    continue

                # p2p pattern corr on union of selected
                U = np.union1d(Xi["sel"], Xj["sel"])
                r = _pearson(_p2p(Xi["ei"])[U], _p2p(Xj["ei"])[U])
                if r < p2p_corr_thr:
                    continue

                # full cosine on concatenated union
                vX = _concat_subset(Xi["ei"], U, shift=0)
                vY = _concat_subset(Xj["ei"], U, shift=lag)
                cos_full = _cos1d(vX, vY)
                if cos_full < sim_full_thr:
                    continue

                pairs.append({
                    "base": infos[i]["rec"],     # X (reference) object
                    "cand": infos[j]["rec"],     # Y (candidate) object
                    "lag_B_to_A": int(lag),      # lag that aligns cand→base on base's top channel
                    "cos_top": float(cos_top),
                    "p2p_r": float(r),
                    "cos_full": float(cos_full),
                    "union_size": int(U.size),
                })

        if not pairs:
            if verbose:
                print("[global-dedup] no candidate duplicates under current thresholds.")
            break

        # sort by strength (highest cos_full first)
        pairs.sort(key=lambda d: d["cos_full"], reverse=True)

        # track removals by object identity
        removed_ids = set()

        for P in pairs:
            # P should be a dict with object identities
            # expected keys: base, cand, lag_B_to_A, cos_top, p2p_r, cos_full
            A = P["base"]   # reference template object (dict)
            B = P["cand"]   # candidate template object (dict)

            # skip if already removed or not in globals anymore
            if id(A) in removed_ids or id(B) in removed_ids:
                continue
            if not (_in_globals(global_templates, A) and _in_globals(global_templates, B)):
                continue

            # current indices for logging only
            ia = _alive_idx(global_templates, A)
            ib = _alive_idx(global_templates, B)
            if ia < 0 or ib < 0:
                continue

            # pull current fields directly from the live objects
            eiA, eiB = A["ei"], B["ei"]
            dcA = int(A.get("detect_channel", np.argmax(_p2p(eiA))))
            dcB = int(B.get("detect_channel", np.argmax(_p2p(eiB))))
            lag = int(P["lag_B_to_A"])         # aligns B -> A on A's top channel
            cos_top  = float(P["cos_top"])
            r_p2p    = float(P["p2p_r"])
            cos_full = float(P["cos_full"])

            # ---- winner/loser by current spike counts (tie by SNR) ----
            nA   = int(np.asarray(A.get("spike_times", [])).size)
            nB   = int(np.asarray(B.get("spike_times", [])).size)
            snrA = float(A.get("snr", 0.0))
            snrB = float(B.get("snr", 0.0))

            if nA > nB or (nA == nB and snrA >= snrB):
                winner, loser = A, B
                lag_wrt_winner =  lag       # we computed B->A; A is winner
                pair_idx = (ia, ib)
                dcW = dcA
                eiW, eiL = eiA, eiB
            else:
                winner, loser = B, A
                lag_wrt_winner = -lag       # flip if B becomes winner
                pair_idx = (ib, ia)
                dcW = dcB
                eiW, eiL = eiB, eiA

            # ---- unique spike union (± jitter_tol) on absolute times ----
            jitter_tol = int(params.get("merge_jitter_tol", 5))
            t_w = np.asarray(winner["spike_times"], dtype=np.int64)
            t_l = np.asarray(loser["spike_times"],  dtype=np.int64)
            if t_l.size == 0:
                continue
            t_l_shift = t_l + lag_wrt_winner
            new_mask  = _unique_new_times(t_w, t_l_shift, tol=jitter_tol)
            new_times = t_l_shift[new_mask]
            n_add     = int(new_times.size)

            # ---- audit plot (robust to missing positions) ----
            if show_plots and plots_done < plot_max:
                eiY_shift = np.vstack([_roll0_1d(eiL[c], lag_wrt_winner) for c in range(eiL.shape[0])])
                plt.figure(figsize=(10, 6))
                if pew is not None and positions is not None:
                    pew.plot_ei_waveforms([eiW, eiY_shift], positions,
                                        ref_channel=dcW, scale=90, box_height=1, box_width=50,
                                        colors=['black', 'tab:red'])
                else:
                    ch = int(dcW)
                    plt.plot(eiW[ch], 'k', lw=1.8, alpha=0.95, label=f'win {pair_idx[0]}')
                    plt.plot(eiY_shift[ch], color='tab:red', lw=1.5, alpha=0.8, label=f'lose {pair_idx[1]} (shifted)')
                    plt.legend(loc='best'); plt.grid(True, alpha=0.3)
                plt.title(f"[MERGE]{' pure-dup remove' if n_add==0 else f' +'+str(n_add)} | "
                        f"win={pair_idx[0]} lose={pair_idx[1]} | "
                        f"lag={lag_wrt_winner:+d} cos_top={cos_top:.2f} r={r_p2p:.2f} cos_full={cos_full:.2f}")
                plt.tight_layout(); plt.show()
                plots_done += 1

            # ---- merge (if any additions) ----
            if n_add > 0:
                merged_times = np.unique(np.concatenate([t_w, new_times]))
                winner["spike_times"] = merged_times

                # optional EI recompute for small winners that gained many spikes
                if (raw_data is not None and
                    nA <= int(params.get("merge_recalc_winner_n_max", 100)) and
                    n_add >= int(params.get("merge_recalc_min_add", 10))):
                    C = eiW.shape[0]
                    snips, valid = axolotl_utils_ram.extract_snippets_fast_ram(
                        raw_data=raw_data, spike_times=merged_times,
                        selected_channels=np.arange(C, dtype=int),
                        window=tuple(params.get("snippet_window", (-40, 80))),
                    )
                    if valid.size > 0:
                        from collision_utils import median_ei_adaptive
                        ei_new = median_ei_adaptive(snips)
                        winner["ei"]  = ei_new
                        p2p_new = _p2p(ei_new)
                        winner["p2p"] = p2p_new
                        winner["snr"] = float(p2p_new.max() / max(np.median(p2p_new), 1e-9))
                        winner["selected_channels"] = _selected_channels(winner, params)

            # ---- remove loser by identity (globals + accepted_eis) ----
            idxL = _alive_idx(global_templates, loser)
            if idxL != -1:
                global_templates.pop(idxL)
            if accepted_eis is not None:
                for k, aobj in enumerate(accepted_eis):
                    if aobj is loser:
                        accepted_eis.pop(k)
                        break
            removed_ids.add(id(loser))

            n_merges_total += 1
            merged_any = True


        if not merged_any:
            if verbose:
                print("[global-dedup] no merges executed in this pass.")
            break

    if verbose:
        print(f"[global-dedup] total merges: {n_merges_total}")
    return {"n_pairs": len(global_templates), "n_merges": n_merges_total}

