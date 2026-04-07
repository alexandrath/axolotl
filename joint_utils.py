import numpy as np
from collision_utils import (
    select_template_channels,
    main_channel_and_neg_peak,
    compute_harm_map_noamp,
    plot_harm_heatmap,
)

import matplotlib.pyplot as plt
import plot_ei_waveforms as pew
import os, pickle, tempfile


def recenter_ei_to_ref_trough(ei: np.ndarray, center_index: int = 40) -> np.ndarray:
    """Zero-padded shift so that the most negative trough on the strongest channel sits at center_index."""
    C, T = ei.shape
    ref_ch = int(np.argmin(ei.min(axis=1)))
    trough_idx = int(np.argmin(ei[ref_ch]))
    shift = center_index - trough_idx
    if shift == 0:
        return ei.copy()
    out = np.zeros_like(ei)
    if shift > 0:
        out[:, shift:] = ei[:, :T-shift]
    else:
        s = -shift
        out[:, :T-s] = ei[:, s:]
    return out

def compute_ei_from_indices(raw_data: np.ndarray, indices: np.ndarray, T: int, center_index: int = 40) -> np.ndarray:
    """Average snippets centered at each index (absolute raw samples) into an EI."""
    C, L = raw_data.shape
    starts = indices.astype(np.int64) - center_index
    # sanity filter (safety against edges)
    valid = (starts >= 0) & (starts + T <= L)
    starts = starts[valid]
    if starts.size == 0:
        return np.zeros((C, T), dtype=np.float32)
    # stack and mean
    stack = np.stack([raw_data[:, s:s+T] for s in starts], axis=0)  # [N, C, T]
    return stack.mean(axis=0).astype(np.float32)

# ===== EI drift assessment (per-channel + overall) =====
def assess_ei_drift(ei_orig_centered, ei_best_centered, rms_thr=5.0, channel_ids=None, title_prefix=""):
    import numpy as np
    import matplotlib.pyplot as plt

    E1 = np.asarray(ei_orig_centered, dtype=np.float32)  # [C, T]
    E2 = np.asarray(ei_best_centered, dtype=np.float32)   # [C, T]

    # Align lengths defensively
    C1, T1 = E1.shape; C2, T2 = E2.shape
    C = min(C1, C2); T = min(T1, T2)
    if (C1 != C2) or (T1 != T2):
        print(f"[assess_ei_drift] shape mismatch (orig {E1.shape}, best {E2.shape}) → using common [C={C}, T={T}]")
        E1 = E1[:C, :T]; E2 = E2[:C, :T]

    if channel_ids is None:
        channel_ids = np.arange(C, dtype=int)
    else:
        channel_ids = np.asarray(channel_ids, dtype=int)
        if channel_ids.size != C:
            # If provided ids don't match C, fall back to 0..C-1
            channel_ids = np.arange(C, dtype=int)

    # Per-channel RMS and pass mask (union)
    rms1 = np.sqrt(np.mean(E1**2, axis=1))
    rms2 = np.sqrt(np.mean(E2**2, axis=1))
    use = (rms1 > rms_thr) | (rms2 > rms_thr)
    idx = np.flatnonzero(use)
    if idx.size == 0:
        print(f"[assess_ei_drift] No channels exceed RMS>{rms_thr}. Nothing to compare.")
        return {
            "idx": idx, "cosine_per_ch": np.array([]), "amp_ratio_per_ch": np.array([]),
            "cosine_concat": np.nan
        }

    # Safe cosine
    def _cos(a, b):
        na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
        return float(np.dot(a, b) / (na * nb)) if (na > 0 and nb > 0) else np.nan

    # Per-channel cosine and amplitude ratio (best/orig)
    cos_ch = np.zeros(idx.size, dtype=np.float32)
    rat_ch = np.zeros(idx.size, dtype=np.float32)

    eps = 1e-9
    for k, c in enumerate(idx):
        v1 = E1[c]; v2 = E2[c]
        cos_ch[k] = _cos(v1, v2)
        a1 = float(np.max(np.abs(v1)))
        a2 = float(np.max(np.abs(v2)))
        rat_ch[k] = a2 / (a1 + eps)

    # Overall cosine on concatenated waveforms (union of channels)
    v1_cat = E1[idx].reshape(-1)
    v2_cat = E2[idx].reshape(-1)
    cos_all = _cos(v1_cat, v2_cat)


        # --- Concatenated cosine EXCLUDING main channel(s) ---
    # main channel = largest trough (most negative) per EI
    trough1 = E1.min(axis=1)  # [C]
    trough2 = E2.min(axis=1)  # [C]
    ch_main1 = int(np.argmin(trough1))
    ch_main2 = int(np.argmin(trough2))

    # Exclude both if they differ; otherwise exclude the shared one
    excl_set = {ch_main1} if ch_main1 == ch_main2 else {ch_main1, ch_main2}
    # keep list of channels (IDs) we excluded for reporting
    excluded_ids = np.asarray(channel_ids)[list(excl_set)]

    # Build union minus excluded mains
    idx_excl = np.array([c for c in idx if c not in excl_set], dtype=int)
    if idx_excl.size > 0:
        v1_cat_ex = E1[idx_excl].reshape(-1)
        v2_cat_ex = E2[idx_excl].reshape(-1)
        cos_all_excl_main = _cos(v1_cat_ex, v2_cat_ex)
    else:
        cos_all_excl_main = np.nan


    # ---- Plots (scatter: x = RMS, y = metric) ----
    # Use the same union logic as selection: RMS = max(rms1, rms2)
    x_rms = np.maximum(rms1, rms2)[idx]

    # 1) Cosine vs RMS
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(x_rms, cos_ch, s=14, alpha=0.85)
    ax.set_xlabel("Channel RMS (max of orig/best)")
    ax.set_ylabel("Cosine (orig vs best)")
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{title_prefix} cosine vs RMS (RMS>{rms_thr}, n={idx.size})")

    # 2) Amplitude ratio vs RMS
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(x_rms, rat_ch, s=14, alpha=0.85)
    ax.axhline(1.0, linestyle='--', linewidth=1.0)
    ax.set_xlabel("Channel RMS (max of orig/best)")
    ax.set_ylabel("Amp ratio (best/orig)")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{title_prefix} |max| ratio vs RMS (RMS>{rms_thr})")

    print(
        "[assess_ei_drift] "
        f"concat cosine (union, n={idx.size}): {cos_all:.4f} | "
        f"concat cosine excl main ch(s) {excluded_ids.tolist()} "
        f"(n={idx_excl.size}): {cos_all_excl_main:.4f}"
    )



    return {
        "idx": idx,
        "channels": channel_ids[idx],
        "cosine_per_ch": cos_ch,
        "amp_ratio_per_ch": rat_ch,
        "cosine_concat": cos_all
    }


# ---- Bimodality helpers (1D k-means + d') ----
def _kmeans_1d_dprime(x, n_init=4, max_iter=50):
    x = np.asarray(x, dtype=np.float32).reshape(-1, 1)
    q25, q75 = float(np.quantile(x, 0.25)), float(np.quantile(x, 0.75))
    best = None
    best_inertia = np.inf
    for _ in range(n_init):
        c1, c2 = q25, q75
        for _ in range(max_iter):
            d1 = np.abs(x - c1); d2 = np.abs(x - c2)
            labels = (d2 < d1).astype(np.int32)
            n1 = int((labels == 0).sum()); n2 = int((labels == 1).sum())
            if n1 == 0 or n2 == 0:
                # nudge centers to avoid collapse
                eps = 1e-3
                c1, c2 = q25 - eps, q75 + eps
                continue
            c1_new = float(x[labels == 0].mean()); c2_new = float(x[labels == 1].mean())
            if abs(c1_new - c1) + abs(c2_new - c2) < 1e-6:
                break
            c1, c2 = c1_new, c2_new
        inertia = float(((x[labels == 0] - c1) ** 2).sum() + ((x[labels == 1] - c2) ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best = (labels.copy(), c1, c2)

    labels, m1, m2 = best
    n1 = int((labels == 0).sum()); n2 = int((labels == 1).sum())
    s1 = float(np.std(x[labels == 0], ddof=1)) if n1 > 1 else np.nan
    s2 = float(np.std(x[labels == 1], ddof=1)) if n2 > 1 else np.nan
    if not (np.isfinite(s1) and np.isfinite(s2)):
        dprime = np.nan
    else:
        dprime = abs(m1 - m2) / np.sqrt(0.5 * (s1 ** 2 + s2 ** 2))

    # order by mean (low, high)
    if m1 <= m2:
        idx_lo = np.flatnonzero(labels == 0).astype(int)
        idx_hi = np.flatnonzero(labels == 1).astype(int)
        means, stds = (m1, m2), (s1, s2)
    else:
        idx_lo = np.flatnonzero(labels == 1).astype(int)
        idx_hi = np.flatnonzero(labels == 0).astype(int)
        means, stds = (m2, m1), (s2, s1)
    return float(dprime), idx_lo, idx_hi, means, stds


def check_bimodality_and_plot(snips_cube, res_obj, ei_pos, ref_ch,
                               dprime_thr=5.0, min_per_cluster=5):
    """
    After a harm-map prune round, scan selected channels for amplitude bimodality.
    If found (d' > dprime_thr), overlay two EIs (low vs high amplitude cohorts)
    and RETURN a payload with indices/EIs. Otherwise return None.

    Returns (dict or None):
      {
        'hit': True,
        'chan': <int>,
        't0': <int>,
        'dprime': <float>,
        'idx_lo': np.ndarray[int],   # indices into CURRENT snips_cube
        'idx_hi': np.ndarray[int],   # indices into CURRENT snips_cube
        'ei_lo': np.ndarray[C,T],
        'ei_hi': np.ndarray[C,T],
      }
    """
    # EI built from the *current* kept spikes
    ei_now = snips_cube.mean(axis=2).astype(np.float32)        # [C, L]
    sel = np.asarray(res_obj["selected_channels"], dtype=int)
    if sel.size == 0:
        return None

    # trough per selected channel and sort by trough depth (most negative first)
    t_neg = np.argmin(ei_now[sel], axis=1)                     # [K_sel]
    trough_vals = ei_now[sel, t_neg]                           # negative numbers
    order = np.argsort(trough_vals)                            # ascending => most negative first

    for jj in order:
        c = int(sel[jj])
        t0 = int(t_neg[jj])
        # robust trough per spike: min over ±1 samples around t0
        lo = max(0, t0 - 1); hi = min(ei_now.shape[1] - 1, t0 + 1)
        amps = -snips_cube[c, lo:hi + 1, :].min(axis=0).astype(np.float32)

        if amps.size < (2 * min_per_cluster):
            continue

        dprime, idx_lo, idx_hi, _, _ = _kmeans_1d_dprime(amps)

        # --- sanitize cohort indices against CURRENT snippet count ---
        Ncur = snips_cube.shape[2]
        idx_lo = np.asarray(idx_lo, dtype=np.int64).ravel()
        idx_hi = np.asarray(idx_hi, dtype=np.int64).ravel()
        valid_lo = (idx_lo >= 0) & (idx_lo < Ncur)
        valid_hi = (idx_hi >= 0) & (idx_hi < Ncur)
        if (not valid_lo.all()) or (not valid_hi.all()):
            nbad = int((~valid_lo).sum() + (~valid_hi).sum())
            print(f"[bimodality] sanitized {nbad} out-of-range indices (N={Ncur}).")
        idx_lo = idx_lo[valid_lo]
        idx_hi = idx_hi[valid_hi]

        if (np.isfinite(dprime) and dprime > dprime_thr and
            idx_lo.size >= min_per_cluster and idx_hi.size >= min_per_cluster):

            # Two EIs across the full array
            ei_lo = snips_cube[:, :, idx_lo].mean(axis=2).astype(np.float32)
            ei_hi = snips_cube[:, :, idx_hi].mean(axis=2).astype(np.float32)


            # # Plot overlay (no figsize in API; pass ax)
            # try:
            #     fig, ax = plt.subplots(figsize=(20, 12))
            #     pew.plot_ei_waveforms(
            #         [ei_lo, ei_hi], ei_pos,
            #         ref_channel=ref_ch,
            #         scale=70.0, box_height=1.0, box_width=50.0,
            #         colors=['black', 'crimson'], linewidth=[0.6, 1.2], alpha=[0.6, 0.95],
            #         ax=ax
            #     )
            #     ax.set_title(f"Bimodality on ch {c}: d'={dprime:.2f} | n_low={idx_lo.size}, n_high={idx_hi.size}")
            #     plt.show()
            # except Exception as e:
            #     print(f"[bimodality] plotting skipped: {e}")

            print(f"[bimodality] ch {c}: d'={dprime:.2f} | n_low={idx_lo.size}, n_high={idx_hi.size}")
            return {
                "hit": True, "chan": c, "t0": t0, "dprime": float(dprime),
                "idx_lo": idx_lo, "idx_hi": idx_hi,
                "ei_lo": ei_lo, "ei_hi": ei_hi
            }

    return None




# ===== K-means (k=2) diagnostics per channel with PCA scatter =====

def _kmeans2_mv(X, n_init=8, max_iter=60, random_state=None):
    """
    Simple Lloyd k-means for k=2 on multivariate data X [N,L].
    Returns labels (0/1), centers [2,L], inertia.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=np.float32)
    N, L = X.shape
    if N < 2:
        return np.zeros(N, dtype=np.int32), np.vstack([X.mean(axis=0), X.mean(axis=0)]), 0.0

    best = None
    best_inertia = np.inf
    for _ in range(n_init):
        # pick two distinct seeds
        if N == 2:
            seed_idx = np.array([0, 1])
        else:
            seed_idx = rng.choice(N, size=2, replace=False)
        C = X[seed_idx].astype(np.float32)  # [2,L]

        for _ in range(max_iter):
            # assign
            # distances^2 to centers
            # D[i, k] = ||X[i]-C[k]||^2
            D0 = np.sum((X - C[0])**2, axis=1)
            D1 = np.sum((X - C[1])**2, axis=1)
            labels = (D1 < D0).astype(np.int32)

            # empty-cluster guard: if any empty, re-seed a random point there
            n0 = int((labels == 0).sum()); n1 = int((labels == 1).sum())
            if n0 == 0 or n1 == 0:
                j = 0 if n0 == 0 else 1
                C[j] = X[rng.integers(0, N)]
                continue

            C0 = X[labels == 0].mean(axis=0)
            C1 = X[labels == 1].mean(axis=0)
            shift = float(np.linalg.norm(C0 - C[0]) + np.linalg.norm(C1 - C[1]))
            C[0], C[1] = C0, C1
            if shift < 1e-6:
                break

        inertia = float(D0[labels == 0].sum() + D1[labels == 1].sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best = (labels.copy(), C.copy(), inertia)

    return best

def _pca2_scores(X):
    """
    PCA scores (PC1, PC2) from centered X [N,L] using SVD.
    Returns scores [N,2].
    """
    X = np.asarray(X, dtype=np.float32)
    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        # Project onto first two PCs
        scores = Xc @ Vt[:2].T
    except np.linalg.LinAlgError:
        # fallback: random tiny jitter to avoid SVD failure
        Xcj = Xc + 1e-6 * np.random.standard_normal(Xc.shape).astype(np.float32)
        U, S, Vt = np.linalg.svd(Xcj, full_matrices=False)
        scores = Xcj @ Vt[:2].T
    return scores.astype(np.float32)

def _dprime_centroid_axis(X, labels, centers):
    """
    Project onto centroid-difference axis and compute 1-D d' between clusters.
    """
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    mu0, mu1 = centers[0], centers[1]
    w = mu1 - mu0
    nw = float(np.linalg.norm(w))
    if nw < 1e-12:
        return float('nan')
    w /= nw
    y = X @ w  # projections
    y0 = y[labels == 0]; y1 = y[labels == 1]
    if y0.size < 2 or y1.size < 2:
        return float('nan')
    m0, m1 = float(y0.mean()), float(y1.mean())
    s0 = float(np.std(y0, ddof=1)); s1 = float(np.std(y1, ddof=1))
    denom = np.sqrt(0.5 * (s0**2 + s1**2))
    return abs(m1 - m0) / denom if denom > 1e-12 else float('inf')

def kmeans_split_diagnostics(snips_cube, ei_ref, channel_ids=None,
                             rms_thr=10.0, n_init=8, max_iter=60,
                             ncols=6, title_prefix=""):
    """
    For each channel with RMS(ei_ref[ch]) > rms_thr:
      - X = per-spike waveform on that channel (full L samples)
      - K-means (k=2) on X (centered across spikes), no feature scaling
      - PCA to 2D for plotting (PC1 vs PC2), color by cluster
      - Title: ch id, RMS, d' along centroid axis, cluster sizes

    snips_cube: [C, L, N_spikes]
    ei_ref:     [C, L]  (the 'new EI' used to rank channels by RMS)
    """

    S = np.asarray(snips_cube, dtype=np.float32)
    E = np.asarray(ei_ref, dtype=np.float32)
    C1, L1, N = S.shape
    C2, L2 = E.shape
    C = min(C1, C2); L = min(L1, L2)
    if (C1 != C2) or (L1 != L2):
        print(f"[kmeans_diag] shape mismatch: snips {S.shape}, EI {E.shape} → using [C={C}, L={L}] common slice")
        S = S[:C, :L, :]
        E = E[:C, :L]

    if channel_ids is None:
        channel_ids = np.arange(C, dtype=int)
    else:
        channel_ids = np.asarray(channel_ids, dtype=int)
        if channel_ids.size != C:
            channel_ids = np.arange(C, dtype=int)

    # Channels to analyze (sorted by RMS on EI)
    rms = np.sqrt(np.mean(E**2, axis=1))  # [C]
    keep = np.flatnonzero(rms > float(rms_thr)).astype(int)
    if keep.size == 0:
        print(f"[kmeans_diag] No channels pass RMS>{rms_thr}. Nothing to show.")
        return {"channels": np.array([]), "dprime": np.array([])}

    order = keep[np.argsort(-rms[keep])]  # sort by RMS descending
    M = order.size

    # Prepare figure grid (4 per row)
    ncols = int(ncols)
    nrows = int(np.ceil(M / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0*ncols, 1.0*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    dprime_list = np.full(M, np.nan, dtype=np.float32)
    ch_list = np.zeros(M, dtype=int)

    m = 0
    for idx, ch in enumerate(order):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        X = S[ch].T  # [N, L] full waveform per spike
        # Skip degenerate channels
        if N < 4 or np.allclose(X, X[0]):
            ax.axis('off')
            continue

        # Center across spikes (feature-wise mean subtraction); keep raw scale
        Xc = X - X.mean(axis=0, keepdims=True)

        labels, centers, _ = _kmeans2_mv(Xc, n_init=n_init, max_iter=max_iter, random_state=None)
        dpr = _dprime_centroid_axis(Xc, labels, centers)

        scores = _pca2_scores(Xc)  # [N, 2]
        # colors for 0/1
        cols = np.where(labels == 0, 'C0', 'C1')
        ax.scatter(scores[:, 0], scores[:, 1], c=cols, s=9, alpha=0.85)
        ax.set_xticks([]); ax.set_yticks([])
        n0 = int((labels == 0).sum()); n1 = int((labels == 1).sum())
        ax.set_title(f"ch {int(channel_ids[ch])} | RMS={rms[ch]:.1f} | d'={dpr:.2f} | n={n0}/{n1}", fontsize=9)

        dprime_list[m] = dpr
        ch_list[m] = int(channel_ids[ch])
        m += 1

    # Hide any unused subplots
    for k in range(M, nrows*ncols):
        r = k // ncols; c = k % ncols
        axes[r, c].axis('off')

    fig.suptitle(f"{title_prefix} k-means (k=2) per channel (RMS>{rms_thr})", y=1.02)
    fig.tight_layout()
    plt.show()

    return {"channels": ch_list[:m], "dprime": dprime_list[:m]}

def split_first_good_channel_and_visualize(snips, ei_best_centered, ei_positions,
                                           rms_thr=10.0, dprime_thr=5.0, min_per_cluster=10,
                                           n_init=8, max_iter=60, lag_radius=0):
    """
    1) Find first channel (by RMS on ei_best_centered) with d'>dprime_thr and both clusters >= min_per_cluster.
    2) Use its k=2 split to:
       - compute EIs per cluster,
       - overlay plot (black = larger-main-channel EI, red = smaller),
       - build union of channels with RMS>rms_thr across the two EIs,
       - compute & plot harm map on those channels for each EI,
       - run global k=2 on flattened multi-channel waveforms over the union channels,
         plot PC1 vs PC2 scatter, and print the number of spikes that changed clusters.

    snips:             [C, L, N]
    ei_best_centered:  [C, L]
    ei_positions:      [C, 2]
    """


    S = np.asarray(snips, dtype=np.float32)
    E = np.asarray(ei_best_centered, dtype=np.float32)
    C, L, N = S.shape
    assert E.shape == (C, L), f"ei_best_centered shape {E.shape} != snips spatial/temporal {(C, L)}"

    # 1) Search channels by RMS descending on ei_best_centered
    rms = np.sqrt(np.mean(E**2, axis=1)).astype(np.float32)
    cand = np.flatnonzero(rms > float(rms_thr)).astype(int)
    if cand.size == 0:
        print(f"[split] No channels pass RMS>{rms_thr}. Aborting.")
        return None
    order = cand[np.argsort(-rms[cand])]  # RMS descending

    chosen_ch = None
    chosen_labels = None
    chosen_dpr = None

    print(f"Total channels {len(order)} : {order}")

    for ch in order:

        print(f"currently k-means on : {ch}")
        X = S[ch].T  # [N, L], full waveform per spike
        if N < 25:
            continue
        # center across spikes, feature-wise; keep raw scale
        Xc = X - X.mean(axis=0, keepdims=True)

        labels, centers, _ = _kmeans2_mv(Xc, n_init=n_init, max_iter=max_iter, random_state=None)
        n0 = int((labels == 0).sum()); n1 = int((labels == 1).sum())
        if n0 < min_per_cluster or n1 < min_per_cluster:
            continue
        dpr = _dprime_centroid_axis(Xc, labels, centers)
        if np.isfinite(dpr) and dpr > float(dprime_thr):
            chosen_ch = int(ch)
            chosen_labels = labels.astype(np.int32)
            chosen_dpr = float(dpr)
            break

    if chosen_ch is None:
        print(f"[split] No channel met d'>{dprime_thr} with >= {min_per_cluster} spikes per cluster.")
        return None

    idx0 = np.flatnonzero(chosen_labels == 0).astype(int)
    idx1 = np.flatnonzero(chosen_labels == 1).astype(int)
    N0, N1 = idx0.size, idx1.size

    # 2) EIs from each cluster (full array)
    EI0 = S[:, :, idx0].mean(axis=2).astype(np.float32)
    EI1 = S[:, :, idx1].mean(axis=2).astype(np.float32)

    # Determine overlay order: black first (larger main-channel amplitude), red second (smaller)
    # try:
    #     ch0, _ = main_channel_and_neg_peak(EI0)
    # except Exception:
    #     ch0 = int(np.argmin(EI0.min(axis=1)))
    # try:
    #     ch1, _ = main_channel_and_neg_peak(EI1)
    # except Exception:
    #     ch1 = int(np.argmin(EI1.min(axis=1)))

    # A0 = float(-EI0[ch0].min())
    # A1 = float(-EI1[ch1].min())
    # if A0 >= A1:
    #     EIs_plot = [EI0, EI1]  # black, red
    #     labels_plot = ("cluster 0", "cluster 1")
    # else:
    #     EIs_plot = [EI1, EI0]  # black, red
    #     labels_plot = ("cluster 1", "cluster 0")

    # Overlay plot
    # fig, ax = plt.subplots(figsize=(20, 12))
    # pew.plot_ei_waveforms(
    #     EIs_plot, ei_positions,
    #     ref_channel=int(chosen_ch),
    #     scale=70.0, box_height=1.0, box_width=50.0,
    #     colors=['black', 'red'], linewidth=[0.8, 1.3], alpha=[0.75, 1.0],
    #     ax=ax
    # )
    # ax.set_title(f"Overlay EIs | split on ch {chosen_ch} (d'={chosen_dpr:.2f}) | n={N0}/{N1}\n"
    #              f"black={labels_plot[0]}, red={labels_plot[1]}")
    # plt.show()

    # 3) Union of channels with RMS>rms_thr across the two EIs
    rms0 = np.sqrt(np.mean(EI0**2, axis=1))
    rms1 = np.sqrt(np.mean(EI1**2, axis=1))
    union_ch = np.flatnonzero((rms0 > float(rms_thr)) | (rms1 > float(rms_thr))).astype(int)
    if union_ch.size == 0:
        print(f"[split] Union had no channels above RMS>{rms_thr}. Aborting harm-map step.")
        return {
            "chosen_channel": chosen_ch, "dprime": chosen_dpr,
            "idx0": idx0, "idx1": idx1,
            "EI0": EI0, "EI1": EI1, "union_channels": union_ch
        }
    print(union_ch.size)

    # 4) Harm maps on union channels for each EI
    def _harm_and_plot(EI, idx, tag):
        sn = S[union_ch][:, :, idx]            # [K, L, N_tag]
        ei_u = EI[union_ch]                    # [K, L]
        res = compute_harm_map_noamp(
            ei=ei_u, snips=sn,
            p2p_thr=-1e9, max_channels=union_ch.size, min_channels=union_ch.size,
            lag_radius=lag_radius, weight_by_p2p=True, weight_beta=0.7,
            force_include_main=True
        )
        # plot_harm_heatmap(res, field="harm_matrix",
        #                   title=f"Harm map ({tag}) | ch {chosen_ch} | K={union_ch.size}, N={sn.shape[2]}")
        return res

    res0 = _harm_and_plot(EI0, idx0, "cluster 0")
    res1 = _harm_and_plot(EI1, idx1, "cluster 1")

    # 5) Global k=2 on concatenated spikes over union channels, scatter PC1 vs PC2, and changed count
    idx_all = np.concatenate([idx0, idx1], axis=0)
    orig = np.concatenate([np.zeros(N0, dtype=np.int32), np.ones(N1, dtype=np.int32)], axis=0)

    S_all = S[union_ch][:, :, idx_all]  # [K, L, N_all]
    N_all = S_all.shape[2]
    X = S_all.transpose(2, 0, 1).reshape(N_all, -1).astype(np.float32)  # [N_all, K*L]
    Xc = X - X.mean(axis=0, keepdims=True)

    labels2, centers2, _ = _kmeans2_mv(Xc, n_init=n_init, max_iter=max_iter, random_state=None)

    # Best label permutation against the original split
    mism0 = int((labels2 != orig).sum())
    mism1 = int(((1 - labels2) != orig).sum())
    changed = min(mism0, mism1)

    # Scatter plot PC1 vs PC2
    # scores = _pca2_scores(Xc)  # [N_all, 2]
    # col = np.where(labels2 == 0, 'C0', 'C1')
    # fig, ax = plt.subplots(figsize=(6, 5))
    # ax.scatter(scores[:, 0], scores[:, 1], c=col, s=10, alpha=0.85)
    # ax.set_title(f"Global k-means (union channels) | changed={changed}/{N_all}")
    # ax.set_xticks([]); ax.set_yticks([])
    # plt.show()

    print(f"[split] Chosen channel: {chosen_ch} | d'={chosen_dpr:.2f} | n0={N0}, n1={N1}")
    print(f"[split] Union channels (RMS>{rms_thr}): {union_ch.size} channels")
    print(f"[split] Spikes that changed clusters vs original split: {changed} / {N0+N1}")

    return {
        "chosen_channel": chosen_ch,
        "dprime": chosen_dpr,
        "idx0": idx0, "idx1": idx1,
        "EI0": EI0, "EI1": EI1,
        "union_channels": union_ch,
        "harm0": res0, "harm1": res1,
        "changed": changed
    }


def _mean_over_indices_streaming(S, idx, block=2048):
    # S: [C,L,N], idx: 1D int (absolute in S)
    idx = np.asarray(idx, dtype=int).ravel()
    C, L, _ = S.shape
    out = np.zeros((C, L), dtype=np.float32)
    n = idx.size
    if n == 0:
        return out
    for s in range(0, n, int(block)):
        jj = idx[s:min(s+int(block), n)]
        out += S[:, :, jj].sum(axis=2, dtype=np.float32)
    out /= float(n)
    return out

def cosine_two_eis(ei0, ei1, rms_gate=5.0, use_abs=True, best_align_lag=6):
    """
    Cosine similarity between two EIs with pairwise RMS>rms_gate channel masking
    and optional best global time alignment in [-best_align_lag, +best_align_lag].
    Returns: (cosine, nch_used, best_lag, overlap_len)
    """
    import numpy as np
    ei0 = np.asarray(ei0, dtype=np.float32)
    ei1 = np.asarray(ei1, dtype=np.float32)
    if ei0.shape != ei1.shape:
        C = min(ei0.shape[0], ei1.shape[0]); L = min(ei0.shape[1], ei1.shape[1])
        ei0 = ei0[:C, :L]; ei1 = ei1[:C, :L]
    rms0 = np.sqrt(np.nanmean(ei0**2, axis=1)).astype(np.float32)
    rms1 = np.sqrt(np.nanmean(ei1**2, axis=1)).astype(np.float32)
    mask = (rms0 > rms_gate) | (rms1 > rms_gate)
    mcount = int(mask.sum())
    if mcount == 0:
        return np.nan, 0, 0, 0

    x = ei0[mask, :]; y = ei1[mask, :]
    L = x.shape[1]; Lmax = int(best_align_lag)
    best_score = -np.inf; best_s = 0; best_len = 0

    for s in range(-Lmax, Lmax + 1):
        if s == 0:
            x_seg, y_seg = x, y
        elif s > 0:
            if s >= L: continue
            x_seg, y_seg = x[:, :L - s], y[:, s:]
        else:
            t = -s
            if t >= L: continue
            x_seg, y_seg = x[:, t:], y[:, :L - t]
        xv = x_seg.ravel(); yv = y_seg.ravel()
        ni = np.linalg.norm(xv); nj = np.linalg.norm(yv)
        if ni == 0.0 or nj == 0.0: continue
        v = float(np.dot(xv, yv) / (ni * nj))
        score = abs(v) if use_abs else v
        if score > best_score:
            best_score = score; best_s = s; best_len = x_seg.shape[1]

    if best_score == -np.inf:
        return np.nan, mcount, 0, 0
    return np.float32(best_score), mcount, best_s, best_len


def cosine_two_eis_asym(ei0, ei1, rms_gate=5.0, use_abs=True, best_align_lag=6, shared_lag=True):
    """
    Asymmetric cosine similarities between two EIs.

    Returns a triple (cos_on_ei0, cos_on_ei1, cos_on_union), where masks are:
      m0 = (RMS(ei0) > rms_gate), m1 = (RMS(ei1) > rms_gate), mu = m0 | m1.

    Lag handling:
      - If shared_lag=True (default): pick the single best lag on the UNION mask,
        then evaluate all three cosines at that lag (good for comparability).
      - If shared_lag=False: each mask picks its own best lag.

    Cosines are |cos| if use_abs=True.
    np.nan is returned for any mask with zero channels passing the gate.
    """
    import numpy as np

    ei0 = np.asarray(ei0, dtype=np.float32)
    ei1 = np.asarray(ei1, dtype=np.float32)

    # Align shapes defensively (crop to common [C,L])
    C = min(ei0.shape[0], ei1.shape[0])
    L = min(ei0.shape[1], ei1.shape[1])
    ei0 = ei0[:C, :L]
    ei1 = ei1[:C, :L]

    # Per-channel RMS + masks
    rms0 = np.sqrt(np.nanmean(ei0**2, axis=1)).astype(np.float32)
    rms1 = np.sqrt(np.nanmean(ei1**2, axis=1)).astype(np.float32)
    m0   = (rms0 > rms_gate)
    m1   = (rms1 > rms_gate)
    mu   = (m0 | m1)

    Lmax = int(best_align_lag)

    def _best_cos_for_mask(x, y, mask, lag_use=None):
        """Return best cosine (or cosine at given lag_use) restricted to mask."""
        if mask.sum() == 0:
            return np.nan

        X = x[mask, :]
        Y = y[mask, :]
        T = X.shape[1]

        def _cos(xs, ys):
            xv = xs.ravel(); yv = ys.ravel()
            ni = np.linalg.norm(xv); nj = np.linalg.norm(yv)
            if ni == 0.0 or nj == 0.0:
                return -np.inf
            v = float(np.dot(xv, yv) / (ni * nj))
            return abs(v) if use_abs else v

        # If a specific lag is provided, evaluate only that lag
        if lag_use is not None:
            s = int(lag_use)
            if s == 0:
                return np.float32(_cos(X, Y))
            elif s > 0:
                if s >= T: 
                    return np.nan
                return np.float32(_cos(X[:, :T - s], Y[:, s:]))
            else:
                t = -s
                if t >= T: 
                    return np.nan
                return np.float32(_cos(X[:, t:], Y[:, :T - t]))

        # Otherwise, find best lag in [-Lmax, Lmax]
        best = -np.inf
        for s in range(-Lmax, Lmax + 1):
            if s == 0:
                xs, ys = X, Y
            elif s > 0:
                if s >= T: 
                    continue
                xs, ys = X[:, :T - s], Y[:, s:]
            else:
                t = -s
                if t >= T: 
                    continue
                xs, ys = X[:, t:], Y[:, :T - t]
            c = _cos(xs, ys)
            if c > best:
                best = c
        return np.float32(best if best != -np.inf else np.nan)

    # Decide lag once on the union (if shared_lag), then reuse for all masks
    if shared_lag and mu.any():
        # find union-best lag
        XU = ei0[mu, :]; YU = ei1[mu, :]
        T  = XU.shape[1]
        best = -np.inf; best_s = 0
        for s in range(-Lmax, Lmax + 1):
            if s == 0:
                xs, ys = XU, YU
            elif s > 0:
                if s >= T: 
                    continue
                xs, ys = XU[:, :T - s], YU[:, s:]
            else:
                t = -s
                if t >= T: 
                    continue
                xs, ys = XU[:, t:], YU[:, :T - t]
            # cosine at this lag
            xv = xs.ravel(); yv = ys.ravel()
            ni = np.linalg.norm(xv); nj = np.linalg.norm(yv)
            if ni == 0.0 or nj == 0.0:
                continue
            v = float(np.dot(xv, yv) / (ni * nj))
            sc = abs(v) if use_abs else v
            if sc > best:
                best = sc; best_s = s

        cos_u = np.float32(best if best != -np.inf else np.nan)
        # Evaluate other masks at the same lag for comparability
        cos_0 = _best_cos_for_mask(ei0, ei1, m0, lag_use=best_s)
        cos_1 = _best_cos_for_mask(ei0, ei1, m1, lag_use=best_s)
        return cos_0, cos_1, cos_u

    # Per-mask independent lag search
    cos_0 = _best_cos_for_mask(ei0, ei1, m0, lag_use=None)
    cos_1 = _best_cos_for_mask(ei0, ei1, m1, lag_use=None)
    cos_u = _best_cos_for_mask(ei0, ei1, mu, lag_use=None)
    return cos_0, cos_1, cos_u



def split_first_good_channel_subset_and_harm(
    snips, ei_centered, ei_positions,
    *, rms_thr=6.0, dprime_thr=3.0, min_per_cluster=10,
    n_init=8, max_iter=60, lag_radius=0,
    max_diag_channels=30, subsample_n=8000, subsample_seed=0,
    ei_subsample_n=1000, ei_subsample_seed=0,
    active_idx=None,                 # streaming pool
    assign_block=8192,               # streaming block size
    # --- NEW: skip spurious channels with near-identical child EIs (subset-based) ---
    try_next_if_similar=True,
    sim_reject_cos_thr=0.95,
    sim_gate_rms=5.0,
    sim_best_align_lag=6,
    sim_channels_max=24,
    sim_ei_cap=800                   # per child, from the subset
):
    """
    Fast split: per-channel k-means on a SUBSET to pick the best channel by d′,
    then one-shot assign ALL spikes using the two learned centroids (no global k-means).
    Returns a res_split dict like your original function or None if no split.
    """
    import numpy as np
    # We assume these exist (same ones your original uses)
    from joint_utils import _kmeans2_mv, _dprime_centroid_axis

    C, L, N = snips.shape

    def _ei_mean_on_channels(S, chsel, idx, block=256):
        chsel = np.asarray(chsel, dtype=int).ravel()
        idx   = np.asarray(idx,   dtype=int).ravel()
        M = chsel.size
        Lloc = S.shape[1]
        out = np.zeros((M, Lloc), dtype=np.float32)
        n = idx.size
        if n == 0:
            return out

        # precompute the middle-axis slice as an index array for np.ix_
        L_idx = np.arange(Lloc, dtype=int)

        for s in range(0, n, int(block)):
            jj = idx[s:min(s + int(block), n)]
            # Use np.ix_ to form (M, L, b) without broadcasting errors or big temps
            blk = S[np.ix_(chsel, L_idx, jj)]            # shape (M, L, b)
            out += blk.sum(axis=2, dtype=np.float32)

        out /= float(n)
        out -= out.mean(axis=1, keepdims=True)
        return out

    if active_idx is None:
        idx_all = np.arange(N, dtype=int)
    else:
        idx_all = np.asarray(active_idx, dtype=int).ravel()
    N_active = int(idx_all.size)

    # ---- candidate channels: RMS gate + top-K by RMS
    rms = np.sqrt(np.mean(ei_centered**2, axis=1)).astype(np.float32)  # [C]
    chans = np.flatnonzero(rms > rms_thr).astype(int)
    if chans.size == 0:
        print("[subset-split] no channels pass RMS gate"); return None
    
    order = chans[np.argsort(-rms[chans])]
    order = order[:int(max_diag_channels)]

    # Channels to use when checking EI similarity (keep small for speed)
    _sim_ch = np.argsort(-rms)[:int(sim_channels_max)].astype(int)
    rng = np.random.default_rng(subsample_seed)

    best_any = dict(dpr=-np.inf, ch=None, centers=None, mu_feat=None, n_sub=0)  # fallback
    best = dict(dpr=-np.inf, ch=None, centers=None, mu_feat=None, n_sub=0)      # chosen after sim-check

    # ---- evaluate each channel on a subset
    for ch in order:

        # N_active is the cohort size at this node
        if N_active < max(25, 2*min_per_cluster):
            continue

        # choose indices on the active pool
        if (subsample_n is None) or (N_active <= int(subsample_n)):
            idx_sub_local = idx_all
        else:
            idx_sub_local = rng.choice(idx_all, size=int(subsample_n), replace=False)

        # subset matrix (samples, L) — NOTE: no .T here (advanced indexing yields (s,L))
        X_sub = snips[ch, :, idx_sub_local].astype(np.float32)
        mu_feat = X_sub.mean(axis=0, keepdims=True)
        Xc_sub = X_sub - mu_feat

        labels_sub, centers_sub, _ = _kmeans2_mv(
            Xc_sub, n_init=max(3, min(4, n_init)), max_iter=min(50, max_iter), random_state=None
        )
        n0s = int((labels_sub == 0).sum()); n1s = int((labels_sub == 1).sum())
        if n0s < min_per_cluster or n1s < min_per_cluster:
            continue

        dpr_sub = _dprime_centroid_axis(Xc_sub, labels_sub, centers_sub)
        if not (np.isfinite(dpr_sub) and (dpr_sub > float(dprime_thr))):
            continue

        # track the best by d′ (fallback if all are rejected by the similarity check)
        if dpr_sub > best_any["dpr"]:
            best_any.update(dict(
                dpr=float(dpr_sub), ch=int(ch),
                centers=centers_sub.copy(), mu_feat=mu_feat.copy(),
                n_sub=int(idx_sub_local.size)
            ))

        # Optional: reject “spurious” splits whose child EIs are near-identical on a small channel set
        if try_next_if_similar:
            idx0_sub = idx_sub_local[labels_sub == 0]
            idx1_sub = idx_sub_local[labels_sub == 1]
            # cap per child for the EI similarity check
            if sim_ei_cap is not None:
                if idx0_sub.size > int(sim_ei_cap):
                    idx0_sub = rng.choice(idx0_sub, size=int(sim_ei_cap), replace=False)
                if idx1_sub.size > int(sim_ei_cap):
                    idx1_sub = rng.choice(idx1_sub, size=int(sim_ei_cap), replace=False)

            # build small EIs on a small channel set
            EI0_sub = _ei_mean_on_channels(snips, _sim_ch, idx0_sub, block=256)
            EI1_sub = _ei_mean_on_channels(snips, _sim_ch, idx1_sub, block=256)

            # cosine similarity on subset EIs; if "too similar", try next channel
            cos_sim, nch_s, lag_s, olen_s = cosine_two_eis(
                EI0_sub, EI1_sub,
                rms_gate=float(sim_gate_rms), use_abs=True, best_align_lag=int(sim_best_align_lag)
            )
            if np.isfinite(cos_sim) and (cos_sim >= float(sim_reject_cos_thr)):
                # reject this channel; continue to next candidate
                continue

        # accept this channel (either similarity check is off, or cos < threshold)
        best.update(dict(
            dpr=float(dpr_sub), ch=int(ch),
            centers=centers_sub.copy(), mu_feat=mu_feat.copy(),
            n_sub=int(idx_sub_local.size)
        ))
        break  # stop scanning channels once an acceptable one is found


    if best["ch"] is None:
        # No channel passed the similarity check — fall back to best by d′ (original behavior)
        if best_any["ch"] is None:
            print("[subset-split] no channel produced a valid subset split"); return None
        else:
            best = best_any


    # ---- assign ALL spikes using equidistance to the two subset centroids
    ch = best["ch"]
    mu0, mu1 = best["centers"][0], best["centers"][1]
    w = (mu1 - mu0).astype(np.float32)
    b = 0.5 * (float(np.dot(mu1, mu1)) - float(np.dot(mu0, mu0)))

    g = np.empty(N_active, dtype=np.float32)
    mu_feat = best["mu_feat"].astype(np.float32)

    for s in range(0, N_active, int(assign_block)):
        j = slice(s, min(s + int(assign_block), N_active))
        jj = idx_all[j]
        # [block, L]
        Xb = snips[ch, :, jj].astype(np.float32)
        Xb -= mu_feat
        g[j] = Xb @ w - b

    labels = (g >= 0).astype(np.int32)
    idx0_local = np.flatnonzero(labels == 0).astype(int)
    idx1_local = np.flatnonzero(labels == 1).astype(int)

    # map back to the ORIGINAL snips index space
    idx0 = idx_all[idx0_local]
    idx1 = idx_all[idx1_local]

    if (idx0.size < min_per_cluster) or (idx1.size < min_per_cluster):
        print(f"[subset-split] best ch {ch} failed min cluster on full assign "
              f"(N0={idx0.size}, N1={idx1.size})")
        return None

    # --- refine centers + compute fair d′ on FULL assignment (streaming, no Xc_full) ---
    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())

    # guard: if one side empty (shouldn't happen due to earlier checks), fall back
    if (n0 == 0) or (n1 == 0):
        centers_full = best["centers"].astype(np.float32)
        dpr_full = np.nan
    else:
        # Pass 1: cluster means in centered space (Xc = X - mu_feat)
        L = int(best["mu_feat"].shape[1])
        S0 = np.zeros(L, dtype=np.float32)
        S1 = np.zeros(L, dtype=np.float32)

        for s in range(0, N_active, int(assign_block)):
            j  = slice(s, min(s + int(assign_block), N_active))
            jj = idx_all[j]
            Xb = snips[ch, :, jj].astype(np.float32)    # shape (B, L)
            Xc = Xb - best["mu_feat"].astype(np.float32)     # center by subset mean

            m0 = (labels[j] == 0)
            if m0.any():
                S0 += Xc[m0].sum(axis=0)
            m1 = ~m0
            if m1.any():
                S1 += Xc[m1].sum(axis=0)

        mu0_full = S0 / float(n0)
        mu1_full = S1 / float(n1)
        centers_full = np.vstack([mu0_full, mu1_full]).astype(np.float32)

        # Pass 2: d′ along the centroid-difference axis (unit vector)
        w = (mu1_full - mu0_full).astype(np.float32)
        nw = float(np.linalg.norm(w))
        if nw == 0.0:
            dpr_full = np.nan
        else:
            w /= nw
            n0a = 0; n1a = 0
            s0  = 0.0; s1  = 0.0
            q0  = 0.0; q1  = 0.0

            for s in range(0, N_active, int(assign_block)):
                j  = slice(s, min(s + int(assign_block), N_active))
                jj = idx_all[j]
                Xb = snips[ch, :, jj].astype(np.float32)    # shape (B, L)
                Xc = Xb - best["mu_feat"].astype(np.float32)
                y  = Xc @ w                                      # [B]

                m0 = (labels[j] == 0)
                if m0.any():
                    y0 = y[m0]
                    n0a += y0.size
                    s0  += float(y0.sum())
                    q0  += float(np.dot(y0, y0))

                m1 = ~m0
                if m1.any():
                    y1 = y[m1]
                    n1a += y1.size
                    s1  += float(y1.sum())
                    q1  += float(np.dot(y1, y1))

            # robust d′ with ddof=1
            m0hat = s0 / max(n0a, 1)
            m1hat = s1 / max(n1a, 1)
            v0 = (q0 - n0a * (m0hat * m0hat)) / max(n0a - 1, 1)
            v1 = (q1 - n1a * (m1hat * m1hat)) / max(n1a - 1, 1)
            dpr_full = abs(m1hat - m0hat) / np.sqrt(0.5 * (v0 + v1) + 1e-12)

    print(f"[subset-split] chosen ch={ch}  d′_subset={best['dpr']:.3f} (s={best['n_sub']})  "
        f"d′_full={dpr_full:.3f}  N0={idx0.size} N1={idx1.size}")



    # ---- build EIs from a SUBSAMPLE per child (fast) ----
    _rng_ei = _np.random.default_rng(int(ei_subsample_seed))

    def _pick_subset(idx, cap):
        idx = _np.asarray(idx, dtype=int)
        if cap is None or idx.size <= int(cap):
            return idx
        return _rng_ei.choice(idx, size=int(cap), replace=False)

    idx0_ei = _pick_subset(idx0, ei_subsample_n)
    idx1_ei = _pick_subset(idx1, ei_subsample_n)

    # (C, L, n_sub) -> mean over spikes
    EI0 = _mean_over_indices_streaming(snips, idx0_ei)
    EI1 = _mean_over_indices_streaming(snips, idx1_ei)

    # channel-wise zero-centering
    EI0 -= EI0.mean(axis=1, keepdims=True)
    EI1 -= EI1.mean(axis=1, keepdims=True)

    print(f"[EI] subsampled: N0={idx0.size}→{idx0_ei.size}, N1={idx1.size}→{idx1_ei.size}")


    # EI0 = snips[:, :, idx0].mean(axis=2).astype(np.float32)
    # EI1 = snips[:, :, idx1].mean(axis=2).astype(np.float32)
    # EI0 = EI0 - EI0.mean(axis=1, keepdims=True)
    # EI1 = EI1 - EI1.mean(axis=1, keepdims=True)

    return {
        "ch": ch,
        "idx0": idx0, "idx1": idx1,
        "EI0": EI0, "EI1": EI1,
        "dprime_subset": best["dpr"], "dprime_full": float(dpr_full),
        "n_sub": best["n_sub"]
    }


def plot_grouped_harm_maps_two_eis(
    EI0, EI1, snips, idx0, idx1,
    p2p_thr=30.0, max_channels=80, min_channels=10,
    lag_radius=0, weight_by_p2p=True, weight_beta=0.7,
    title_prefix=""
):
    """
    Compute and plot harm maps for two EIs, using each EI to choose channels.
    Spikes (columns) are ordered as idx0 then idx1 (both sorted).

    Args
    ----
    EI0, EI1 : [C, L] float
        Two EIs (same C,L as snips).
    snips : [C, L, N] float
        Snippets for all explained spikes (all channels).
    idx0, idx1 : 1D int arrays
        Spike indices (into snips' third axis). They need not be disjoint,
        but typically are.

    Kwargs tune channel picking and harm-map scoring.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from collision_utils import (
        select_template_channels,
        main_channel_and_neg_peak,
        compute_harm_map_noamp,
        plot_harm_heatmap,
    )

    # ---- Basic checks
    EI0 = np.asarray(EI0, dtype=np.float32)
    EI1 = np.asarray(EI1, dtype=np.float32)
    S   = np.asarray(snips, dtype=np.float32)

    C, L, N = S.shape
    assert EI0.shape == (C, L) and EI1.shape == (C, L), \
        f"EI shapes {EI0.shape}, {EI1.shape} do not match snips spatial/temporal {(C, L)}"

    idx0 = np.asarray(idx0, dtype=int).ravel()
    idx1 = np.asarray(idx1, dtype=int).ravel()
    if idx0.size == 0 or idx1.size == 0:
        print("[grouped-harm] One of the groups is empty; aborting.")
        return None

    if (idx0.min() < 0) or (idx1.min() < 0) or (idx0.max() >= N) or (idx1.max() >= N):
        raise IndexError("[grouped-harm] Spike indices out of bounds for snips.")

    # Optional: warn if overlap
    inter = np.intersect1d(idx0, idx1)
    if inter.size > 0:
        print(f"[grouped-harm] WARNING: groups overlap on {inter.size} spikes.")

    # ---- Spike ordering: group0 then group1 (sorted by index as requested)
    idx0_sorted = np.sort(idx0)
    idx1_sorted = np.sort(idx1)
    order = np.concatenate([idx0_sorted, idx1_sorted], axis=0)
    N0 = idx0_sorted.size
    N1 = idx1_sorted.size
    S_ord = S[:, :, order]   # [C, L, N0+N1]

    def _build_and_plot(EI, tag):
        # Channel selection from EI, then lock selection in harm-map
        chans, ptp = select_template_channels(
            EI, p2p_thr=p2p_thr, max_n=max_channels, min_n=min_channels, force_include_main=True
        )
        chans = np.asarray(chans, dtype=int)
        if chans.size == 0:
            print(f"[grouped-harm] EI '{tag}' produced no channels (p2p_thr={p2p_thr}).")
            return None

        # Be defensive: ensure EI's main channel is present
        try:
            ch_main, _ = main_channel_and_neg_peak(EI)
        except Exception:
            # fallback like in your earlier code
            mins = EI.min(axis=1)
            ch_main = int(np.argmin(mins))
        if ch_main not in chans:
            # swap in main channel by replacing weakest
            weakest = int(np.argmin(ptp[chans]))
            chans = chans.copy()
            chans[weakest] = int(ch_main)

        EI_sel = EI[chans]               # [K, L]
        S_sel  = S_ord[chans]            # [K, L, N0+N1]

        # Lock channel set by forcing min=max and p2p_thr very low
        res = compute_harm_map_noamp(
            ei=EI_sel, snips=S_sel,
            p2p_thr=-1e9, max_channels=chans.size, min_channels=chans.size,
            lag_radius=lag_radius, weight_by_p2p=weight_by_p2p, weight_beta=weight_beta,
            force_include_main=True
        )

        title = (f"{title_prefix} Harm map using EI {tag} | "
                 f"K={chans.size}, N0={N0}, N1={N1} | p2p_thr={p2p_thr}, lag={lag_radius}")
        try:
            # NOTE: plot_harm_heatmap creates/shows its own figure.
            # Do NOT call plt.gca() afterwards (it spawns a new empty figure on some backends).
            plot_harm_heatmap(res, field="harm_matrix", sort_by_ptp=False, title=title)
        except Exception:
            # Fallback: minimal imshow with separator line (we control the axes here)
            HM = np.asarray(res["harm_matrix"], dtype=np.float32)
            fig, ax = plt.subplots(figsize=(12, 5))
            im = ax.imshow(HM, aspect='auto', interpolation='nearest')
            ax.set_title(title)
            ax.axvline(N0 - 0.5, color='k', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.set_xlabel("spikes (idx0 | idx1)"); ax.set_ylabel("selected channels")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout(); plt.show()


        return {"res": res, "channels": chans, "N0": N0, "N1": N1}

    out0 = _build_and_plot(EI0, tag="0")
    out1 = _build_and_plot(EI1, tag="1")

    return {"EI0": out0, "EI1": out1, "order": order, "N0": N0, "N1": N1}


def classify_two_cells_vs_ab_shard(
    EI0, EI1, snips, idx0, idx1,
    p2p_thr=30.0, max_channels=80, min_channels=10,
    lag_radius=0, weight_by_p2p=True, weight_beta=0.7,
    rms_thr_support=10.0,
    asym_strong_z=2.0,     # z-like threshold for "strong" asymmetry
    asym_pure_z=1.0        # z-like threshold to consider "pure-like" (≈ no diff)
):
    """
    Decide "two cells" vs "AB sharding" from two EIs and spike groups.

    Inputs
    ------
    EI0, EI1 : [C, L]
    snips    : [C, L, N]
    idx0, idx1 : 1D indices into snips' third axis (spikes used to form EI0/EI1)

    What it computes
    ----------------
    - For each EI, select channels via select_template_channels, lock them,
      compute harm-map ΔRMS on (idx0|idx1) ordering.
    - Collapse ΔRMS to a per-spike score using p2p(EI)-weighted average across rows.
    - Summarize medians on group0 and group1, and robust "z" differences via MAD.
    - Measure support containment: RMS(EI) > rms_thr_support sets, overlap fractions.
    - Emit label: "two cells" vs "AB sharding" (+ which EI looks pure/collision).

    Returns dict with metrics and 'label'.
    """


    # Basic shapes / indexing
    EI0 = np.asarray(EI0, dtype=np.float32)
    EI1 = np.asarray(EI1, dtype=np.float32)
    S   = np.asarray(snips, dtype=np.float32)
    C, L, N = S.shape
    assert EI0.shape == (C, L) and EI1.shape == (C, L), "EI shapes must match [C,L] of snips"

    idx0 = np.asarray(idx0, dtype=int).ravel()
    idx1 = np.asarray(idx1, dtype=int).ravel()
    if idx0.size == 0 or idx1.size == 0:
        raise ValueError("Both groups must be non-empty.")
    if idx0.min() < 0 or idx1.min() < 0 or idx0.max() >= N or idx1.max() >= N:
        raise IndexError("Spike indices out of bounds.")

    # Spike order: group0 then group1
    idx0s = np.sort(idx0); idx1s = np.sort(idx1)
    order = np.concatenate([idx0s, idx1s])
    N0, N1 = idx0s.size, idx1s.size
    S_ord = S[:, :, order]

    def _harm_and_score(EI):
        # Channel selection from EI
        chans, ptp_all = select_template_channels(
            EI, p2p_thr=p2p_thr, max_n=max_channels, min_n=min_channels, force_include_main=True
        )
        chans = np.asarray(chans, dtype=int)
        if chans.size == 0:
            raise RuntimeError("No channels selected for EI (increase max_channels or lower p2p_thr).")

        # ensure main channel included
        try:
            ch_main, _ = main_channel_and_neg_peak(EI)
        except Exception:
            mins = EI.min(axis=1); ch_main = int(np.argmin(mins))
        if ch_main not in chans:
            # replace weakest by main channel
            ptp_sel = (EI[chans].max(axis=1) - EI[chans].min(axis=1))
            weakest = int(np.argmin(ptp_sel))
            chans = chans.copy(); chans[weakest] = ch_main

        # lock channels for harm-map
        EI_sel = EI[chans]
        S_sel  = S_ord[chans]

        res = compute_harm_map_noamp(
            ei=EI_sel, snips=S_sel,
            p2p_thr=-1e9, max_channels=chans.size, min_channels=chans.size,
            lag_radius=lag_radius, weight_by_p2p=weight_by_p2p, weight_beta=weight_beta,
            force_include_main=True
        )
        HM = np.asarray(res["harm_matrix"], dtype=np.float32)     # [K, N0+N1]

        # per-spike weighted score with p2p weights from the EI on selected channels
        p2p = (EI_sel.max(axis=1) - EI_sel.min(axis=1)).astype(np.float32)  # [K]
        w = p2p / (p2p.sum() + 1e-12)
        scores = (w[:, None] * HM).sum(axis=0)                    # [N0+N1]

        # robust group summaries
        s0 = scores[:N0]; s1 = scores[N0:]
        med0, med1 = float(np.median(s0)), float(np.median(s1))
        mad0 = float(np.median(np.abs(s0 - med0)) + 1e-9)
        mad1 = float(np.median(np.abs(s1 - med1)) + 1e-9)

        return {
            "res": res, "chans": chans,
            "scores": scores, "scores_g0": s0, "scores_g1": s1,
            "med0": med0, "med1": med1, "mad0": mad0, "mad1": mad1
        }

    out0 = _harm_and_score(EI0)
    out1 = _harm_and_score(EI1)

    # Asymmetries (positive means "other worse than own")
    # For EI0, own group is group0; for EI1, own group is group1.
    A0_raw = out0["med1"] - out0["med0"]
    A1_raw = out1["med0"] - out1["med1"]

    # Robust z-like effects using MAD of the "own" group distribution
    A0_z = A0_raw / out0["mad0"]
    A1_z = A1_raw / out1["mad1"]

    # Support containment via RMS>threshold sets
    rms0 = np.sqrt(np.mean(EI0**2, axis=1))
    rms1 = np.sqrt(np.mean(EI1**2, axis=1))
    sup0 = np.flatnonzero(rms0 > float(rms_thr_support))
    sup1 = np.flatnonzero(rms1 > float(rms_thr_support))
    inter = np.intersect1d(sup0, sup1).size
    f0 = inter / max(1, sup0.size)
    f1 = inter / max(1, sup1.size)
    containment = max(f0, f1)  # near 1 if one support is (almost) subset of the other

    # Decision logic
    # "two cells": both EIs show strong, symmetric asymmetry; supports not nested
    two_cells_flag = (A0_z >= asym_strong_z) and (A1_z >= asym_strong_z) and (containment < 0.8)

    # "AB sharding": one EI is "pure-like" (no asym), the other is strong asym; supports nested
    pure0 = (A0_z <= asym_pure_z)
    pure1 = (A1_z <= asym_pure_z)
    ab_flag = ((pure0 and (A1_z >= asym_strong_z)) or (pure1 and (A0_z >= asym_strong_z))) and (containment >= 0.8)

    if two_cells_flag:
        label = "two cells"
        details = {"pure_like": None, "collision_like": None}
    elif ab_flag:
        if pure0 and (A1_z >= asym_strong_z):
            label = "AB sharding"
            details = {"pure_like": 0, "collision_like": 1}
        else:
            label = "AB sharding"
            details = {"pure_like": 1, "collision_like": 0}
    else:
        # tie-breaker: pick the hypothesis with more evidence
        score_two = (A0_z >= asym_strong_z) + (A1_z >= asym_strong_z) + (containment < 0.7)
        score_ab  = (pure0 or pure1) + (containment >= 0.85)
        label = "two cells" if score_two > score_ab else "AB sharding"
        details = {"pure_like": 0 if pure0 else (1 if pure1 else None),
                   "collision_like": 1 if pure0 else (0 if pure1 else None)}

    # Print a compact summary table
    print("\n=== Harm-map block summary (p2p-weighted medians; lower is better) ===")
    print(f"EI0 on group0 (own): {out0['med0']:.3f}    EI0 on group1: {out0['med1']:.3f}    Δ={A0_raw:.3f}   z≈{A0_z:.2f}")
    print(f"EI1 on group1 (own): {out1['med1']:.3f}    EI1 on group0: {out1['med0']:.3f}    Δ={A1_raw:.3f}   z≈{A1_z:.2f}")
    print(f"Support containment (RMS>{rms_thr_support}): "
          f"|S0∩S1|/|S0|={f0:.2f}, |S0∩S1|/|S1|={f1:.2f} → containment={containment:.2f}")
    print(f"→ Classification: {label}  {('(pure≈EI'+str(details['pure_like'])+', collision≈EI'+str(details['collision_like'])+')') if details['pure_like'] is not None else ''}\n")

    return {
        "label": label,
        "details": details,
        "A0_raw": float(A0_raw), "A1_raw": float(A1_raw),
        "A0_z": float(A0_z), "A1_z": float(A1_z),
        "medians": {
            "EI0:g0": out0["med0"], "EI0:g1": out0["med1"],
            "EI1:g1": out1["med1"], "EI1:g0": out1["med0"],
        },
        "containment": {
            "frac_S0_in_S1": f0, "frac_S1_in_S0": f1, "containment": containment,
            "S0_size": int(sup0.size), "S1_size": int(sup1.size)
        }
    }



# =========================
# Pipeline sweeps & pruners (swappable)
# Contract: each PRUNER takes (state, params) and returns (state_out, report)
# state keys: {'snips'[C,L,N], 'ei'[C,L], 'ref_ch', 'ei_positions', 'idx'[N], 'meta', 'harm_cfg'}
# =========================

import numpy as _np

def _ensure_meta_(state):
    m = state.get("meta", None)
    if m is None:
        m = {}
    if "history" not in m:
        m["history"] = []
    state["meta"] = m
    return state

def _recompute_ei_(snips):
    # local import to avoid global header edits
    from collision_utils import median_ei_adaptive
    ei = median_ei_adaptive(_np.asarray(snips, dtype=_np.float32))
    # keep your existing convention: recenter to strongest-channel trough ~ index 40
    return recenter_ei_to_ref_trough(ei, center_index=40)

def _filter_state_(state, keep_mask):
    keep_mask = _np.asarray(keep_mask, dtype=bool).ravel()
    S = state["snips"]
    assert S.ndim == 3, "snips must be [C,L,N]"
    N = S.shape[2]
    if keep_mask.size != N:
        raise ValueError(f"keep_mask length {keep_mask.size} != N {N}")
    # apply filter
    S2 = S[:, :, keep_mask]
    idx2 = state.get("idx", _np.arange(N, dtype=_np.int64))[keep_mask]
    state2 = dict(state)
    state2["snips"] = S2
    state2["idx"] = idx2
    state2["ei"] = _recompute_ei_(S2) if S2.shape[2] > 0 else state["ei"]
    return state2

def _compute_harm_map_with_optional_override_(ei, snips, harm_cfg):
    """
    Try to call compute_harm_map_noamp with channels_override if provided.
    If not supported by the installed function, fall back to "lock channels" by slicing and min=max=K.
    """
    chans_override = harm_cfg.get("channels_override", None)
    # shared kwargs (selector if no override)
    common = dict(
        p2p_thr=float(harm_cfg.get("p2p_thr", 30.0)),
        max_channels=int(harm_cfg.get("max_channels", snips.shape[0])),
        min_channels=int(harm_cfg.get("min_channels", 10)),
        lag_radius=int(harm_cfg.get("lag_radius", 0)),
        weight_by_p2p=bool(harm_cfg.get("weight_by_p2p", True)),
        weight_beta=float(harm_cfg.get("weight_beta", 0.7)),
        force_include_main=bool(harm_cfg.get("force_include_main", True)),
    )
    if chans_override is None:
        return compute_harm_map_noamp(
            ei=ei, snips=snips,
            **common
        )
    # override path
    chans = _np.asarray(chans_override, dtype=int).ravel()
    ei_sel = ei[chans]
    sn_sel = snips[chans]
    try:
        return compute_harm_map_noamp(
            ei=ei_sel, snips=sn_sel, channels_override=chans, **common
        )
    except TypeError:
        # installed version doesn't accept channels_override; hard-lock by p2p_thr and min=max
        return compute_harm_map_noamp(
            ei=ei_sel, snips=sn_sel,
            p2p_thr=-1e9, max_channels=chans.size, min_channels=chans.size,
            lag_radius=common["lag_radius"],
            weight_by_p2p=common["weight_by_p2p"], weight_beta=common["weight_beta"],
            force_include_main=common["force_include_main"],
        )

# -------------------------
# Sweep 1: detection (per-channel)
# -------------------------
def detect_negative_peaks_on_channel(trace, *, thr, min_gap, pre, post, max_events):
    """
    Find local minima <= thr with >= min_gap separation; enforce [pre, post] window bounds.

    Returns dict:
      picked : np.int64 indices (absolute in `trace`)
      wf     : [N, L] waveforms on the ref channel
      t      : [L] time axis (-pre..post)
    """
    x = _np.asarray(trace, dtype=_np.float32).ravel()
    # local minima below threshold
    cand = _np.where((x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:]) & (x[1:-1] <= float(thr)))[0] + 1
    picked = []
    last = -10**9
    for i in cand:
        if i - last >= int(min_gap) and (i - pre) >= 0 and (i + post) < x.size:
            picked.append(i)
            last = i
            if len(picked) >= int(max_events):
                break
    picked = _np.asarray(picked, dtype=_np.int64)
    L = int(pre) + int(post) + 1
    if picked.size == 0:
        return {"picked": picked, "wf": _np.zeros((0, L), _np.float32), "t": _np.arange(-pre, post+1)}
    wf = _np.stack([x[i - pre : i + post + 1] for i in picked], axis=0).astype(_np.float32)
    t = _np.arange(-pre, post+1, dtype=int)
    return {"picked": picked, "wf": wf, "t": t}

def select_top_by_amplitude(wf, *, top_n):
    """
    Given [N, L] waveforms on a reference channel, select top-N by (negative) trough amplitude.
    Returns dict with sel_idx, wf_sel, median_sel, amp_all.
    """
    W = _np.asarray(wf, dtype=_np.float32)
    if W.size == 0:
        return {"sel_idx": _np.array([], dtype=int), "wf_sel": W, "median_sel": _np.zeros((W.shape[1],), _np.float32), "amp_all": _np.array([], _np.float32)}
    amp_all = -W.min(axis=1)
    k = int(min(int(top_n), W.shape[0]))
    sel_idx = _np.argsort(-amp_all)[:k]
    wf_sel = W[sel_idx]
    med_sel = _np.median(wf_sel, axis=0).astype(_np.float32)
    return {"sel_idx": sel_idx.astype(int), "wf_sel": wf_sel, "median_sel": med_sel, "amp_all": amp_all.astype(_np.float32)}

# -------------------------
# Sweep 2: snippet extraction + EI build (full array)
# -------------------------
def extract_and_build_ei(raw_data, times, *, window_ei, channels=None, reducer="median"):
    """
    Use extract_snippets_fast_ram to build [C,L,N] snippets at `times` and aggregate EI.
    Returns an INITIAL state dict:
      {'snips','ei','idx','meta'}; 'idx' are local 0..N-1 positions corresponding to kept snippets.
    """
    from axolotl_utils_ram import extract_snippets_fast_ram
    from collision_utils import median_ei_adaptive

    times = _np.asarray(times, dtype=_np.int64).ravel()
    T, C = _np.asarray(raw_data).shape
    if channels is None or (isinstance(channels, str) and channels == "all"):
        channels = _np.arange(C, dtype=int)
    chs = _np.asarray(channels, dtype=int).ravel()
    snips, valid_times = extract_snippets_fast_ram(raw_data, times, tuple(window_ei), chs)
    if snips.shape[2] == 0:
        raise RuntimeError("No valid snippets (edge overlap?).")
    # map valid_times back to input times (best-effort)
    pos = {int(t): i for i, t in enumerate(times.tolist())}
    idx = _np.array([pos.get(int(t), -1) for t in valid_times], dtype=_np.int64)
    # EI
    if reducer == "median":
        ei = median_ei_adaptive(snips)
    else:
        ei = snips.mean(axis=2).astype(_np.float32)
    ei = recenter_ei_to_ref_trough(ei, center_index=40)
    return {
        "snips": snips.astype(_np.float32),
        "ei": ei.astype(_np.float32),
        "idx": idx,
        "meta": {"history": []}
    }

# -------------------------
# Sweep 3 (optional): harm-map diagnostic (no filtering)
# -------------------------
def harm_map_diagnostic(state, *, params=None):
    """
    Compute & store a harm-map for visualization, but DO NOT filter spikes.
    Stores result under state['meta']['last_harm'].
    """
    params = params or {}
    state = _ensure_meta_(state)
    harm_cfg = state.get("harm_cfg", {})
    res = _compute_harm_map_with_optional_override_(state["ei"], state["snips"], harm_cfg)
    state["meta"]["last_harm"] = res
    # Optional: plot via plot_harm_heatmap here if you want every call to visualize
    return state, {"name": "harm_map_diagnostic", "kept": state["snips"].shape[2], "dropped": 0}

# -------------------------
# PRUNER P1: Amplitude gate on main channel
# -------------------------
def amplitude_gate_pruner(state, *, params):
    """
    Drop spikes with main-channel trough amplitude below a threshold:
      thr = frac * mean(topk amplitudes)
    params: {'topk':int, 'frac':float}
    """
    from collision_utils import main_channel_and_neg_peak
    state = _ensure_meta_(state)
    S = state["snips"]; E = state["ei"]
    if S.shape[2] == 0:
        return state, {"name":"amplitude_gate", "kept":0, "dropped":0}
    try:
        ch0, t0 = main_channel_and_neg_peak(E)
    except Exception:
        mins = E.min(axis=1); ch0 = int(_np.argmin(mins)); t0 = int(_np.argmin(E[ch0]))
    amps = (-S[ch0, t0, :]).astype(_np.float32)
    topk = int(params.get("topk", 25))
    frac = float(params.get("frac", 0.75))
    k = min(topk, amps.size)
    mu_top = float(_np.mean(_np.sort(amps)[-k:])) if k > 0 else 0.0
    thr = frac * mu_top
    keep = (amps >= thr) if k > 0 else _np.ones(amps.size, bool)
    kept = int(keep.sum()); drop = int((~keep).sum())
    st2 = _filter_state_(state, keep)
    st2["meta"]["history"].append({"step":"amplitude_gate", "kept":kept, "dropped":drop, "topk":k, "frac":frac, "thr":thr, "main_ch":int(ch0), "t0":int(t0)})
    rep = {"name":"amplitude_gate", "kept":kept, "dropped":drop, "thr":thr, "topk":k, "frac":frac, "main_ch":int(ch0), "t0":int(t0)}
    return st2, rep

# -------------------------
# PRUNER P2: Two-cells vs AB-shard split
# -------------------------
def two_cells_or_ab_shard_pruner(state, *, params):
    """
    Use split_first_good_channel_and_visualize + classify_two_cells_vs_ab_shard.
    If label == 'two cells': keep the larger cohort; else keep all.
    params: {'rms_thr','dprime_thr','min_per_cluster','n_init','max_iter','lag_radius', plus classifier knobs if desired}
    """
    st = _ensure_meta_(state)
    S, E, ei_pos = st["snips"], st["ei"], st["ei_positions"]
    res_split = split_first_good_channel_and_visualize(
        S, E, ei_pos,
        rms_thr=float(params.get("rms_thr", 10.0)),
        dprime_thr=float(params.get("dprime_thr", 5.0)),
        min_per_cluster=int(params.get("min_per_cluster", 10)),
        n_init=int(params.get("n_init", 8)),
        max_iter=int(params.get("max_iter", 60)),
        lag_radius=int(params.get("lag_radius", 0))
    )
    if res_split is None:
        return st, {"name":"two_cells_or_ab_shard", "kept": S.shape[2], "dropped": 0, "note":"no split candidate"}
    # classify
    met = classify_two_cells_vs_ab_shard(
        res_split["EI0"], res_split["EI1"], S, res_split["idx0"], res_split["idx1"],
        p2p_thr=float(params.get("p2p_thr", 30.0)),
        max_channels=int(params.get("max_channels", 80)),
        min_channels=int(params.get("min_channels", 10)),
        lag_radius=int(params.get("lag_radius", 0)),
        weight_by_p2p=bool(params.get("weight_by_p2p", True)),
        weight_beta=float(params.get("weight_beta", 0.7)),
        rms_thr_support=float(params.get("rms_thr_support", 10.0)),
        asym_strong_z=float(params.get("asym_strong_z", 2.0)),
        asym_pure_z=float(params.get("asym_pure_z", 1.0))
    )
    if met["label"] != "two cells":
        # keep all (no change)
        st["meta"]["history"].append({"step":"two_cells_or_ab_shard", "label":met["label"], "kept":S.shape[2], "dropped":0})
        return st, {"name":"two_cells_or_ab_shard", "label": met["label"], "kept": S.shape[2], "dropped": 0}
    # keep larger cluster
    idx0, idx1 = res_split["idx0"], res_split["idx1"]
    keep_local = idx0 if idx0.size >= idx1.size else idx1
    mask = _np.zeros(S.shape[2], dtype=bool)
    mask[_np.asarray(keep_local, dtype=int)] = True
    st2 = _filter_state_(st, mask)
    kept, drop = int(mask.sum()), int((~mask).sum())
    st2["meta"]["history"].append({"step":"two_cells_or_ab_shard", "label":met["label"], "kept":kept, "dropped":drop})
    rep = {"name":"two_cells_or_ab_shard", "label":"two cells", "kept":kept, "dropped":drop}
    return st2, rep

# -------------------------
# PRUNER P3: Harm-map compliance (per-spike rules)
# -------------------------
def harm_compliance_pruner(state, *, params):
    """
    Enforce:
      mean(ΔRMS) <= MEAN_THR
      max(ΔRMS across channels) <= CHAN_THR
      ΔRMS on ref_ch < REF_THR (if ref_ch among selected channels)
    params: {'MEAN_THR','CHAN_THR','REF_THR'}
    Uses state['harm_cfg'] for harm-map computation.
    """
    st = _ensure_meta_(state)
    S, E = st["snips"], st["ei"]
    harm_cfg = st.get("harm_cfg", {})
    res = _compute_harm_map_with_optional_override_(E, S, harm_cfg)

    HM = _np.asarray(res["harm_matrix"], dtype=_np.float32)   # [K, N]
    sel = _np.asarray(res["selected_channels"], dtype=int)    # [K]
    K, N = HM.shape

    # if N>0:
    #     plot_harm_heatmap(res, field="harm_matrix", title=f"Harm map")

    # --- weighted mean across channels (aligns with lag selection logic) ---
    # Recreate the per-channel weights the harm-map used internally.
    w = _np.asarray(res.get("channel_ptp", _np.ones(sel.size, dtype=_np.float32)), dtype=_np.float32)
    if bool(harm_cfg.get("weight_by_p2p", True)):
        w = w ** float(harm_cfg.get("weight_beta", 0.7))
    else:
        w = _np.ones_like(w)
    ws = float(w.sum())
    if ws > 0:
        w = w / ws

    mean_d = (HM * w[:, None]).sum(axis=0)  # [N] weighted mean ΔRMS

    # --- robust channel gate (configurable); default 1.0 = strict max (old behavior) ---
    q = float(harm_cfg.get("chan_gate_quantile", 1.0))
    if q >= 1.0:
        chan_stat = HM.max(axis=0)                      # old behavior
    else:
        chan_stat = _np.quantile(HM, q, axis=0)         # e.g., q=0.90

    # --- ref-channel check remains the same ---
    ref_ch = int(st["ref_ch"])
    ref_matches = _np.where(sel == ref_ch)[0]
    if ref_matches.size == 0:
        ref_ok = _np.ones(N, dtype=bool)
        ref_missing = True
    else:
        ref_row = int(ref_matches[0])
        ref_ok = HM[ref_row] < float(params.get("REF_THR", -5.0))
        ref_missing = False

    keep = (
        (mean_d   <= float(params.get("MEAN_THR", -2.0))) &
        (chan_stat <= float(params.get("CHAN_THR", 15.0))) &
        ref_ok
    )

    # diagnostics for offenders by max-d channel
    # diagnostics for offenders by channel gate (quantile-aware)
    thr_chan = float(params.get("CHAN_THR", 15.0))
    bad_idx = _np.where(chan_stat > thr_chan)[0]
    ch_of_max = None
    if bad_idx.size > 0:
        ch_of_max = _np.argmax(HM[:, bad_idx], axis=0)


    st2 = _filter_state_(st, keep)
    kept, drop = int(keep.sum()), int((~keep).sum())
    st2["meta"]["last_harm"] = res
    st2["meta"]["history"].append({
        "step":"harm_compliance",
        "kept": kept, "dropped": drop,
        "mean_thr": float(params.get("MEAN_THR", -2.0)),
        "chan_thr": float(params.get("CHAN_THR", 15.0)),
        "ref_thr": float(params.get("REF_THR", -5.0)),
        "ref_missing": bool(ref_missing)
    })
    rep = {
        "name":"harm_compliance",
        "kept": kept, "dropped": drop,
        "n_mean_viol": int((mean_d   > float(params.get("MEAN_THR", -2.0))).sum()),
        "n_chan_viol": int((chan_stat > float(params.get("CHAN_THR", 15.0))).sum()),
        "n_ref_viol":  int((~ref_ok).sum()) if not ref_missing else 0,
        "chan_gate_quantile": q,
        "top_offending_channels": _np.array([] if ch_of_max is None else _np.bincount(ch_of_max).argsort()[::-1], dtype=int)
    }
    return st2, rep

# -------------------------
# PRUNER P4: Bimodality split (amplitude-driven)
# -------------------------
def bimodality_split_pruner(state, *, params):
    """
    Look for bimodality on a selected channel (via check_bimodality_and_plot),
    then keep one cohort according to 'rule':
      - 'larger_n' (default)
      - 'higher_amp_on_ref' (compare EI trough on ref_ch)
    If no hit, keep all.
    params: {'dprime_thr','min_per_cluster','rule'}
    """
    st = _ensure_meta_(state)
    S, E, ei_pos = st["snips"], st["ei"], st["ei_positions"]
    # need a harm result to pass to the checker; reuse last if present, else compute
    harm_res = st["meta"].get("last_harm", None)
    if harm_res is None:
        harm_res = _compute_harm_map_with_optional_override_(E, S, st.get("harm_cfg", {}))
    tmp = check_bimodality_and_plot(
        S, harm_res, ei_pos, int(st["ref_ch"]),
        dprime_thr=float(params.get("dprime_thr", 5.0)),
        min_per_cluster=int(params.get("min_per_cluster", 5))
    )
    if not tmp or not tmp.get("hit", False):
        st["meta"]["history"].append({"step":"bimodality_split", "hit": False, "kept": S.shape[2], "dropped": 0})
        return st, {"name":"bimodality_split", "hit": False, "kept": S.shape[2], "dropped": 0}
    idx_lo = _np.asarray(tmp["idx_lo"], dtype=_np.int64)
    idx_hi = _np.asarray(tmp["idx_hi"], dtype=_np.int64)
    N = S.shape[2]
    # sanitize
    valid_lo = (idx_lo >= 0) & (idx_lo < N)
    valid_hi = (idx_hi >= 0) & (idx_hi < N)
    idx_lo = idx_lo[valid_lo]; idx_hi = idx_hi[valid_hi]
    if idx_lo.size == 0 or idx_hi.size == 0:
        st["meta"]["history"].append({"step":"bimodality_split", "hit": False, "note":"empty cohort after filtering"})
        return st, {"name":"bimodality_split", "hit": False, "kept": N, "dropped": 0, "note":"empty cohort after filtering"}

    rule = str(params.get("rule", "larger_n"))
    if rule == "higher_amp_on_ref":
        ref = int(st["ref_ch"])
        amp_lo = float(-tmp["ei_lo"][ref].min())
        amp_hi = float(-tmp["ei_hi"][ref].min())
        pick_hi = (amp_hi >= amp_lo)
        keep_local = idx_hi if pick_hi else idx_lo
        cohort = "high" if pick_hi else "low"
    else:
        pick_hi = (idx_hi.size >= idx_lo.size)
        keep_local = idx_hi if pick_hi else idx_lo
        cohort = "high" if pick_hi else "low"

    mask = _np.zeros(N, dtype=bool); mask[keep_local] = True
    st2 = _filter_state_(st, mask)
    kept, drop = int(mask.sum()), int((~mask).sum())
    st2["meta"]["history"].append({"step":"bimodality_split", "hit": True, "cohort": cohort, "kept": kept, "dropped": drop})
    rep = {"name":"bimodality_split", "hit": True, "cohort": cohort, "kept": kept, "dropped": drop,
           "sizes": {"low": int(idx_lo.size), "high": int(idx_hi.size)}}
    return st2, rep



# ### === Trace masking utilities (pre-detection) ===
# Mask the current detection channel trace using previously discovered templates.
# Each template entry must provide: 'ei' [C,T_ei], 'spike_times' [N], and 'relevant_ch' (int).
# We mask the segment [t + (i_first - center_index) : t + (i_last - center_index)], where
# i_first/i_last are the first/last indices in the EI waveform on 'target_ch' whose |amp| >= amp_thr.

def make_mask_entry(ei, spike_times, relevant_ch, amp_thr=25.0, center_index=40, meta=None):
    """
    Create a standard template-bank entry for masking.

    Args
    ----
    ei           : np.ndarray [C, T_ei]  (aligned so the main trough is at 'center_index')
    spike_times  : 1D array of absolute sample indices (aligned to center_index on the main channel)
    relevant_ch  : int  -- the channel on which this template should mask when that channel is being detected
    amp_thr      : float (ADC units), absolute amplitude threshold to define the template's active window
    center_index : int (default 40), index in EI corresponding to spike time alignment (same convention as your pipeline)
    meta         : optional dict for tags (unit id, notes, etc.)

    Returns
    -------
    dict with keys: {'ei','spike_times','relevant_ch','amp_thr','center_index','meta'}
    """
    return {
        "ei": _np.asarray(ei, dtype=_np.float32),
        "spike_times": _np.asarray(spike_times, dtype=_np.int64).ravel(),
        "relevant_ch": int(relevant_ch),
        "amp_thr": float(amp_thr),
        "center_index": int(center_index),
        "meta": ({} if meta is None else dict(meta)),
    }


def mask_trace_with_template_bank(trace, target_ch, bank,
                                  *, amp_thr_default=25.0, center_index_default=40,
                                  thr=None, fill_value=None, in_place=False):
    """
    Mask a 1D trace for channel 'target_ch' using all templates in 'bank' whose 'relevant_ch' == target_ch.

    For each matching template entry:
      1) On EI[target_ch], find the first and last index where |EI| >= amp_thr (default 25 ADC).
      2) For each spike time t in entry['spike_times'], compute segment:
            start = t - center_index + i_first
            end   = t - center_index + i_last
         Clip to [0, T-1] and mark as masked.

    Implementation uses a difference-array trick to set all segments in O(T + M) time, where M = #segments.

    Args
    ----
    trace              : 1D np.ndarray (float32/float64)  -- raw samples for the current detection channel
    target_ch          : int  -- the channel we are about to detect on
    bank               : list[dict]  -- template entries from make_mask_entry(...)
    amp_thr_default    : float, fallback absolute amplitude threshold (ADC) if entry has no 'amp_thr'
    center_index_default: int, fallback center index if entry has no 'center_index' (default 40)
    thr                : float or None. If provided and fill_value is None, masked samples are set to max(0, thr + 1)
                         which guarantees they can't cross a negative detection threshold.
    fill_value         : float or None. If None and thr is None, masked samples set to 0.0.
    in_place           : bool. If True, modify input trace; otherwise operate on a copy.

    Returns
    -------
    masked_trace : 1D np.ndarray (same dtype as input, unless float cast required)
    report       : dict with {'target_ch','segments','masked_samples','masked_fraction'}
    """
    if bank is None or len(bank) == 0:
        # nothing to do
        tr_out = trace if in_place else _np.array(trace, copy=True)
        return tr_out, {"target_ch": int(target_ch), "segments": 0, "masked_samples": 0, "masked_fraction": 0.0}

    x = trace if in_place else _np.array(trace, dtype=_np.float32, copy=True)
    T = x.size
    if T == 0:
        return x, {"target_ch": int(target_ch), "segments": 0, "masked_samples": 0, "masked_fraction": 0.0}

    # Collect all [start, end] inclusive segments across matching templates
    starts_all = []
    ends_all = []
    seg_count = 0

    for entry in bank:
        # if int(entry.get("relevant_ch", -1)) != int(target_ch):
        #     continue
        EI = _np.asarray(entry["ei"], dtype=_np.float32)
        C, L = EI.shape
        ch = int(target_ch)
        if ch < 0 or ch >= C:
            # EI doesn't cover this channel; skip
            continue

        amp_thr = float(entry.get("amp_thr", amp_thr_default))
        center_index = int(entry.get("center_index", center_index_default))
        w = EI[ch]  # waveform for the target channel
        idx = _np.flatnonzero(_np.abs(w) >= amp_thr)

        if idx.size == 0:
            # No part of this channel's waveform crosses threshold → nothing to mask for this template on this ch
            continue

        i_first = int(idx[0])
        i_last  = int(idx[-1])
        rel_start = i_first - center_index
        rel_end   = i_last  - center_index

        times = _np.asarray(entry["spike_times"], dtype=_np.int64).ravel()
        if times.size == 0:
            continue

        # Vectorized compute + clipping
        s = times + rel_start
        e = times + rel_end
        # ensure start <= end
        swap = s > e
        if swap.any():
            s, e = e.copy(), s.copy()

        # Clip to [0, T-1]
        s = _np.clip(s, 0, T - 1)
        e = _np.clip(e, 0, T - 1)

        # Drop empty segments (possible if everything clipped to same invalid index)
        valid = (e >= s)
        s = s[valid]; e = e[valid]

        starts_all.append(s.astype(_np.int64))
        ends_all.append(e.astype(_np.int64))
        seg_count += int(s.size)

    if seg_count == 0:
        return x, {"target_ch": int(target_ch), "segments": 0, "masked_samples": 0, "masked_fraction": 0.0}

    # Concatenate all segments; build boolean mask via difference array (fast)
    starts_all = _np.concatenate(starts_all, axis=0)
    ends_all   = _np.concatenate(ends_all, axis=0)

    # Difference array sized T+1 so we can safely do mask[e+1] -= 1
    diff = _np.zeros(T + 1, dtype=_np.int32)
    _np.add.at(diff, starts_all, 1)
    # guard e+1 at T -> write into the sentinel at diff[T]
    ends_plus1 = _np.minimum(ends_all + 1, T)
    _np.add.at(diff, ends_plus1, -1)
    cum = _np.cumsum(diff[:-1])
    mask = cum > 0

    # Choose fill
    if fill_value is None:
        if thr is not None:
            fill = float(max(0.0, float(thr) + 1.0))
        else:
            fill = 0.0
    else:
        fill = float(fill_value)

    x[mask] = fill
    n_mask = int(mask.sum())
    report = {
        "target_ch": int(target_ch),
        "segments": int(seg_count),
        "masked_samples": n_mask,
        "masked_fraction": (n_mask / T) if T > 0 else 0.0
    }
    return x, report


# === Mask-bank I/O ===

def save_mask_bank_pickle(path: str, bank: list):
    """
    Atomically save the mask_bank (list of dicts from make_mask_entry).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=".maskbank_", dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            pickle.dump(bank, f, protocol=4)  # protocol 4+ handles large arrays well
        os.replace(tmp_path, path)  # atomic rename
    except Exception:
        try:
            os.remove(tmp_path)
        finally:
            raise

def load_mask_bank_pickle(path: str) -> list:
    """
    Load a mask_bank previously saved via save_mask_bank_pickle.
    Returns [] if file does not exist.
    """
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        bank = pickle.load(f)
    # minimal shape/type sanity (optional)
    for i, e in enumerate(bank):
        if not isinstance(e, dict):
            raise TypeError(f"mask_bank entry {i} is not a dict")
        _ = e["ei"], e["spike_times"], e["relevant_ch"]  # will KeyError if malformed
    return bank










# === KS vs AX ====

# === KS ↔ AX matching utilities ===
import numpy as _np

def _p2p_per_channel(ei):
    """Return p2p amplitude per channel [C]."""
    return (ei.max(axis=1) - ei.min(axis=1)).astype(_np.float32)

def _zero_pad_shift(ei, lag):
    """
    Shift EI in time by lag (int) with zero padding (no wrap).
    ei: [C,T]
    lag > 0 shifts right (delays); lag < 0 shifts left (advances).
    """
    C, T = ei.shape
    out = _np.zeros_like(ei)
    if lag == 0:
        return ei
    if lag > 0:
        # zeros at start, truncate tail
        out[:, lag:] = ei[:, :T-lag]
    else:
        s = -lag
        out[:, :T-s] = ei[:, s:]
    return out

def _cosine_on_channels(A, B, chan_idx):
    """
    Cosine similarity between EI A and B restricted to chan_idx.
    A,B: [C,T], chan_idx: 1D int array
    """
    if len(chan_idx) == 0:
        return _np.nan
    a = A[chan_idx].reshape(-1).astype(_np.float64)
    b = B[chan_idx].reshape(-1).astype(_np.float64)
    na = _np.linalg.norm(a)
    nb = _np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a.dot(b) / (na * nb))

# --- REPLACE the whole definition of _best_cosines_with_small_lags with this ---
def _best_cosines_with_small_lags(
    ei_ks, ei_ax, main_ch,
    sig_thr=30.0,
    try_lags=(-1, 0, 1),
    early_accept_all=0.95,
    early_accept_excl=0.95,
):
    """
    Cosine similarity on union of significant channels (p2p>sig_thr) and on the same set excluding main_ch.
    Early accept at lag=0 if (cos_all >= early_accept_all) OR (cos_excl >= early_accept_excl).
    Otherwise scan try_lags and return the best pair (maximize min(cos_all, cos_excl); tie: larger sum).
    Always returns a dict (never None).
    """
    p2p_ks = _p2p_per_channel(ei_ks)
    p2p_ax = _p2p_per_channel(ei_ax)

    sig_ks = _np.flatnonzero(p2p_ks > sig_thr)
    sig_ax = _np.flatnonzero(p2p_ax > sig_thr)
    chans_all = _np.unique(_np.concatenate([sig_ks, sig_ax], axis=0))
    if chans_all.size == 0:
        chans_all = _np.array([int(main_ch)], dtype=int)

    if chans_all.size > 1 and int(main_ch) in chans_all:
        chans_excl = chans_all[chans_all != int(main_ch)]
    else:
        chans_excl = _np.array([], dtype=int)

    # Lag 0 first
    cos_all_0 = _cosine_on_channels(ei_ks, ei_ax, chans_all)
    cos_exc_0 = _cosine_on_channels(ei_ks, ei_ax, chans_excl) if chans_excl.size else _np.nan

    # OR-based early accept
    if (
        (_np.isfinite(cos_all_0) and cos_all_0 >= float(early_accept_all)) or
        (_np.isfinite(cos_exc_0) and cos_exc_0 >= float(early_accept_excl))
    ):
        return {
            "best_lag": 0,
            "cos_all": float(cos_all_0),
            "cos_excl_main": float(cos_exc_0) if _np.isfinite(cos_exc_0) else _np.nan,
            "cos_all_lag0": float(cos_all_0),
            "cos_excl_main_lag0": float(cos_exc_0) if _np.isfinite(cos_exc_0) else _np.nan,
            "n_channels_all": int(chans_all.size),
            "n_channels_excl_main": int(chans_excl.size),
        }

    # Initialize "best" from lag 0 so we ALWAYS have a return value
    def score_pair(ca, cx):
        a = -_np.inf if not _np.isfinite(ca) else ca
        b = -_np.inf if not _np.isfinite(cx) else cx
        return (min(a, b), a + b)

    best = {"best_lag": 0, "cos_all": float(cos_all_0), "cos_excl_main": float(cos_exc_0)}
    best_score = score_pair(cos_all_0, cos_exc_0)

    # Scan small lags
    for lag in try_lags:
        if int(lag) == 0:
            continue
        ax_shift = _zero_pad_shift(ei_ax, int(lag))
        ca = _cosine_on_channels(ei_ks, ax_shift, chans_all)
        cx = _cosine_on_channels(ei_ks, ax_shift, chans_excl) if chans_excl.size else _np.nan
        sc = score_pair(ca, cx)
        if sc > best_score:
            best_score = sc
            best = {"best_lag": int(lag), "cos_all": float(ca), "cos_excl_main": float(cx)}

    # Always return a dict with lag-0 values recorded too
    best.update({
        "cos_all_lag0": float(cos_all_0),
        "cos_excl_main_lag0": float(cos_exc_0) if _np.isfinite(cos_exc_0) else _np.nan,
        "n_channels_all": int(chans_all.size),
        "n_channels_excl_main": int(chans_excl.size),
    })
    return best



def precompute_ax_p2p_table(mask_bank, C, nan_floor=50.0):
    """
    Returns:
      p2p_ax : [C, N_ax] p2p table (values < nan_floor set to np.nan),
      eis_ax : list of EIs [C,T] aligned,
      ax_ids : list of unit ids (meta['unit_id'] if present else bank index)
      rel_ch : np.array of relevant_ch per entry
    """
    N = len(mask_bank)
    p2p_ax = _np.full((C, N), _np.nan, dtype=_np.float32)
    eis_ax = []
    ax_ids = []
    rel = _np.empty(N, dtype=int)
    for j, entry in enumerate(mask_bank):
        ei = _np.asarray(entry["ei"], dtype=_np.float32)
        assert ei.shape[0] == C, f"AX EI has C={ei.shape[0]} != expected {C}"
        eis_ax.append(ei)
        rel[j] = int(entry.get("relevant_ch", -1))
        uid = entry.get("meta", {}).get("unit_id", j)
        ax_ids.append(int(uid))
        p = _p2p_per_channel(ei)
        p[p < float(nan_floor)] = _np.nan
        p2p_ax[:, j] = p
    return p2p_ax, eis_ax, ax_ids, rel

def find_ks_ax_matches(
    h5, mask_bank, *,
    amp_ratio_bounds=(0.7, 1.3),
    sig_thr=30.0,
    # NEW: separable accept thresholds (defaults keep old behavior)
    early_accept_all=0.95,
    early_accept_excl=0.95,
    false_reject=0.60,
    plot_limit=20,
    ei_positions=None,
    colors=("k", "r")
):
    """
    Compare KS units in H5 against Ax units in mask_bank.

    Returns dict:
      {
        'ks_to_ax': {ks_uid: [ {ax_uid, ax_idx, status, cos_all, cos_excl, best_lag, n_ch_all, n_ch_exc}, ... ]},
        'ax_to_ks': {ax_uid: [ {ks_uid, ks_idx, status, ...}, ... ]},
        'tested_candidates': int,
        'plotted': int
      }
    """
    import matplotlib.pyplot as plt
    import plot_ei_waveforms as pew

    # KS datasets
    eis_ds = h5["eis"]           # [Nks, C, T]
    unit_ids = h5["unit_ids"][:] # [Nks]
    ref_ch_all = h5["ref_channel"][:] if "ref_channel" in h5 else None
    Nks, C, T = eis_ds.shape

    # Precompute AX p2p table (NaN below 50)
    p2p_ax, eis_ax, ax_ids, rel_ch = precompute_ax_p2p_table(mask_bank, C, nan_floor=50.0)

    ks_to_ax = {}
    ax_to_ks = {}
    plotted = 0
    tested = 0

    lo, hi = float(amp_ratio_bounds[0]), float(amp_ratio_bounds[1])

    for i in range(Nks):
        ks_uid = int(unit_ids[i])
        ei_ks = _np.asarray(eis_ds[i], dtype=_np.float32)  # [C,T]
        p2p_ks = _p2p_per_channel(ei_ks)
        main_ch = int(_np.argmax(p2p_ks)) if ref_ch_all is None else int(ref_ch_all[i])

        amp_ks_main = float(p2p_ks[main_ch])
        if not _np.isfinite(amp_ks_main) or amp_ks_main <= 0:
            continue

        # Candidate AX units: same channel, amplitude within ratio bounds
        # Note: p2p_ax[main_ch, j] may be NaN (if <50), which excludes those candidates automatically.
        amp_ax_on_main = p2p_ax[main_ch]  # [Nax] or NaNs
        with _np.errstate(invalid="ignore", divide="ignore"):
            ratio = amp_ax_on_main / amp_ks_main
        cand = _np.flatnonzero((ratio >= lo) & (ratio <= hi))

        if cand.size == 0:
            ks_to_ax.setdefault(ks_uid, [])
            continue

        for j in cand:
            tested += 1
            ax_uid = int(ax_ids[j])
            ei_ax = eis_ax[j]

            # Compute cosines; early stop if both >= early_accept at lag 0
            metr = _best_cosines_with_small_lags(
                ei_ks, ei_ax, main_ch,
                sig_thr=float(sig_thr),
                try_lags=(-1, 0, 1),
                early_accept_all=float(early_accept_all),
                early_accept_excl=float(early_accept_excl),
            )

            cos_all = float(metr["cos_all"])
            cos_exc = float(metr["cos_excl_main"]) if _np.isfinite(metr["cos_excl_main"]) else _np.nan
            status = None

            # Accept if either metric clears its own threshold
            if (cos_all >= float(early_accept_all)) or (cos_exc >= float(early_accept_excl)):
                status = "match"
            elif (cos_all < float(false_reject)) or (cos_exc < float(false_reject)):
                status = "false"
            else:
                status = "mid"


            # record both directions
            rec = {
                "ax_uid": ax_uid, "ax_idx": int(j),
                "ks_uid": ks_uid, "ks_idx": int(i),
                "status": status,
                "cos_all": cos_all,
                "cos_excl": cos_exc,
                "best_lag": int(metr["best_lag"]),
                "n_ch_all": int(metr["n_channels_all"]),
                "n_ch_exc": int(metr["n_channels_excl_main"]),
            }
            ks_to_ax.setdefault(ks_uid, []).append(rec.copy())
            ax_to_ks.setdefault(ax_uid, []).append(rec.copy())

            # mid-range: plot overlay (up to plot_limit)
            if status == "mid" and plotted < int(plot_limit) and ei_positions is not None:
                plotted += 1
                # use the lag that gave the best score for Ax visualization
                ei_ax_best = _zero_pad_shift(ei_ax, int(metr["best_lag"]))

                fig, ax = plt.subplots(1, 1, figsize=(20,18))
                pew.plot_ei_waveforms([ei_ks, ei_ax_best], ei_positions,
                                      ref_channel=main_ch,
                                      colors=[colors[0], colors[1]],
                                      scale=70.0, box_height=1.0, box_width=50.0, ax=ax)
                ax.set_title(f"KS {ks_uid} vs AX {ax_uid} | cos_all={cos_all:.2f} cos_exc={cos_exc:.2f} | lag={int(metr['best_lag'])} | ch={main_ch}")
                plt.tight_layout(); plt.show()

    return {
        "ks_to_ax": ks_to_ax,
        "ax_to_ks": ax_to_ks,
        "tested_candidates": int(tested),
        "plotted": int(plotted),
    }


# === Best crossmatches (one-to-best with early accept) ===
import numpy as _np

def _prep_ax_bank_for_matching(mask_bank, C, nan_floor=50.0):
    """
    Wrap precompute_ax_p2p_table but also return a numpy array for quick access.
    """
    p2p_ax, eis_ax, ax_ids, rel_ch = precompute_ax_p2p_table(mask_bank, C, nan_floor=nan_floor)
    # keep as list for EIs (variable length T allowed), p2p table as [C, Nax]
    return dict(p2p=p2p_ax, eis=eis_ax, ids=_np.asarray(ax_ids, int), rel=_np.asarray(rel_ch, int))

def _prep_ks_for_matching(h5, nan_floor=50.0):
    """
    Build lightweight views for KS side without loading entire array into RAM at once.
    Returns a dict with:
      - ds: the H5 dataset for EIs (shape [Nks,C,T])
      - ids: np.array of KS unit ids [Nks]
      - main: np.array of main channel per unit [Nks] (from ref_channel if present, else argmax p2p)
      - p2p: [C, Nks] p2p table with values < nan_floor set to NaN
    """
    eis_ds = h5["eis"]            # [Nks,C,T]
    unit_ids = _np.asarray(h5["unit_ids"][:], dtype=int)
    Nks, C, T = eis_ds.shape

    if "ref_channel" in h5:
        main = _np.asarray(h5["ref_channel"][:], dtype=int)
    else:
        # compute argmax p2p lazily over units
        main_list = []
        for i in range(Nks):
            p2p = _p2p_per_channel(_np.asarray(eis_ds[i], dtype=_np.float32))
            main_list.append(int(_np.argmax(p2p)))
        main = _np.asarray(main_list, dtype=int)

    # p2p table for KS (for amplitude prefiltering when KS is the *target* in ax_to_ks)
    p2p = _np.full((C, Nks), _np.nan, dtype=_np.float32)
    for i in range(Nks):
        p = _p2p_per_channel(_np.asarray(eis_ds[i], dtype=_np.float32))
        p[p < float(nan_floor)] = _np.nan
        p2p[:, i] = p

    return dict(ds=eis_ds, ids=unit_ids, main=main, p2p=p2p)

def best_crossmatches(
    h5, mask_bank, *,
    direction="ks_to_ax",           # or "ax_to_ks"
    amp_ratio_bounds=(0.7, 1.3),
    sig_thr=30.0,
    accept_all=0.90,                # union-of-sig-channels cosine threshold
    accept_excl=0.85,               # excluding-main-channel cosine threshold
    try_lags=(-1, 0, 1),
    nan_floor=50.0
):
    """
    For each source unit (KS or AX), return its best target (AX or KS) by sum of two cosines
    after amplitude+channel prefiltering, with early accept if both cosines pass thresholds.

    Returns:
      {
        'direction': 'ks_to_ax' or 'ax_to_ks',
        'mapping': {
            src_uid: {
               'uid_src': int,
               'uid_tgt': int or None,
               'idx_tgt': int (index into target set) or -1,
               'main_ch_src': int,
               'cos_all': float (or np.nan),
               'cos_excl': float (or np.nan),
               'best_lag': int (or np.nan),
               'accepted': bool,              # met both thresholds
               'reason': 'matched' | 'best_below_accept' | 'no_candidates',
               'n_candidates': int
            }, ...
        },
        'params': {...}   # echoes thresholds and settings
      }
    """
    eis_ks = h5["eis"]
    Nks, C, T = eis_ks.shape
    ks = _prep_ks_for_matching(h5, nan_floor=nan_floor)
    ax = _prep_ax_bank_for_matching(mask_bank, C, nan_floor=nan_floor)

    lo, hi = float(amp_ratio_bounds[0]), float(amp_ratio_bounds[1])

    # Define views depending on direction
    if direction == "ks_to_ax":
        # source = KS, target = AX
        N_src = ks["ids"].size
        def get_src(i):
            return dict(uid=int(ks["ids"][i]),
                        ei=_np.asarray(ks["ds"][i], dtype=_np.float32),
                        main=int(ks["main"][i]),
                        amp_main=float(_p2p_per_channel(_np.asarray(ks["ds"][i], dtype=_np.float32))[int(ks["main"][i])]))
        tgt_ids  = ax["ids"]
        tgt_eis  = ax["eis"]
        tgt_p2p  = ax["p2p"]            # [C, Nax]
        # for candidate indexing stability
        def n_tgt(): return tgt_ids.size

    elif direction == "ax_to_ks":
        # source = AX, target = KS
        N_src = len(mask_bank)
        def get_src(i):
            ei = _np.asarray(mask_bank[i]["ei"], dtype=_np.float32)
            # main channel for AX source: by p2p argmax (recentered already in your pipeline)
            p2p = _p2p_per_channel(ei)
            main = int(_np.argmax(p2p))
            uid = int(mask_bank[i].get("meta", {}).get("unit_id", i))
            return dict(uid=uid, ei=ei, main=main, amp_main=float(p2p[main]))
        tgt_ids  = ks["ids"]
        tgt_eis  = None                 # we'll read from H5 by index
        tgt_p2p  = ks["p2p"]            # [C, Nks]
        def n_tgt(): return tgt_ids.size
    else:
        raise ValueError("direction must be 'ks_to_ax' or 'ax_to_ks'")

    mapping = {}

    for i in range(N_src):
        src = get_src(i)
        uid_src   = src["uid"]
        ei_src    = src["ei"]
        main_src  = src["main"]
        amp_src_m = src["amp_main"]

        # amplitude+channel prefilter on target set, using the SOURCE main channel
        # Only targets with finite p2p on main_src and ratio within [lo, hi]
        amp_tgt_on_main = tgt_p2p[main_src]   # shape [N_tgt], NaNs where below nan_floor
        with _np.errstate(invalid="ignore", divide="ignore"):
            ratio = amp_tgt_on_main / amp_src_m
        cand = _np.flatnonzero((ratio >= lo) & (ratio <= hi))

        if cand.size == 0:
            mapping[uid_src] = {
                "uid_src": uid_src, "uid_tgt": None, "idx_tgt": -1,
                "main_ch_src": int(main_src),
                "cos_all": _np.nan, "cos_excl": _np.nan,
                "best_lag": _np.nan,
                "accepted": False, "reason": "no_candidates",
                "n_candidates": 0,
            }
            continue

        # Scan candidates; early accept if both metrics >= thresholds
        best_sum = -_np.inf
        best_rec = None

        for j in cand:
            if direction == "ks_to_ax":
                ei_tgt = tgt_eis[int(j)]
            else:  # ax_to_ks
                ei_tgt = _np.asarray(h5["eis"][int(j)], dtype=_np.float32)

            # Get best over small lags; disable helper's early-accept by passing huge thresholds
            metr = _best_cosines_with_small_lags(
                ei_src, ei_tgt, main_src,
                sig_thr=float(sig_thr),
                try_lags=tuple(int(x) for x in try_lags),
                early_accept_all=1e9,
                early_accept_excl=1e9,
            )
            cos_all = float(metr["cos_all"])
            cos_exc = float(metr["cos_excl_main"]) if _np.isfinite(metr["cos_excl_main"]) else _np.nan
            lag     = int(metr["best_lag"])

            # Accept if BOTH exceed thresholds (AND)
            if (cos_all >= float(accept_all)) and (cos_exc >= float(accept_excl)):
                uid_tgt = int(tgt_ids[int(j)])
                mapping[uid_src] = {
                    "uid_src": uid_src, "uid_tgt": uid_tgt, "idx_tgt": int(j),
                    "main_ch_src": int(main_src),
                    "cos_all": cos_all, "cos_excl": cos_exc,
                    "best_lag": lag, "accepted": True,
                    "reason": "matched", "n_candidates": int(cand.size),
                }
                break

            # Otherwise track the best by sum of cosines (treat NaN as -inf so it never wins)
            s = (cos_all if _np.isfinite(cos_all) else -1e9) + (cos_exc if _np.isfinite(cos_exc) else -1e9)
            if s > best_sum:
                best_sum = s
                best_rec = {
                    "uid_src": uid_src, "uid_tgt": int(tgt_ids[int(j)]), "idx_tgt": int(j),
                    "main_ch_src": int(main_src),
                    "cos_all": cos_all, "cos_excl": cos_exc,
                    "best_lag": lag, "accepted": False,
                    "reason": "best_below_accept", "n_candidates": int(cand.size),
                }
        else:
            # loop exhausted without early accept
            if best_rec is not None:
                mapping[uid_src] = best_rec

        # Safety: if we broke early (matched), nothing to do; if best_rec stayed None (shouldn't happen), mark no_candidates
        if uid_src not in mapping:
            mapping[uid_src] = {
                "uid_src": uid_src, "uid_tgt": None, "idx_tgt": -1,
                "main_ch_src": int(main_src),
                "cos_all": _np.nan, "cos_excl": _np.nan,
                "best_lag": _np.nan,
                "accepted": False, "reason": "no_candidates",
                "n_candidates": int(cand.size),
            }

    return {
        "direction": direction,
        "mapping": mapping,
        "params": {
            "amp_ratio_bounds": tuple(amp_ratio_bounds),
            "sig_thr": float(sig_thr),
            "accept_all": float(accept_all),
            "accept_excl": float(accept_excl),
            "try_lags": tuple(int(x) for x in try_lags),
            "nan_floor": float(nan_floor),
        }
    }





### === BEGIN KS per-event assignment (union harm-map) ===

# --- deps already available in this repo ---
import numpy as _np
from typing import Dict, Any, Tuple, List
from collision_utils import (
    select_template_channels,
    compute_harm_map_noamp,
)

# ----------------------------------------------------------
# Helpers: recenter EI to trough on a SPECIFIC channel
# (your existing recenter_ei_to_ref_trough uses strongest ch)
# ----------------------------------------------------------
def recenter_ei_to_channel_trough(ei: _np.ndarray, ch_ref: int, center_index: int = 40) -> _np.ndarray:
    """Zero-padded shift so that the trough on ch_ref sits at center_index."""
    ei = _np.asarray(ei, dtype=_np.float32)
    C, T = ei.shape
    ch_ref = int(ch_ref)
    trough_idx = int(_np.argmin(ei[ch_ref]))
    shift = int(center_index - trough_idx)
    if shift == 0:
        return ei.copy()
    out = _np.zeros_like(ei)
    if shift > 0:
        out[:, shift:] = ei[:, :T-shift]
    else:
        s = -shift
        out[:, :T-s] = ei[:, s:]
    return out

def _amp_on_channel(ei: _np.ndarray, ch: int) -> float:
    """Positive magnitude of negative trough on channel ch."""
    v = _np.asarray(ei[ch], dtype=_np.float32)
    return float(-v.min())

# ----------------------------------------------------------
# Build candidate set by ref-channel amplitude
# ----------------------------------------------------------
def _select_candidates_by_ref_amp(EIs: _np.ndarray, u0: int, ref_ch: int,
                                  amp_ratio_bounds=(0.5, 1.5), min_abs_amp=5.0) -> _np.ndarray:
    """
    EIs: [C,T,U]; choose candidates whose amp on ref_ch is within bounds *times* amp of u0.
    Returns 1D array of candidate unit indices (including u0), sorted by |ratio-1|.
    """
    C, T, U = EIs.shape
    a0 = _amp_on_channel(EIs[:, :, u0], ref_ch)
    lo, hi = float(amp_ratio_bounds[0]) * a0, float(amp_ratio_bounds[1]) * a0
    amps = _np.array([_amp_on_channel(EIs[:, :, u], ref_ch) for u in range(U)], dtype=_np.float32)
    keep = (amps >= max(1e-6, min_abs_amp)) & (amps >= lo) & (amps <= hi)
    idx = _np.flatnonzero(keep)
    # sort by closeness of ratio to 1.0
    ratio = amps[idx] / (a0 + 1e-12)
    order = _np.argsort(_np.abs(ratio - 1.0))
    return idx[order].astype(int)

# ----------------------------------------------------------
# Union channels across candidates; cap by max_union
# ----------------------------------------------------------
def _union_channels(EIs_centered: _np.ndarray, cand_uids: _np.ndarray, p2p_thr=25.0,
                    min_n=10, max_n=30, ref_ch: int = None) -> _np.ndarray:
    """
    EIs_centered: [C,T,U] after recentering to ref_ch trough at center_index.
    For each candidate, pick channels with your selector, then take union.
    If union > max_n, keep top by max p2p across candidates. Always include ref_ch.
    """
    C, T, U = EIs_centered.shape
    picked = []
    p2p_max = _np.zeros(C, dtype=_np.float32)
    for u in cand_uids:
        ei = EIs_centered[:, :, int(u)]
        chans, ptp = select_template_channels(
            ei, p2p_thr=float(p2p_thr), max_n=int(max_n), min_n=int(min_n), force_include_main=True
        )
        picked.extend(list(chans))
        # track strength per channel for later capping
        p2p_u = (ei.max(axis=1) - ei.min(axis=1)).astype(_np.float32)
        p2p_max = _np.maximum(p2p_max, p2p_u)
    uni = _np.unique(_np.array(picked, dtype=int))
    if ref_ch is not None and ref_ch not in uni:
        uni = _np.concatenate([uni, _np.array([int(ref_ch)], dtype=int)])
        uni = _np.unique(uni)
    if uni.size > int(max_n):
        # keep the strongest by p2p_max
        order = _np.argsort(-p2p_max[uni])
        uni = uni[order[:int(max_n)]]
        # ensure ref_ch survives
        if ref_ch is not None and ref_ch not in uni:
            uni[-1] = int(ref_ch)
    return _np.sort(uni)

# ----------------------------------------------------------
# Harm scores per template on a fixed union of channels
# ----------------------------------------------------------
def _harm_scores_for_template(ei_sel: _np.ndarray, snips_sel: _np.ndarray, ref_ch: int,
                              lag_radius=0, weight_by_p2p=True, weight_beta=0.7
                              ) -> Dict[str, _np.ndarray]:
    """
    ei_sel:   [K,L] template EI on union channels
    snips_sel:[K,L,N] snippets on the same union channels
    Returns dict with:
      'mean' [N], 'max' [N], 'ref' [N] (NaN if ref not in rows), 'rows' [K] channel ids (global)
      and the raw harm_matrix for debugging.
    """
    K, L = ei_sel.shape
    K2, L2, N = snips_sel.shape
    # defensive crop on L
    Lc = min(L, L2)
    ei_use = ei_sel[:, :Lc]
    S_use  = snips_sel[:, :Lc, :]
    # lock channels by setting min=max=K and a very low p2p_thr
    res = compute_harm_map_noamp(
        ei=ei_use, snips=S_use,
        p2p_thr=-1e9, max_channels=K, min_channels=K,
        lag_radius=int(lag_radius),
        weight_by_p2p=bool(weight_by_p2p), weight_beta=float(weight_beta),
        force_include_main=True
    )

    plot_harm_heatmap(res, field="harm_matrix")

    HM = _np.asarray(res["harm_matrix"], dtype=_np.float32)  # [K, N]
    # weights: p2p per row from the EI itself
    p2p = (ei_use.max(axis=1) - ei_use.min(axis=1)).astype(_np.float32)
    w = p2p / (float(p2p.sum()) + 1e-12)
    mean = (w[:, None] * HM).sum(axis=0)                     # [N]
    mx = HM.max(axis=0)
    # ref row if available
    rows = _np.asarray(res.get("selected_channels", _np.arange(K, dtype=int)), dtype=int)
    if ref_ch in rows:
        ref_row = int(_np.flatnonzero(rows == int(ref_ch))[0])
        refv = HM[ref_row]
    else:
        refv = _np.full(N, _np.nan, dtype=_np.float32)
    return {"mean": mean, "max": mx, "ref": refv, "rows": rows, "harm_matrix": HM}

# ----------------------------------------------------------
# Public: assign events to KS templates (union harm-map)
# ----------------------------------------------------------
def assign_events_to_ks_templates(
    EIs: _np.ndarray,          # [C,T,U]
    unit_id: int,              # desired KS unit index (0..U-1)
    snips: _np.ndarray,        # [C,T,N]
    *,
    center_index: int = 40,
    amp_ratio_bounds=(0.5, 1.5),
    min_abs_amp: float = 5.0,
    per_template_p2p_thr: float = 25.0,
    union_min_channels: int = 10,
    union_max_channels: int = 30,
    harm_MEAN_THR: float = -2.0,
    harm_CHAN_THR: float = 15.0,
    harm_REF_THR:  float = -5.0,
    lag_radius: int = 0,
    ambiguity_margin: float = 0.5,
    weight_by_p2p: bool = True,
    weight_beta: float = 0.7,
) -> Dict[str, Any]:
    """
    Returns:
      {
        'ref_ch': int,
        'candidates': np.ndarray[U_sel],            # candidate unit ids
        'union_channels': np.ndarray[K],
        'scores_mean': np.ndarray[N, U_sel],
        'scores_max':  np.ndarray[N, U_sel],
        'scores_ref':  np.ndarray[N, U_sel],        # NaN if ref row missing
        'accepted':    np.ndarray[N, U_sel],        # bool
        'best_uid':    np.ndarray[N]                # unit id or -1 (unknown) or -2 (ambiguous)
      }
    """
    EIs = _np.asarray(EIs, dtype=_np.float32)
    snips = _np.asarray(snips, dtype=_np.float32)
    C, T, U = EIs.shape
    C2, T2, N = snips.shape
    if C2 != C:
        raise ValueError(f"snips C={C2} != EIs C={C}")
    # 1) ref channel from desired unit
    ei0 = EIs[:, :, int(unit_id)]
    ref_ch = int(_np.argmin(ei0.min(axis=1)))
    print(f"Ref channel: {ref_ch}")

    # 2) candidate set by amplitude on ref_ch
    cand = _select_candidates_by_ref_amp(
        EIs, int(unit_id), ref_ch, amp_ratio_bounds=amp_ratio_bounds, min_abs_amp=float(min_abs_amp)
    )
    if cand.size == 0:
        return {
            "ref_ch": ref_ch, "candidates": _np.array([], dtype=int),
            "union_channels": _np.array([], dtype=int),
            "scores_mean": _np.empty((N, 0), _np.float32),
            "scores_max": _np.empty((N, 0), _np.float32),
            "scores_ref": _np.empty((N, 0), _np.float32),
            "accepted": _np.empty((N, 0), bool),
            "best_uid": _np.full(N, -1, dtype=int),
        }
    print(f"{len(cand)} Candidates: {cand}")

    # 3) recenter ALL candidates to trough on ref_ch at center_index
    EIs_ref = _np.empty_like(EIs)
    for u in cand:
        EIs_ref[:, :, int(u)] = recenter_ei_to_channel_trough(EIs[:, :, int(u)], ref_ch, center_index=center_index)
    # 4) union channels across candidates (cap)
    union = _union_channels(
        EIs_ref, cand, p2p_thr=float(per_template_p2p_thr),
        min_n=int(union_min_channels), max_n=int(union_max_channels), ref_ch=ref_ch
    )

    print(f"{len(union)} Channels: {union}")

    # Subset snips to union channels; crop time if needed
    Lc = min(T, T2)
    S_union = snips[union, :Lc, :]  # [K,Lc,N]
    # 5) score each candidate
    Usel = cand.size
    scores_mean = _np.zeros((N, Usel), _np.float32)
    scores_max  = _np.zeros((N, Usel), _np.float32)
    scores_ref  = _np.full((N, Usel), _np.nan, _np.float32)
    accepted    = _np.zeros((N, Usel), bool)
    harm_all    = []
    for j, u in enumerate(cand):
        EI_u = EIs_ref[:, :Lc, int(u)]
        EI_u_sel = EI_u[union]            # [K,Lc]
        sc = _harm_scores_for_template(
            EI_u_sel, S_union, ref_ch=ref_ch, lag_radius=lag_radius,
            weight_by_p2p=weight_by_p2p, weight_beta=weight_beta
        )
        harm_all.append(np.asarray(sc["harm_matrix"], dtype=np.float32))
        scores_mean[:, j] = sc["mean"]
        scores_max[:, j]  = sc["max"]
        scores_ref[:, j]  = sc["ref"]
        ref_ok = _np.ones(N, bool) if _np.isnan(sc["ref"]).all() else (sc["ref"] < float(harm_REF_THR))
        accepted[:, j] = (sc["mean"] <= float(harm_MEAN_THR)) & (sc["max"] <= float(harm_CHAN_THR)) & ref_ok
    # 6) choose best / mark ambiguous / unknown
    best_uid = _np.full(N, -1, dtype=int)
    if Usel > 0:
        # set non-accepted to +inf so argmin ignores them
        tmp = _np.where(accepted, scores_mean, _np.inf)
        best_idx = _np.argmin(tmp, axis=1)           # [N]
        best_val = tmp[_np.arange(N), best_idx]
        has = _np.isfinite(best_val)
        best_uid[has] = cand[best_idx[has]]
        # ambiguity: check 2nd-best among accepted, but avoid inf-inf → NaN
        if Usel > 1:
            sort_idx = _np.argsort(tmp, axis=1)                  # [N, Usel]
            second_val = _np.take_along_axis(
                tmp, sort_idx[:, 1:2], axis=1
            ).ravel()                                           # [N]

            # Only compute difference where both best and second are finite
            valid = has & _np.isfinite(second_val)
            diff = _np.zeros_like(best_val, dtype=_np.float32)
            diff[valid] = second_val[valid] - best_val[valid]

            ambiguous = valid & (diff <= float(ambiguity_margin))
            best_uid[ambiguous] = -2  # mark ambiguous

    return {
        "ref_ch": ref_ch,
        "candidates": cand,
        "union_channels": union,
        "scores_mean": scores_mean,
        "scores_max":  scores_max,
        "scores_ref":  scores_ref,
        "accepted":    accepted,
        "best_uid":    best_uid,
        "harm_matrices": harm_all,
        "aligned_EIs": EIs_ref,
    }


### === END KS per-event assignment (union harm-map) ===




### === BEGIN KS pruning + discriminative panel + assignment ===
# Minimal, self-contained KS→event assignment:
# A) prune templates by snippet evidence
# B) pick discriminative channels by pairwise EI differences
# C) assign spikes by LS scalar on the discriminative panel
#
# Python 3.9.21 compatible. Dependencies: numpy, matplotlib, (optional) plot_ei_waveforms as pew.

import numpy as _np
import matplotlib.pyplot as _plt

try:
    import plot_ei_waveforms as _pew  # optional; used only for EI plots if present
except Exception:
    _pew = None


# ---------- Helpers ----------

def _main_trough_channel(ei_2d: _np.ndarray) -> int:
    """Strongest negative trough channel of a single EI [C,T]."""
    # per-channel minimum over time; pick the most negative
    mins = ei_2d.min(axis=1)
    return int(_np.argmin(mins))

def recenter_ei_to_specific_channel_trough(ei: _np.ndarray, ch_ref: int, center_index: int = 40) -> _np.ndarray:
    """Zero-pad shift so trough on ch_ref lands at center_index."""
    ei = _np.asarray(ei, dtype=_np.float32)
    C, T = ei.shape
    trough_idx = int(_np.argmin(ei[int(ch_ref)]))
    s = int(center_index - trough_idx)
    if s == 0:
        return ei.copy()
    out = _np.zeros_like(ei)
    if s > 0:
        out[:, s:] = ei[:, :T - s]
    else:
        s2 = -s
        out[:, :T - s2] = ei[:, s2:]
    return out

def _ptp_per_channel(ei_or_snips: _np.ndarray) -> _np.ndarray:
    """Peak-to-peak across time per channel. Accepts [C,T] or [C,T,N] (returns [C] or [C,N])."""
    if ei_or_snips.ndim == 2:
        return (ei_or_snips.max(axis=1) - ei_or_snips.min(axis=1)).astype(_np.float32)
    elif ei_or_snips.ndim == 3:
        return (ei_or_snips.max(axis=1) - ei_or_snips.min(axis=1)).astype(_np.float32)
    else:
        raise ValueError("Expected 2D or 3D array")


# ---------- Step A: Evidence over snippets & template pruning ----------

def compute_channel_evidence(snips: _np.ndarray, *, mode: str = "ptp", q: float = 95.0) -> _np.ndarray:
    """
    snips: [C,T,N]. Evidence E[j] = high percentile over spikes of a per-spike magnitude summary on channel j.
    - mode='ptp': per-spike peak-to-peak across time, then percentile across spikes (robust and intuitive).
    - mode='l2' : per-spike L2 across time, then percentile across spikes.
    Returns E: [C]
    """
    snips = _np.asarray(snips, dtype=_np.float32)
    C, T, N = snips.shape
    if N == 0:
        return _np.zeros(C, _np.float32)

    if mode == "ptp":
        M = (snips.max(axis=1) - snips.min(axis=1))  # [C,N]
    elif mode == "l2":
        M = _np.sqrt(_np.sum(snips * snips, axis=1))  # [C,N]
    else:
        raise ValueError("mode must be 'ptp' or 'l2'")

    E = _np.percentile(M, q, axis=1).astype(_np.float32)  # [C]
    return E

def prune_templates_by_channel_evidence(
    EIs: _np.ndarray,             # [C,T,U] (not yet recentered)
    unit_id: int,                 # focal KS unit
    snips: _np.ndarray,           # [C,T,N]
    *,
    center_index: int = 40,
    amp_ratio_bounds=(0.5, 1.5),  # candidate set by ref-channel trough vs focal
    min_abs_amp: float = 5.0,
    evidence_mode: str = "ptp",
    evidence_q: float = 95.0,
    rho_commit: float = 0.25,     # how "big" a channel must be (rel. to ref) to be required by a template
    tau_evidence: float = 0.40,   # min fraction of template magnitude evidence must show on required channels
    plot: bool = True,
    ei_positions: _np.ndarray = None,
) -> dict:
    """
    Returns dict:
      {
        'ref_ch': int,
        'candidates_amp_gated': np.ndarray[U1],
        'candidates_pruned':    np.ndarray[U2],   # after evidence pruning
        'E': np.ndarray[C],                       # evidence per channel
        'EIs_ref': np.ndarray[C,T,U1],            # recentered to ref_ch for candidate set
        'commit_masks': List[np.ndarray],         # per-candidate channels in commitment mask
      }
    """
    EIs = _np.asarray(EIs, _np.float32)
    snips = _np.asarray(snips, _np.float32)
    C, T, U = EIs.shape

    # Focal unit & ref channel
    ei0 = EIs[:, :, int(unit_id)]
    ref_ch = _main_trough_channel(ei0)

    # Amplitude-gated candidate set by ref-ch trough
    def _amp_on_ref(ei2d):
        return float(-ei2d[int(ref_ch)].min())
    a0 = _amp_on_ref(ei0)
    lo, hi = a0 * float(amp_ratio_bounds[0]), a0 * float(amp_ratio_bounds[1])

    amps_all = _np.array([_amp_on_ref(EIs[:, :, u]) for u in range(U)], _np.float32)
    mask_amp = (amps_all >= max(min_abs_amp, 1e-6)) & (amps_all >= lo) & (amps_all <= hi)
    cand_idx = _np.flatnonzero(mask_amp)
    # Recenter candidates so trough on ref_ch is at center_index
    EIs_ref = _np.zeros((C, T, cand_idx.size), _np.float32)
    for j, u in enumerate(cand_idx):
        EIs_ref[:, :, j] = recenter_ei_to_specific_channel_trough(EIs[:, :, int(u)], ref_ch, center_index=center_index)

    # Evidence over the snippet batch
    E = compute_channel_evidence(snips, mode=evidence_mode, q=float(evidence_q))  # [C]

    # Commitment mask per template & evidence check
    cand_keep = []
    commit_masks = []
    for j, u in enumerate(cand_idx):
        ei = EIs_ref[:, :, j]
        # Required channels for this template (relative to ref channel)
        ref_norm = _np.linalg.norm(ei[int(ref_ch)])
        ch_norms = _np.linalg.norm(ei, axis=1)
        req = _np.flatnonzero(ch_norms >= float(rho_commit) * max(1e-6, ref_norm))
        commit_masks.append(req)
        # Feasibility: our snippet set must show enough on any such channel (global evidence)
        ok = True
        thr = float(tau_evidence)
        for ch in req:
            if E[int(ch)] < thr * ch_norms[int(ch)]:
                ok = False
                break
        if ok:
            cand_keep.append(u)
    cand_keep = _np.asarray(cand_keep, dtype=int)

    # ---- Diagnostics plots ----
    if plot:
        # 1) Evidence overview
        _plt.figure(figsize=(12, 3))
        _plt.plot(E, lw=0.8)
        _plt.title("Channel evidence (95th percentile per channel)")
        _plt.xlabel("Channel"); _plt.ylabel("Evidence"); _plt.tight_layout()

        # 2) For up to 6 templates: commitment vs evidence on those channels
        n_show = min(6, cand_idx.size)
        for k in range(n_show):
            j = k
            ei = EIs_ref[:, :, j]
            ch_norms = _np.linalg.norm(ei, axis=1)
            ref_norm = _np.linalg.norm(ei[int(ref_ch)])
            req = commit_masks[j]
            if req.size == 0:
                continue
            _plt.figure(figsize=(6, 3))
            _plt.bar(_np.arange(req.size) - 0.2, ch_norms[req], width=0.4, label="|template|")
            _plt.bar(_np.arange(req.size) + 0.2, E[req], width=0.4, label="evidence")
            _plt.xticks(_np.arange(req.size), req, fontsize=6, rotation=90)
            _plt.axhline(float(rho_commit) * max(1e-6, ref_norm), color='k', ls='--', lw=0.8, alpha=0.6)
            _plt.title(f"Commitment vs evidence · cand {int(cand_idx[j])} · ref_ch {ref_ch}")
            _plt.legend(); _plt.tight_layout()

        # 3) Optional EI waveforms for the focal + some candidates
        if _pew is not None and ei_positions is not None:
            ei_list = [recenter_ei_to_specific_channel_trough(ei0, ref_ch)]
            labels = [f"focal {int(unit_id)}"]
            for u in cand_keep[:3]:
                ei_list.append(recenter_ei_to_specific_channel_trough(EIs[:, :, int(u)], ref_ch))
                labels.append(f"cand {int(u)}")
            _plt.figure(figsize=(20, 12))
            _pew.plot_ei_waveforms(
                ei_list, ei_positions, ref_channel=int(ref_ch),
                scale=70.0, box_height=1.0, box_width=50.0, colors=['black', 'C1', 'C2', 'C3']
            )
            _plt.title("Focal + sample candidates (aligned at ref_ch)")
            _plt.tight_layout()

    return {
        "ref_ch": ref_ch,
        "candidates_amp_gated": cand_idx,
        "candidates_pruned": cand_keep,
        "E": E,
        "EIs_ref": EIs_ref,            # still includes all amp-gated (cand_idx); pruned list in cand_keep
        "commit_masks": commit_masks,
    }


# ---------- Step B: discriminative panel by pairwise template differences ----------

def select_discriminative_panel(
    EIs_ref: _np.ndarray,     # [C,T,U1] already aligned to ref_ch
    cand_ids_all: _np.ndarray,# original candidate idx (U1)
    cand_ids_keep: _np.ndarray,# pruned subset (U2)
    *,
    panel_cap: int = 24,
    per_pair_top_m: int = 4,
    include_ref_ch: int = None,
    plot: bool = True
) -> dict:
    """
    Compute per-channel discriminativeness: sum over candidate pairs of ||tk - tl||_2 on that channel.
    Keep top 'panel_cap' channels. Also collect a small witness set (top-m channels) per pair.
    Returns dict with 'panel_channels', 'diff_sum_per_ch', 'pair_witnesses'.
    """
    # Restrict to pruned candidates
    if cand_ids_keep.size == 0:
        return {"panel_channels": _np.array([], dtype=int), "diff_sum_per_ch": None, "pair_witnesses": {}}

    # Map kept -> local indices in EIs_ref
    keep_locs = _np.array([_np.flatnonzero(cand_ids_all == u)[0] for u in cand_ids_keep], dtype=int)
    C, T, _U1 = EIs_ref.shape
    U2 = keep_locs.size

    # Pairwise per-channel L2 diffs
    diff_sum = _np.zeros(C, _np.float32)
    pair_witnesses = {}  # (u_i, u_j) -> top_m channels

    for a in range(U2):
        for b in range(a + 1, U2):
            i, j = int(keep_locs[a]), int(keep_locs[b])
            ei_a = EIs_ref[:, :, i]   # [C,T]
            ei_b = EIs_ref[:, :, j]
            d = _np.linalg.norm(ei_a - ei_b, axis=1)  # [C]
            diff_sum += d
            # top-m channels for this pair
            top = _np.argsort(-d)[:int(per_pair_top_m)]
            pair_witnesses[(int(cand_ids_keep[a]), int(cand_ids_keep[b]))] = top.astype(int)

    # Pick top channels globally
    order = _np.argsort(-diff_sum)
    panel = order[:int(panel_cap)].astype(int)
    if include_ref_ch is not None and include_ref_ch not in panel:
        if panel.size > 0:
            panel[-1] = int(include_ref_ch)
        else:
            panel = _np.array([int(include_ref_ch)], dtype=int)
    panel = _np.unique(panel)

    # ---- Diagnostics plots ----
    if plot:
        _plt.figure(figsize=(12, 3))
        _plt.plot(diff_sum, lw=0.8)
        if include_ref_ch is not None:
            _plt.axvline(int(include_ref_ch), color='k', ls='--', lw=0.8, alpha=0.6)
        _plt.scatter(panel, diff_sum[panel], s=20)
        _plt.title(f"Discriminative score per channel (sum of pairwise EI diffs); selected K={panel.size}")
        _plt.xlabel("Channel"); _plt.ylabel("Σ||Δ EI||"); _plt.tight_layout()

        # Heatmap for top ~64 channels
        k = min(64, C)
        top64 = _np.argsort(-diff_sum)[:k]
        H = diff_sum[top64][None, :]
        _plt.figure(figsize=(12, 1.6))
        _plt.imshow(H, aspect='auto', cmap='viridis')
        _plt.yticks([]); _plt.xticks(range(k), top64, fontsize=6, rotation=90)
        _plt.title("Top channels by discriminativeness (heat strip)")
        _plt.tight_layout()

    return {"panel_channels": panel, "diff_sum_per_ch": diff_sum, "pair_witnesses": pair_witnesses}


# ---------- Step C: assign spikes on the panel with a 1D scalar fit ----------

def assign_spikes_on_panel_via_ls(
    EIs_ref: _np.ndarray,         # [C,T,U1] aligned to ref_ch; candidates in 'cand_ids_keep' (subset)
    cand_ids_all: _np.ndarray,    # U1 (amp-gated)
    cand_ids_keep: _np.ndarray,   # U2 (pruned)
    snips: _np.ndarray,           # [C,T,N]
    panel_channels: _np.ndarray,  # [J]
    *,
    batch_size: int = 10000,
    alpha_max: float = 3.0,
    improve_frac_min: float = 0.15,       # require ≥15% reduction vs null to accept any template
    ambiguity_margin_frac: float = 0.05,  # if (res2 - res1)/res1 <= 5% -> ambiguous
    plot: bool = True,
    plot_sample_scatter: int = 5000
) -> dict:
    """
    For each spike and each kept candidate, fit scalar α >= 0 on panel channels (LS closed-form),
    compute residual SSE, and decide: best, ambiguous, or unknown.
    Returns dict with residual matrices, alphas, labels, and summary counts.
    """
    snips = _np.asarray(snips, _np.float32)
    C, T, N = snips.shape
    J = int(panel_channels.size)
    if J == 0 or cand_ids_keep.size == 0 or N == 0:
        return {
            "labels": _np.full(N, -1, int),
            "residuals": _np.zeros((N, 0), _np.float32),
            "alphas": _np.zeros((N, 0), _np.float32),
            "best_idx": _np.full(N, -1, int),
            "best_unit": _np.full(N, -1, int),
            "improve_frac": _np.zeros(N, _np.float32),
            "counts": {"unknown": N, "ambiguous": 0, "assigned": 0}
        }

    # map kept -> local indices in EIs_ref
    keep_locs = _np.array([_np.flatnonzero(cand_ids_all == u)[0] for u in cand_ids_keep], dtype=int)
    U2 = keep_locs.size

    # Flatten templates on panel; precompute ||T||^2 per candidate
    Lc = T
    T_flat = _np.zeros((U2, J * Lc), _np.float32)
    T_norm2 = _np.zeros(U2, _np.float32)
    for k in range(U2):
        ei = EIs_ref[:, :, int(keep_locs[k])]
        Ti = ei[panel_channels, :Lc].reshape(-1).astype(_np.float32)
        T_flat[k, :] = Ti
        T_norm2[k] = float(_np.dot(Ti, Ti)) + 1e-12

    # Outputs
    residuals = _np.zeros((N, U2), _np.float32)
    alphas = _np.zeros((N, U2), _np.float32)

    # Process spikes in batches to avoid huge RAM
    for b0 in range(0, N, int(batch_size)):
        b1 = min(b0 + int(batch_size), N)
        S = snips[panel_channels, :Lc, b0:b1].reshape(J * Lc, b1 - b0).astype(_np.float32)  # [J*L, B]
        S_norm2 = _np.sum(S * S, axis=0)  # [B]
        # For each candidate, compute dots & residuals
        for k in range(U2):
            Ti = T_flat[k, :]                                        # [J*L]
            dots = Ti @ S                                            # [B]
            alpha = _np.clip(dots / T_norm2[k], 0.0, float(alpha_max))
            res = S_norm2 - 2.0 * alpha * dots + (alpha * alpha) * T_norm2[k]
            residuals[b0:b1, k] = res
            alphas[b0:b1, k] = alpha

    # Decide per spike
    # Null residual is ||S||^2 on the panel
    # (We recompute globally once; same as S_norm2 but for all spikes.)
    S_all = snips[panel_channels, :Lc, :].reshape(J * Lc, N).astype(_np.float32)
    null_res = _np.sum(S_all * S_all, axis=0) + 1e-12  # [N]

    best_idx = _np.argmin(residuals, axis=1)           # [N]
    best_val = residuals[_np.arange(N), best_idx]
    # Second best
    # Handle U2==1 safely
    if U2 >= 2:
        part = _np.partition(residuals, 1, axis=1)
        second_val = part[:, 1]
    else:
        second_val = _np.full(N, _np.inf, _np.float32)

    improve = null_res - best_val
    improve_frac = improve / null_res

    # Rules
    assigned = (improve_frac >= float(improve_frac_min))   # must beat the null by enough
    margin_frac = (second_val - best_val) / (best_val + 1e-12)
    ambiguous = assigned & (margin_frac <= float(ambiguity_margin_frac))
    labels = _np.full(N, -1, int)                          # -1 unknown
    labels[assigned] = (cand_ids_keep[best_idx[assigned]]).astype(int)
    labels[ambiguous] = -2                                 # -2 ambiguous (override)

    # ---- Diagnostics plots ----
    if plot:
        # Distribution of improvement
        _plt.figure(figsize=(6, 3))
        _plt.hist(improve_frac, bins=100, alpha=0.8)
        _plt.axvline(float(improve_frac_min), color='r', ls='--', label='accept thr')
        _plt.title("Improvement fraction vs null (panel)"); _plt.xlabel("improve_frac"); _plt.ylabel("#spikes")
        _plt.legend(); _plt.tight_layout()

        # Best vs 2nd-best residual (sample)
        if N > 0:
            idx = _np.arange(N)
            sample = idx if N <= int(plot_sample_scatter) else _np.random.default_rng(0).choice(idx, size=int(plot_sample_scatter), replace=False)
            _plt.figure(figsize=(4, 4))
            _plt.scatter(best_val[sample], second_val[sample], s=4, alpha=0.5)
            _plt.plot([best_val[sample].min(), best_val[sample].max()],
                      [best_val[sample].min(), best_val[sample].max()], 'k--', lw=0.8)
            _plt.xlabel("best residual"); _plt.ylabel("second residual")
            _plt.title("Best vs second residual (sample)"); _plt.tight_layout()

        # Counts
        cnt_assigned = int((labels >= 0).sum())
        cnt_amb = int((labels == -2).sum())
        cnt_unknown = int((labels == -1).sum())
        _plt.figure(figsize=(4, 3))
        _plt.bar(["assigned", "ambiguous", "unknown"], [cnt_assigned, cnt_amb, cnt_unknown])
        _plt.title("Assignment counts"); _plt.tight_layout()

    return {
        "labels": labels,                        # per-spike: unit id or -1 unknown or -2 ambiguous
        "residuals": residuals,                  # [N,U2]
        "alphas": alphas,                        # [N,U2]
        "best_idx": best_idx,                    # [N] index in kept list
        "best_unit": _np.where(labels >= 0, labels, -1),
        "improve_frac": improve_frac,
        "counts": {
            "assigned": int((labels >= 0).sum()),
            "ambiguous": int((labels == -2).sum()),
            "unknown": int((labels == -1).sum())
        }
    }


# ---------- Convenience wrapper (A+B+C) with plots everywhere ----------

def ks_prune_panel_assign(
    EIs: _np.ndarray, snips: _np.ndarray, unit_id: int,
    *,
    center_index: int = 40,
    amp_ratio_bounds=(0.5, 1.5),
    min_abs_amp: float = 5.0,
    evidence_mode: str = "ptp",
    evidence_q: float = 95.0,
    rho_commit: float = 0.25,
    tau_evidence: float = 0.40,
    panel_cap: int = 24,
    per_pair_top_m: int = 4,
    alpha_max: float = 3.0,
    improve_frac_min: float = 0.15,
    ambiguity_margin_frac: float = 0.05,
    batch_size: int = 10000,
    plot: bool = True,
    ei_positions: _np.ndarray = None
) -> dict:
    """
    One-call pipeline doing A (prune), B (panel), C (assign) with diagnostics.
    Returns a dict that includes everything needed for debugging.
    """
    # A) prune
    A = prune_templates_by_channel_evidence(
        EIs, unit_id, snips,
        center_index=center_index,
        amp_ratio_bounds=amp_ratio_bounds,
        min_abs_amp=min_abs_amp,
        evidence_mode=evidence_mode,
        evidence_q=evidence_q,
        rho_commit=rho_commit,
        tau_evidence=tau_evidence,
        plot=plot,
        ei_positions=ei_positions
    )
    ref_ch = int(A["ref_ch"])
    cand_all = A["candidates_amp_gated"]
    cand_keep = A["candidates_pruned"]

    if plot:
        _plt.figure(figsize=(5, 0.8))
        _plt.text(0.01, 0.4, f"Candidates: amp-gated {cand_all.size} → pruned {cand_keep.size}", fontsize=10)
        _plt.axis("off"); _plt.tight_layout()

    # B) discriminative panel
    B = select_discriminative_panel(
        EIs_ref=A["EIs_ref"],
        cand_ids_all=cand_all,
        cand_ids_keep=cand_keep,
        panel_cap=panel_cap,
        per_pair_top_m=per_pair_top_m,
        include_ref_ch=ref_ch,
        plot=plot
    )
    panel = B["panel_channels"]

    # C) assign
    C = assign_spikes_on_panel_via_ls(
        EIs_ref=A["EIs_ref"],
        cand_ids_all=cand_all,
        cand_ids_keep=cand_keep,
        snips=snips,
        panel_channels=panel,
        batch_size=batch_size,
        alpha_max=alpha_max,
        improve_frac_min=improve_frac_min,
        ambiguity_margin_frac=ambiguity_margin_frac,
        plot=plot
    )

    return {"A": A, "B": B, "C": C, "ref_ch": ref_ch, "panel": panel}

### === END KS pruning + discriminative panel + assignment ===


### === BEGIN template-required vs observed per-channel (line plots) ===
import numpy as _np
import matplotlib.pyplot as _plt

def _recenter_ei_to_channel_trough__local(ei: _np.ndarray, ch_ref: int, center_index: int = 40) -> _np.ndarray:
    """
    Zero-padded shift so the trough on ch_ref sits at center_index.
    Kept local to avoid dependency surprises.
    """
    ei = _np.asarray(ei, dtype=_np.float32)
    C, T = ei.shape
    ch_ref = int(ch_ref)
    t0 = int(_np.argmin(ei[ch_ref]))
    s = int(center_index - t0)
    if s == 0:
        return ei.copy()
    out = _np.zeros_like(ei)
    if s > 0:
        out[:, s:] = ei[:, :T - s]
    else:
        s2 = -s
        out[:, :T - s2] = ei[:, s2:]
    return out

def plot_required_vs_observed_per_template(
    EIs: _np.ndarray,          # [C, T, U]
    snips: _np.ndarray,        # [C, T, N]  (snippets for the target ref channel; aligned similarly to EIs)
    *,
    template_ids=None,         # list/array of template indices to plot; None -> plot all
    window_half: int = 3,      # ± samples around each template’s trough time per channel
    ref_ch: int = None,        # optional: recenter every template so its trough on THIS channel is at center_index
    center_index: int = 40,
    threshold_ratio: float = 1.25,  # "throw out" if template_peak > threshold_ratio * spike_P95
    min_required_amp: float = 0.0,  # ignore channels where template_peak < this (optional; 0.0 by default)
    figsize=(12, 4),
    return_data: bool = False
):
    """
    For each selected template k, makes ONE line plot with TWO curves across channels:
      - curve A: template negative peak per channel:   A[c] = -min(EI_k[c, :])
      - curve B: 95th percentile of snippet negative peaks at the template’s expected trough time:
                  For each channel c:
                     let t_kc = argmin(EI_k[c, :])  (after optional recenter to ref_ch)
                     per-snippet trough_i(c) = -min( snips[c, t_kc-window_half : t_kc+window_half+1, i] )
                  B[c] = percentile_95( trough_i(c) over i )

    Title: counts channels where A[c] > threshold_ratio * B[c] (those are "thrown out").
    If min_required_amp>0, channels with A[c] < min_required_amp are ignored in the count.

    Returns (if return_data=True): list of dicts with arrays for each plotted template.
    """
    EIs = _np.asarray(EIs, _np.float32)
    snips = _np.asarray(snips, _np.float32)

    C, T_ei, U = EIs.shape
    C2, T_sn, N = snips.shape
    if C2 != C:
        raise ValueError(f"snips has C={C2}, but EIs has C={C}")

    tmpl_ids = _np.arange(U, dtype=int) if template_ids is None else _np.asarray(template_ids, dtype=int).ravel()
    out = [] if return_data else None

    for k in tmpl_ids:
        ei_k = EIs[:, :, int(k)]

        # Optional: recenter this template so its trough on ref_ch sits at center_index
        if ref_ch is not None:
            ei_k = _recenter_ei_to_channel_trough__local(ei_k, int(ref_ch), center_index=center_index)

        # Curve A: template negative peak per channel
        tmpl_negpeak = (-ei_k.min(axis=1)).astype(_np.float32)   # [C]

        # For curve B, we need channel-wise trough indices from THIS template
        trough_idx = _np.argmin(ei_k, axis=1).astype(int)        # [C]
        B = _np.zeros(C, _np.float32)

        # Window bounds must respect snippet length (T_sn)
        for c in range(C):
            t0 = int(trough_idx[c])
            if t0 < 0 or t0 >= T_sn:
                B[c] = 0.0
                continue
            lo = max(0, t0 - int(window_half))
            hi = min(T_sn, t0 + int(window_half) + 1)
            # per-snippet negative trough near expected time on channel c
            # snips[c, lo:hi, :] -> [win, N]; take min over time, then negate to get positive amplitude
            per_spike = -snips[c, lo:hi, :].min(axis=0) if hi > lo else _np.zeros((N,), _np.float32)
            if per_spike.size == 0:
                B[c] = 0.0
            else:
                B[c] = _np.percentile(per_spike, 95).astype(_np.float32)

        # Channels to count as "violations"
        if float(min_required_amp) > 0.0:
            valid = tmpl_negpeak >= float(min_required_amp)
        else:
            valid = _np.ones(C, dtype=bool)

        violated = valid & (tmpl_negpeak > float(threshold_ratio) * (B + 1e-12))
        n_viol = int(violated.sum())

        # Plot
        _plt.figure(figsize=figsize)
        _plt.plot(tmpl_negpeak, label="template trough (per channel)", lw=1.6)
        _plt.plot(B, label="spikes: 95% trough near expected time", lw=1.6)
        _plt.xlabel("Channel")
        _plt.ylabel("Amplitude (ADC)")
        ttl = f"Template {int(k)} · throw_out_channels={n_viol}  (rule: tmpl > {threshold_ratio:.2f}× spikeP95"
        if min_required_amp > 0:
            ttl += f", only where tmpl≥{min_required_amp:g}"
        ttl += ")"
        _plt.title(ttl)
        _plt.legend(loc="upper right", fontsize=9)
        # Optional visual: mark violations (light shading)
        if n_viol > 0:
            idx = _np.flatnonzero(violated)
            _plt.scatter(idx, tmpl_negpeak[idx], s=10, c="r", alpha=0.6, label="_nolegend_")
        _plt.tight_layout()

        if return_data:
            out.append({
                "template_id": int(k),
                "template_curve": tmpl_negpeak,
                "snipP95_curve": B,
                "violated_mask": violated,
                "n_violated": n_viol
            })

    return out
### === END template-required vs observed per-channel (line plots) ===
