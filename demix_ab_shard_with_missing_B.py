from __future__ import annotations

import numpy as np

# --- small helpers (zero‑padded integer shifts) ---------------------------------

def _shift_pad_1d(x: np.ndarray, shift: int) -> np.ndarray:
    T = x.shape[0]
    y = np.zeros_like(x)
    if shift == 0:
        y[:] = x
    elif shift > 0:
        y[shift:] = x[:T-shift]
    else:
        s = -shift
        y[:T-s] = x[s:]
    return y


def _shift_pad_2d(ei: np.ndarray, shift: int) -> np.ndarray:
    """Shift all channels by integer samples with zero padding. ei: [C, T]."""
    C, T = ei.shape
    y = np.zeros_like(ei)
    if shift == 0:
        y[:] = ei
    elif shift > 0:
        y[:, shift:] = ei[:, :T-shift]
    else:
        s = -shift
        y[:, :T-s] = ei[:, s:]
    return y


def _align_on_channel(snips: np.ndarray, ch: int, center_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Align each spike so the most negative sample on channel *ch* is at center_idx.
    snips: [C, T, N] → returns (aligned [C,T,N], per‑spike integer shifts [N])."""
    C, T, N = snips.shape
    out = np.zeros_like(snips)
    shifts = np.zeros(N, dtype=int)
    for s in range(N):
        trough_idx = int(np.argmin(snips[ch, :, s]))
        shift = center_idx - trough_idx
        shifts[s] = shift
        out[:, :, s] = _shift_pad_2d(snips[:, :, s], shift)
    return out, shifts


def _noise_std_from_baseline(snips: np.ndarray, k: int = 10) -> np.ndarray:
    """Estimate per‑channel noise from the first/last k samples across spikes.
    Returns σ_c with shape [C]."""
    C, T, N = snips.shape
    idx = np.r_[np.arange(k), np.arange(T-k, T)]
    noise = snips[:, idx, :].reshape(C, -1)
    return np.std(noise, axis=1) + 1e-6


def _p2p_per_channel(ei: np.ndarray) -> np.ndarray:
    return ei.max(axis=1) - ei.min(axis=1)


def _best_shift_for_residual(resid: np.ndarray, ref: np.ndarray, channels: list[int], maxlag: int) -> int:
    """Return lag d \in [-maxlag, maxlag] maximizing Σ_c ⟨resid_c, ref_c shifted by d⟩."""
    C, T = resid.shape
    best, lag_best = -1e18, 0
    for d in range(-maxlag, maxlag + 1):
        score = 0.0
        for ch in channels:
            if d >= 0:
                aa = resid[ch, d:]; bb = ref[ch, :T-d]
            else:
                s = -d; aa = resid[ch, :T-s]; bb = ref[ch, s:]
            score += float(np.dot(aa, bb))
        if score > best:
            best, lag_best = score, d
    return lag_best


def _bounded_beta(y_minus_A: np.ndarray, B_shifted: np.ndarray, W: np.ndarray,
                   lower: float, upper: float) -> float:
    """Argmin_β || y−A − β B ||_W^2 with β in [lower, upper]."""
    num = float(np.sum(W[:, None] * (y_minus_A * B_shifted)))
    den = float(np.sum(W[:, None] * (B_shifted * B_shifted)) + 1e-9)
    beta = num / den
    return float(np.clip(beta, lower, upper))


def _err_weighted(y_minus_A: np.ndarray, extra: np.ndarray | float, W: np.ndarray) -> float:
    r = y_minus_A if (isinstance(extra, (int, float)) and extra == 0.0) else (y_minus_A - extra)
    return float(np.sum(W[:, None] * r * r))


# --- main solver ---------------------------------------------------------------

def demix_ab_shard_with_missing_B(
    snips: np.ndarray,
    ei_positions: np.ndarray | None = None,
    anchor_chan: int | None = None,
    delta_max: int = 3,
    B_topk: int | None = None,
    use_amp: bool = True,
    beta_bounds: tuple[float, float] = (0.0, 1.6),
    max_iter: int = 6,
    snr_thresh: float = 2.0,
    presence_hard_thresh: float = 0.7,
    verbose: bool = False,
):
    """
    Two‑Templates‑One‑Jitter with missing B spikes and random interferers.

    Model per spike s: Y_s(c,t) ≈ A(c,t) + 1_{B present}(s) * β_s * B(c, t−δ_s).

    Inputs
    ------
    snips : [C, T, N] float32
        Shard snippets centered *approximately* on the stationary cell A.
    anchor_chan : int (optional)
        The A‑soma channel. If None, we pick the channel with the most negative
        median trough across spikes.
    delta_max : int
        Max integer jitter allowed for δ_s (samples).
    B_topk : int (optional)
        Number of channels to keep for B’s support (fixed across iterations).
        If None, we use max(12, min(32, C//8)).
    use_amp : bool
        If True, learns per‑spike β_s (bounded by beta_bounds). If False, β_s ≡ 1.
    snr_thresh : float
        Matched‑filter SNR threshold for B‑presence (≈2–3 works in practice).
    presence_hard_thresh : float
        Spikes with p_present ≥ this are used to update B each iteration.

    Returns
    -------
    A : [C, T]  stationary EI
    B : [C, T]  jittered EI
    deltas : [N] int  per‑spike δ_s (samples)
    betas  : [N] float  per‑spike β_s
    p_present : [N] float in [0,1]   soft presence probability of B
    info : dict with quick diagnostics
    """
    C, T, N = snips.shape

    # 0) choose anchor and align so the anchor trough sits at the center sample
    if anchor_chan is None:
        med = np.median(snips, axis=2)
        trough = med.min(axis=1)
        anchor_chan = int(np.argmin(trough))
    aligned, anchor_shifts = _align_on_channel(snips, anchor_chan, center_idx=T // 2)

    # 1) initial A as median across spikes; per‑channel noise weights
    A = np.median(aligned, axis=2)
    sigma = _noise_std_from_baseline(aligned, k=max(5, T // 8))
    W = 1.0 / (sigma + 1e-6) ** 2  # [C]

    # 2) pick a small fixed channel set for B (robust to interferers)
    R = aligned - A[:, :, None]
    resid_rms = np.sqrt(np.mean(R * R, axis=(1, 2)))  # [C]
    p2p_A = _p2p_per_channel(A)
    score = resid_rms / (p2p_A + 1e-3)
    idx_sorted = np.argsort(score)[::-1]
    if B_topk is None:
        B_topk = max(12, min(32, C // 8))
    B_ch = idx_sorted[:B_topk].tolist()
    M_B = np.zeros(C, dtype=bool); M_B[B_ch] = True

    # 3) crude δ_s via residual cross‑corr vs. a high‑energy residual reference
    energies = np.array([np.sum(R[B_ch, :, s] ** 2) for s in range(N)])
    s_ref = int(np.argmax(energies))
    ref = R[:, :, s_ref]
    deltas = np.zeros(N, dtype=int)
    for s in range(N):
        deltas[s] = _best_shift_for_residual(R[:, :, s], ref, B_ch, maxlag=delta_max)

    # 4) B₀ = median of residuals after de‑jittering, restricted to B_ch
    aligned_resid = np.zeros_like(aligned)
    for s in range(N):
        aligned_resid[:, :, s] = _shift_pad_2d(R[:, :, s], -deltas[s])
    B = np.zeros((C, T), np.float32)
    B[B_ch] = np.median(aligned_resid[B_ch], axis=2)

    # 5) iterate: per‑spike δ/β + matched‑filter presence → update A and B
    betas = np.ones(N, np.float32)
    p_present = np.zeros(N, np.float32)

    if verbose:
        print(f"Anchor ch={anchor_chan}, fixed B_ch={len(B_ch)}")

    for it in range(max_iter):
        # E‑step: per spike, find lag by correlating with B on B_ch; compute matched‑filter score
        snr_scores = np.zeros(N, float)
        errA = np.zeros(N); errAB = np.zeros(N)
        for s in range(N):
            Y = aligned[:, :, s]
            YmA = Y - A
            # best lag using only B channels
            lag = _best_shift_for_residual(YmA, B, B_ch, maxlag=delta_max)
            deltas[s] = lag
            Bsh = _shift_pad_2d(B, lag)

            # matched filter on B channels: s = (⟨YmA, B⟩_W) / ||B||_W
            num = float(np.sum(W[M_B, None] * (YmA[M_B] * Bsh[M_B])))
            den = float(np.sqrt(np.sum(W[M_B, None] * (Bsh[M_B] * Bsh[M_B])) + 1e-9))
            snr = num / (den + 1e-9)
            snr_scores[s] = snr

            if use_amp:
                beta = num / (den * den + 1e-9)
                betas[s] = float(np.clip(beta, beta_bounds[0], beta_bounds[1]))
            else:
                betas[s] = 1.0

            errA[s] = _err_weighted(YmA[M_B], 0.0, W[M_B])
            errAB[s] = _err_weighted(YmA[M_B], betas[s] * Bsh[M_B], W[M_B])

        # convert to a soft presence probability; threshold snr≈2–3 works well
        p_present = 1.0 / (1.0 + np.exp(-(snr_scores - snr_thresh)))

        # M‑step: update A using all spikes with soft subtraction of the B part
        Ycorr = np.zeros_like(aligned)
        for s in range(N):
            Y = aligned[:, :, s]
            Bsh = _shift_pad_2d(B, deltas[s])
            Ycorr[:, :, s] = Y - (p_present[s] * betas[s]) * Bsh
        A = np.median(Ycorr, axis=2)

        # Update B using only spikes with high B‑presence on the restricted channel set
        mask_sp = (p_present >= presence_hard_thresh)
        if mask_sp.any():
            S_sel = int(mask_sp.sum())
            R2 = np.zeros((len(B_ch), T, S_sel), np.float32)
            k = 0
            for s in range(N):
                if not mask_sp[s]:
                    continue
                YmA = aligned[:, :, s] - A
                R2[:, :, k] = _shift_pad_2d(YmA, -deltas[s])[B_ch] / max(betas[s], 1e-3)
                k += 1
            B[B_ch] = np.median(R2, axis=2)
        B[~M_B] = 0.0  # keep support tight

    info = {
        'presence_fraction': float(np.mean(p_present >= presence_hard_thresh)),
        'snr_mean': float(np.mean(snr_scores)),
        'anchor_chan': int(anchor_chan),
        'B_channels': B_ch,
        'err_improvement_mean_on_B': float(np.mean(errA - errAB)),
    }
    return A, B, deltas, betas, p_present, info


# --- convenience runner for quick experiment -----------------------------------

def run_demix_from_raw(
    raw_data: np.ndarray,               # [T, C] int16/float
    spike_times: np.ndarray,            # [N]
    window: tuple[int, int],            # (pre, post), e.g. (-40, 40)
    selected_channels: np.ndarray,      # [K]
    anchor_chan: int | None,
    ei_positions: np.ndarray | None = None,
    delta_max: int = 3,
    **kwargs,
):
    """
    Convenience wrapper that pulls snippets via extract_snippets_fast_ram,
    runs the demixer, and returns everything needed for plotting.
    """
    from axolotl_utils_ram import extract_snippets_fast_ram

    snips, valid_times = extract_snippets_fast_ram(raw_data, spike_times, window, selected_channels)
    A, B, deltas, betas, p_present, info = demix_ab_shard_with_missing_B(
        snips, ei_positions=ei_positions, anchor_chan=anchor_chan, delta_max=delta_max, **kwargs
    )
    return {
        'A': A, 'B': B,
        'deltas': deltas, 'betas': betas, 'p_present': p_present,
        'valid_times': valid_times, 'info': info,
        'selected_channels': selected_channels,
    }


# --- minimal plotting helper (uses your existing plot_ei_waveforms) -------------

def quick_plots(A: np.ndarray, B: np.ndarray, ei_positions: np.ndarray,
                deltas: np.ndarray, p_present: np.ndarray,
                ref_channel: int | None = None):
    """Two panels: EI overlay, and δ histogram gated by presence."""
    import matplotlib.pyplot as plt
    import plot_ei_waveforms as pew

    fig = plt.figure(figsize=(20, 12))
    ax1 = fig.add_subplot(2, 2, 1)
    pew.plot_ei_waveforms([A, B], ei_positions, ref_channel=ref_channel,
                           scale=70.0, ax=ax1, colors=['black', 'tab:orange'],
                           alpha=0.9, linewidth=0.8, box_height=1.0, box_width=50.0)
    ax1.set_title('Recovered A (black) and B (orange)')

    ax2 = fig.add_subplot(2, 2, 2)
    mask = p_present >= 0.7
    ax2.hist(deltas[mask], bins=np.arange(deltas.min()-0.5, deltas.max()+1.5, 1.0))
    ax2.set_title('δ histogram for spikes with high B‑presence')
    ax2.set_xlabel('δ (samples)'); ax2.set_ylabel('count')

    fig.tight_layout()
    return fig
