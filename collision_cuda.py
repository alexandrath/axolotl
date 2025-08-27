# ─────────── collision_cuda.py ───────────
import torch
from torch import fft, einsum

# --------------------------------------------------------------------
# one-time constants – keep in sync with collision_utils.py
TOP_K         = 3
MAX_W_UNITS   = 15
BETA          = 0.5     # weight exponent
MAX_LAG       = 3       # ±3 → 7 lags   (edit if you use another value)
T_SNIP        = 121     # snippet / EI length
PEAK_SAMPLE   = 40      # EI peak index
# --------------------------------------------------------------------

def to_device(arr, dtype=torch.float32, device='cuda'):
    return torch.as_tensor(arr, dtype=dtype, device=device)

# ---------- 1.  batched peak-channel x-corr -------------------------
def quick_unit_filter_t(raw_chunk,            # [C,Ttotal]  torch
                        EIs,                  # [U,C,T]
                        peak_ch):             # [U]  int32
    """
    Returns tensor lag_raw  [U]  (integer lag of EI peak to raw trace)
    """
    U = EIs.shape[0]
    raw_peak = raw_chunk[peak_ch, :]          # [U,Ttot]
    ei_peak  = EIs[torch.arange(U), peak_ch]  # [U,T]
    fft_len  = raw_peak.shape[1] + T_SNIP - 1

    raw_fft  = fft.rfft(raw_peak, n=fft_len)
    ei_fft   = fft.rfft(ei_peak,  n=fft_len)
    xcorr    = fft.irfft(raw_fft.conj() * ei_fft, n=fft_len)

    lag_raw  = torch.argmax(xcorr, dim=1) - (T_SNIP - 1)
    return lag_raw.to(torch.int16)

# ---------- 2.  rolled EI bank -------------------------------------
def build_rolled_bank(EIs, max_lag=MAX_LAG):
    """Return rolled_EI  [U,L,C,T]  with L = 2*max_lag+1"""
    shifts = range(-max_lag, max_lag + 1)
    return torch.stack([torch.roll(EIs, s, dims=-1) for s in shifts], dim=1)

# ---------- 3.  per-unit best lags (vector Δ-RMS) ------------------
def scan_unit_lags_t(raw_snip,                # [C,T]
                     rolled_EI,               # [U,L,C,T]
                     weights):                # [C]
    """
    Returns (best_lags  [U,TOP_K]  int8,
             best_gain  [U,TOP_K]  float32 )
    """
    U, L, C, T = rolled_EI.shape
    raw  = raw_snip.unsqueeze(0).unsqueeze(0)         # [1,1,C,T]
    err  = raw - rolled_EI                            # [U,L,C,T]
    rms_r= torch.sqrt((err**2).mean(-1))              # [U,L,C]
    rms_x= torch.sqrt((raw**2).mean(-1)).expand_as(rms_r) # [U,L,C]

    delta = (weights * (rms_x - rms_r)).sum(-1)       # [U,L]
    gain, idx = torch.topk(delta, k=TOP_K, dim=1)
    lags = (idx - MAX_LAG).to(torch.int8)             # shift back to ±
    return lags, gain.to(torch.float32)

# ---------- 4.  score_active_set  (batch) ---------------------------
def score_active_set_t(active_masks,          # [B,U,L] bool  1 where used
                       rolled_EI,             # [U,L,C,T]
                       raw_snip,              # [C,T]
                       weights):              # [C]
    """
    Return Δ-RMS score  [B]  for B combos in parallel
    """
    # template sum: einsum over uid×lag
    tmpl = einsum('bul,ulct->bct', active_masks.float(), rolled_EI)
    err  = raw_snip.unsqueeze(0) - tmpl                 # [B,C,T]
    rms_r= torch.sqrt((err**2).mean(-1))                # [B,C]
    rms_x= torch.sqrt((raw_snip**2).mean(-1))           # [C]
    delta= (weights * (rms_x - rms_r)).sum(-1)          # [B]
    return delta

# ---------- 5.  helper to build weights once per chunk -------------
def channel_weights(p2p_max, sigma):
    """p2p_max, sigma  [C] numpy or torch → torch.float32"""
    w = p2p_max / (sigma**2 + 1e-9)
    return w / w.mean()         # normalise to mean=1 (optional)
# ───────────────────────────────────────────────────────────────────
