# ────────── collision_cuda_clean.py ──────────
import torch, math
# ---------- constants you already use ----------
MAX_LAG   = 20        # ±20 → 41 lags
TOP_K     = 3         # keep 3 best
SNIP_LEN  = 121

# ------------------------------------------------
def build_rolled_bank(EIs, max_lag=MAX_LAG):
    """EIs [U,C,T] → rolled [U, (2*max_lag+1), C, T]"""
    shifts = range(-max_lag, max_lag + 1)
    return torch.stack([torch.roll(EIs, s, dims=-1) for s in shifts], dim=1)

# -----------------------------------------------
PEAK_IDX  = 40          # EI peak sample
CLIP_W    = 200.0       # µV weight clip

# ------------------------------------------------------------
# ---------- collision_cuda_quick.py (replace previous helper) ----------

def quick_unit_filter_gpu(raw_snip,                     # [C,121]
                          EIs, peak_ch, sel_mask,       # tensors on cuda
                          top_k=3):
    """
    Returns
    --------
    good_rows : list[int]                 units that improve Δ-RMS
    best_lags : torch.int8 [U, top_k]     top-k lags per unit (-20…+20);
                                           meaningless for rows not in good_rows
    """
    device = raw_snip.device
    U, C, T = EIs.shape

    # ---------- roll EI bank once for all 41 shifts ------------------
    shifts = torch.arange(0, 41, device=device)                # 0 … 40
    rolled = torch.stack(
            [torch.roll(EIs, int(s), dims=-1) for s in shifts], dim=1)  # [U,41,C,T]

    # ---------- peak-channel dot products ----------------------------
    raw_pk = raw_snip[peak_ch]                                  # [U,T]
    ei_pk  = EIs[torch.arange(U, device=device), peak_ch]       # [U,T]
    dots = (torch.stack(
            [torch.roll(ei_pk, int(s), dims=-1) for s in shifts], 1)    # [U,41]
        * raw_pk.unsqueeze(1)).sum(-1)
    idx0  = torch.argmax(dots, dim=1)                            # 0 … 40
    lag0  = idx0.to(torch.int16)                                 # 0 … 40

    # ---------- Δ-RMS at this single lag -----------------------------
    aligned = rolled[torch.arange(U, device=device), idx0]       # [U,C,T]

    mask3d   = sel_mask.float().unsqueeze(-1)                   # [U,C,1]
    raw_sel  = raw_snip.unsqueeze(0) * mask3d                  # [U,C,T]
    aligned  = aligned * mask3d                                 # [U,C,T]

    rms_raw  = torch.sqrt((raw_sel**2).mean(-1))                # [U,C]
    rms_res  = torch.sqrt(((raw_sel - aligned)**2).mean(-1))
    weights  = (aligned.max(-1).values - aligned.min(-1).values
                ).clamp_(max=CLIP_W)                            # [U,C]
    delta_u  = (weights * (rms_res - rms_raw)).sum(-1)          # [U]

    good_rows = torch.where(delta_u < 0)[0]
    if good_rows.numel() == 0:
        return [], torch.zeros(U, top_k, dtype=torch.int8, device=device)

    # ---------- fine scan: Δ-RMS for all 41 lags, good rows only -----
    g = good_rows
    rms_raw_g = rms_raw[g].unsqueeze(1)                         # [G,1,C]
    rms_res_g = torch.sqrt(((raw_snip.unsqueeze(0) - rolled[g])**2).mean(-1))
    weights_g = (rolled[g].max(-1).values -
                 rolled[g].min(-1).values).clamp_(max=CLIP_W)
    delta41   = (weights_g * (rms_res_g - rms_raw_g)).sum(-1)   # [G,41]

    gain, idx41 = torch.topk(delta41, k=top_k, dim=1, largest=False)
    best_lags = torch.zeros(U, top_k, dtype=torch.int8, device=device)
    best_lags[g] = idx41.to(torch.int8)                          # 0 … 40

    return good_rows.tolist(), best_lags

# ------------------------------------------------
def scan_unit_lags_t(raw_snip, rolled_EIs):
    """
    raw_snip     [C,T]
    rolled_EIs   [U,41,C,T]
    Returns lags [U,TOP_K] int8   (best TOP_K lags per unit)
    """
    U, L, C, T = rolled_EIs.shape
    raw_b = raw_snip.unsqueeze(0).unsqueeze(0)     # [1,1,C,T]
    err   = raw_b - rolled_EIs                     # [U,L,C,T]

    rms_r = torch.sqrt((err**2).mean(-1))          # [U,L,C]
    rms_x = torch.sqrt((raw_snip**2).mean(-1))     # [C]
    delta = (rms_x - rms_r).sum(-1)                # [U,L]

    gain, idx = torch.topk(delta, k=TOP_K, dim=1)
    lags = (idx - MAX_LAG).to(torch.int8)          # back to ±20
    return lags                                    # [U,3]

# ------------------------------------------------
def score_active_set_t(active_mask, rolled_EIs, raw_snip):
    """
    active_mask [B,U,41] bool
    rolled_EIs  [U,41,C,T]
    raw_snip    [C,T]
    Return      score [B] ΔRMS (higher is better)
    """
    tmpl = torch.einsum('bul,ulct->bct', active_mask.float(), rolled_EIs)
    err  = raw_snip.unsqueeze(0) - tmpl            # [B,C,T]
    rms_r= torch.sqrt((err**2).mean(-1))           # [B,C]
    rms_x= torch.sqrt((raw_snip**2).mean(-1))      # [C]
    return (rms_x - rms_r).sum(-1)                 # [B]
# ────────────────────────────────────────────────
