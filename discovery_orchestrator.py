"""
Discovery orchestrator — integrated version (Stage A+B)

Single-owner design:
  • TemplateIndex + OrchestratorConfig + helper utilities (Stage A)
  • Integrated DiscoveryOrchestrator that owns prescreen, cached discovery (FinderSession),
    judging (ΔRMS harm-map + gate + lag-health + exceed-veto), local peeling, merge/dedup,
    and post-accept plotting (Stage B, no callbacks).

This file intentionally keeps `collision_utils.py` unchanged and imports helpers from it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable, Any
import numpy as np
import importlib, traceback

# ---- External helpers from your project (must exist in your repo) ----
from sklearn.mixture import GaussianMixture
import axolotl_utils_ram

# Scoring / plotting utilities from collision_utils
import collision_utils
importlib.reload(collision_utils)
try:
    from collision_utils import (
        compute_harm_map_noamp,
        compute_spike_gate,
        build_ideal_delta,
        plot_harm_heatmap,
        plot_spike_delta_summary,
        plot_help_harm_lines,
        plot_help_harm_scatter_swapped,
        plot_deviation_lines,
        per_channel_gmm_bimodality,
        median_ei_adaptive,
        compute_global_baseline_mean,
    )
except Exception as e:
    raise ImportError(f"collision_utils is missing required helpers: {e}")

# Optional EI comparison implementation
try:
    from compare_eis import compare_eis
except Exception:
    compare_eis = None


try:
    import plot_ei_waveforms as _pew
    importlib.reload(_pew)           # ensure fresh
except ImportError:
    _pew = None
except Exception:
    traceback.print_exc()
    _pew = None




__all__ = (
    "OrchestratorConfig",
    "TemplateRecord",
    "TemplateIndex",
    "DiscoveryOrchestrator",
    "ChannelStats",
)

# =====================================================================================
# Configuration
# =====================================================================================
@dataclass
class OrchestratorConfig:
    # Prescreen / reuse
    USE_PRESCREEN: bool = True
    P2P_REUSE_THR: float = 100.0  # ADC units
    PRESCREEN_TOP_T: int = 6
    PRESCREEN_STOP_FRACTION: float = 0.65
    MIN_PRESCREEN_ACCEPT: int = 15

    # Harm map parameters
    P2P_THR: float = 30.0
    MAX_CHANNELS: int = 80
    MIN_CHANNELS: int = 10
    LAG_RADIUS: int = 3
    WEIGHT_BY_P2P: bool = True
    WEIGHT_BETA: float = 0.7

    # Gate thresholds (see compute_spike_gate)
    THR_GLOBAL: float = -2.0
    THR_CHANNEL: float = 0.0
    MIN_GOOD_FRAC: float = 0.45
    MAX_BAD_DELTA: float = 10.0

    # Profile exceed veto (uses build_ideal_delta stats)
    EXCEED_THRESH: Optional[float] = 20.0

    # Lag-health reporting
    CENTRAL_BAND: int = 1
    WILSON_Z: float = 1.96

    # Residual trimming on detect channel after subtraction (local only)
    RESIDUAL_SAMPLE: int = 40
    RESIDUAL_KEEP_THR: float = -100.0

    # Same-unit decision thresholds
    EI_MAX_LAG: int = 2
    EI_SIM_THR: float = 0.92
    OVERLAP_TOL_SAMPLES: int = 2
    OVERLAP_FRAC_THR: float = 0.80

    # Dedup score weights (choose_canonical)
    W_LOGN: float = 2.0
    W_SNR: float = 1.0
    W_MAD: float = 1.0
    W_EDGE: float = 1.0
    W_EXCEED: float = 0.5

    # Diagnostics / plotting
    SHOW_DIAGNOSTICS: bool = True
    SHOW_PLOTS: bool = False


# =====================================================================================
# Template record & registry
# =====================================================================================
@dataclass
class TemplateRecord:
    id: int
    ei: np.ndarray                 # [C,T]
    home_channel: int              # canonical anchor (timebase)
    peak_channel: int
    detect_channels: List[Tuple[int, int]]  # list of (detect_ch, n_spikes_contributed)
    spike_times: np.ndarray        # absolute times in HOME timebase (int64)
    selected_channels: np.ndarray  # channels used in harm map (≤ MAX_CHANNELS)
    p2p: np.ndarray                # [C] per-channel p2p
    gbm: float
    snr: float
    tpeak: np.ndarray              # [C] per-channel negative-peak sample index

    # QC summaries (optional)
    lag_mad: Optional[float] = None
    central_lb: Optional[float] = None
    edge_frac: Optional[float] = None
    n_exceed_max_gap: Optional[int] = None
    quality_flag: str = "good"

    def add_detect_context(self, ch: int, n: int):
        self.detect_channels.append((int(ch), int(n)))


class TemplateIndex:
    """In-memory registry of accepted templates and a channel→templates map."""

    def __init__(self):
        self.accepted: List[TemplateRecord] = []
        self.by_channel: Dict[int, List[int]] = {}
        self._next_id = 1

    def allocate_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def get(self, tid: int) -> TemplateRecord:
        return next(t for t in self.accepted if t.id == tid)

    def register(self, rec: TemplateRecord):
        self.accepted.append(rec)
        # index by selected channels so reuse can quickly find strong candidates
        for c in rec.selected_channels.tolist():
            self.by_channel.setdefault(int(c), []).append(rec.id)

    def update_spike_times(self, tid: int, new_times_home: np.ndarray, *,
                           tolerance: int = 5, detect_ch: Optional[int] = None):
        rec = self.get(tid)
        merged = union_spike_times(rec.spike_times, new_times_home, tolerance)
        rec.spike_times = merged
        if detect_ch is not None:
            rec.add_detect_context(detect_ch, int(new_times_home.size))

    def find_candidates_for_channel(self, ch: int, *, p2p_reuse_thr: float) -> List[TemplateRecord]:
        ids = self.by_channel.get(int(ch), [])
        out: List[TemplateRecord] = []
        for tid in ids:
            rec = self.get(tid)
            if rec.p2p[int(ch)] >= p2p_reuse_thr:
                out.append(rec)
        out.sort(key=lambda r: float(r.p2p[int(ch)]), reverse=True)
        return out

    def mark_alias(self, loser_id: int, winner_id: int):
        rec = self.get(loser_id)
        rec.quality_flag = "alias"


# =====================================================================================
# Core helpers (time mapping, spike unions, lag metrics, evaluation, peeling, dedup)
# =====================================================================================

def compute_tpeak(ei: np.ndarray) -> np.ndarray:
    return np.argmin(ei, axis=1).astype(np.int32)


def map_spike_times_to_home(spike_times_detect: np.ndarray, detect_channel: int,
                            home_channel: int, tpeak: np.ndarray) -> np.ndarray:
    spike_times_detect = np.asarray(spike_times_detect, dtype=np.int64)
    delta = int(tpeak[int(detect_channel)]) - int(tpeak[int(home_channel)])
    return spike_times_detect - np.int64(delta)


def union_spike_times(existing_home: np.ndarray, new_home: np.ndarray, tol: int = 5) -> np.ndarray:
    if existing_home is None or existing_home.size == 0:
        return np.unique(new_home.astype(np.int64))
    a = np.asarray(existing_home, dtype=np.int64)
    b = np.asarray(new_home, dtype=np.int64)
    a.sort(); b.sort()
    out = []
    i = j = 0
    while i < a.size and j < b.size:
        va, vb = a[i], b[j]
        if vb < va - tol:
            out.append(vb); j += 1
        elif va < vb - tol:
            out.append(va); i += 1
        else:
            out.append(min(va, vb)); i += 1; j += 1
    if i < a.size: out.extend(a[i:].tolist())
    if j < b.size: out.extend(b[j:].tolist())
    return np.unique(np.asarray(out, dtype=np.int64))


def _wilson_lower_bound(k: int, n: int, z: float = 1.96) -> float:
    if n <= 0: return float("nan")
    phat = k / n
    denom = 1.0 + (z*z)/n
    center = phat + (z*z)/(2*n)
    spread = z * np.sqrt((phat*(1.0 - phat) + (z*z)/(4*n)) / n)
    return float((center - spread) / denom)


def lag_metrics_from_res(res: Dict, mask: Optional[np.ndarray] = None,
                         central_band: int = 1, z: float = 1.96) -> Dict[str, float]:
    best = np.asarray(res["best_lag_per_spike"])  # [N]
    if mask is not None:
        best = best[np.asarray(mask, dtype=bool)]
    N = int(best.size)
    if N == 0:
        return dict(N=0, central_LB=np.nan, edge_frac=np.nan, mad=np.nan)
    lags = np.asarray(res["lags"])  # [L]
    lmin, lmax = int(lags.min()), int(lags.max())
    central_LB = _wilson_lower_bound(int((np.abs(best) <= central_band).sum()), N, z)
    edge_frac = float(((best == lmin) | (best == lmax)).mean())
    med = float(np.median(best))
    mad = float(np.median(np.abs(best - med)))
    return dict(N=N, central_LB=central_LB, edge_frac=edge_frac, mad=mad)


@dataclass
class EvalResult:
    accept_mask: np.ndarray
    best_lag_per_spike: np.ndarray
    res: Dict
    ideal: Dict
    gate: Dict
    lag_metrics: Dict[str, float]
    order: np.ndarray


def _build_order(res: Dict, accept_mask: np.ndarray) -> np.ndarray:
    mdw = np.asarray(res["mean_delta_weighted"])  # [N]
    idx_in = np.where(accept_mask)[0]
    idx_out = np.where(~accept_mask)[0]
    order = np.r_[idx_in[np.argsort(mdw[idx_in])], idx_out[np.argsort(mdw[idx_out])]]
    return order.astype(int)


def evaluate_template_on_channel(template: TemplateRecord, snips_pool: np.ndarray,
                                 cfg: OrchestratorConfig) -> EvalResult:
    res = compute_harm_map_noamp(
        template.ei, snips_pool,
        p2p_thr=cfg.P2P_THR, max_channels=cfg.MAX_CHANNELS, min_channels=cfg.MIN_CHANNELS,
        lag_radius=cfg.LAG_RADIUS, weight_by_p2p=cfg.WEIGHT_BY_P2P, weight_beta=cfg.WEIGHT_BETA,
    )
    ideal = build_ideal_delta(res, subset=None, good_thresh=cfg.THR_GLOBAL, top_frac=0.25)
    gate = compute_spike_gate(
        res,
        thr_global=cfg.THR_GLOBAL,
        thr_channel=cfg.THR_CHANNEL,
        min_good_frac=cfg.MIN_GOOD_FRAC,
        max_bad_delta=cfg.MAX_BAD_DELTA,
        weighted=True,
        weight_beta=cfg.WEIGHT_BETA,
        ideal=ideal,
        exceed_thresh=cfg.EXCEED_THRESH,
    )
    accept_mask = np.asarray(gate["accept_mask"], dtype=bool)
    lag_metrics = lag_metrics_from_res(res, mask=accept_mask,
                                       central_band=cfg.CENTRAL_BAND, z=cfg.WILSON_Z)
    order = _build_order(res, accept_mask)
    return EvalResult(
        accept_mask=accept_mask,
        best_lag_per_spike=np.asarray(res["best_lag_per_spike"]),
        res=res,
        ideal=ideal,
        gate=gate,
        lag_metrics=lag_metrics,
        order=order,
    )


def _roll_zero_1d(a: np.ndarray, s: int) -> np.ndarray:
    if s == 0: return a
    out = np.zeros_like(a)
    if s > 0: out[s:] = a[:-s]
    else:     out[:s] = a[-s:]
    return out


def peel_spikes_local(snips_pool: np.ndarray, ei: np.ndarray,
                      accept_mask: np.ndarray, best_lag_per_spike: np.ndarray, *,
                      detect_channel: int, residual_sample: int = 40, residual_keep_thr: float = -100.0
                      ) -> Tuple[np.ndarray, np.ndarray]:
    C, T, N = snips_pool.shape
    acc = np.asarray(accept_mask, dtype=bool)
    if not np.any(acc):
        return snips_pool, np.zeros(0, dtype=int)
    idx = np.where(acc)[0]
    for j in idx:
        lag = int(best_lag_per_spike[j])
        shifted = np.vstack([_roll_zero_1d(ei[c], lag) for c in range(C)])
        snips_pool[:, :, j] = snips_pool[:, :, j] - shifted
    amp_post = snips_pool[int(detect_channel), int(residual_sample), :]
    keep_mask = (amp_post < residual_keep_thr)
    peeled_idx = np.where(~keep_mask)[0]
    new_snips = snips_pool[:, :, keep_mask]
    return new_snips, peeled_idx.astype(int)


def _ei_cosine_sim_with_shift(ei_a: np.ndarray, ei_b: np.ndarray, max_lag: int = 2) -> float:
    def _norm(x):
        n = np.linalg.norm(x)
        return x / n if n > 0 else x
    best = -1.0
    for s in range(-max_lag, max_lag + 1):
        if s == 0:
            rolled = ei_b
        else:
            rolled = np.zeros_like(ei_b)
            if s > 0: rolled[:, s:] = ei_b[:, :-s]
            else:     rolled[:, :s] = ei_b[:, -s:]
        sim = float(np.dot(_norm(ei_a.ravel()), _norm(rolled.ravel())))
        best = max(best, sim)
    return best


def _spike_overlap_fraction(times_a: np.ndarray, times_b: np.ndarray, tol: int = 2) -> float:
    if times_a.size == 0 and times_b.size == 0: return 1.0
    if times_a.size == 0 or times_b.size == 0: return 0.0
    a = np.asarray(times_a, dtype=np.int64); a.sort()
    b = np.asarray(times_b, dtype=np.int64); b.sort()
    i = j = 0; match = 0
    while i < a.size and j < b.size:
        if b[j] < a[i] - tol: j += 1
        elif a[i] < b[j] - tol: i += 1
        else: match += 1; i += 1; j += 1
    denom = max(a.size, b.size)
    return match / denom


def same_unit(A: TemplateRecord, B: TemplateRecord, *, cfg: OrchestratorConfig) -> Tuple[bool, Dict[str, float]]:
    if compare_eis is not None:
        try:
            sim_mat = compare_eis([A.ei, B.ei], max_lag=cfg.EI_MAX_LAG)
            ei_sim = float(sim_mat[0][1])
        except Exception:
            ei_sim = _ei_cosine_sim_with_shift(A.ei, B.ei, max_lag=cfg.EI_MAX_LAG)
    else:
        ei_sim = _ei_cosine_sim_with_shift(A.ei, B.ei, max_lag=cfg.EI_MAX_LAG)
    delta_B_to_A_home = int(B.tpeak[B.home_channel]) - int(B.tpeak[A.home_channel])
    times_A = np.asarray(A.spike_times, dtype=np.int64)
    times_B_on_A = np.asarray(B.spike_times - np.int64(delta_B_to_A_home), dtype=np.int64)
    overlap = _spike_overlap_fraction(times_A, times_B_on_A, tol=cfg.OVERLAP_TOL_SAMPLES)
    is_same = (ei_sim >= cfg.EI_SIM_THR) and (overlap >= cfg.OVERLAP_FRAC_THR)
    return is_same, {"ei_sim": ei_sim, "overlap": overlap}


def choose_canonical(A: TemplateRecord, B: TemplateRecord, *, cfg: OrchestratorConfig) -> TemplateRecord:
    def score(rec: TemplateRecord) -> float:
        n = max(1, int(rec.spike_times.size))
        logn = np.log(n)
        mad = 0.0 if rec.lag_mad is None else float(rec.lag_mad)
        edge = 0.0 if rec.edge_frac is None else float(rec.edge_frac)
        ex = 0.0 if rec.n_exceed_max_gap is None else float(rec.n_exceed_max_gap) / max(1, rec.selected_channels.size)
        return (
            cfg.W_LOGN * logn + cfg.W_SNR * float(rec.snr)
            - cfg.W_MAD * mad - cfg.W_EDGE * edge - cfg.W_EXCEED * ex
        )
    return A if score(A) >= score(B) else B


def _safe_call_plotter(fn, *args, **kwargs):
    try:
        if fn is not None:
            return fn(*args, **kwargs)
    except Exception as e:
        print(f"[plot] {getattr(fn,'__name__',str(fn))} failed: {e}")
        return None


def post_accept_plots(*, eval_result: EvalResult, title_prefix: str = "", show_plots: bool = True):
    if not show_plots:
        return
    res = eval_result.res
    order = eval_result.order
    acc = eval_result.accept_mask
    vline_at = int(np.sum(acc))
    _safe_call_plotter(plot_harm_heatmap, res, sort_by_ptp=True, spike_order=order,
                       vclip=None, title=f"{title_prefix} Harm Δ heatmap (accepted→left)",
                       vline_at=vline_at, vline_kwargs={"color":"k","linestyle":"--","linewidth":1.5})
    _safe_call_plotter(plot_spike_delta_summary, res, weighted=True,
                       title=f"{title_prefix} Per-spike mean ΔRMS (weighted)")
    _safe_call_plotter(plot_help_harm_lines, res, thr=0.0, spike_order=order,
                       weighted=True, weight_beta=0.7,
                       title=f"{title_prefix} Counts and group means vs spike order")
    _safe_call_plotter(plot_help_harm_scatter_swapped, res, thr=0.0, spike_order=order,
                       weighted=True, weight_beta=0.7,
                       big_mask=acc, s_small=14, s_big=64,
                       title=f"{title_prefix} Mean Δ vs #channels; big = accepted")
    _safe_call_plotter(plot_deviation_lines, res, eval_result.ideal, var_threshold=3.0,
                       exceed_thresh=20.0, spike_order=order,
                       title=f"{title_prefix} Channel-profile deviation diagnostics")


def make_template_record(*, ei: np.ndarray, spike_times_detect: np.ndarray, detect_channel: int,
                         peak_channel: int, selected_channels: np.ndarray, p2p: np.ndarray,
                         gbm: float, snr: float, index: TemplateIndex) -> TemplateRecord:
    home_channel = int(peak_channel)
    tpeak = compute_tpeak(ei)
    spike_times_home = map_spike_times_to_home(spike_times_detect, detect_channel, home_channel, tpeak)
    rec = TemplateRecord(
        id=index.allocate_id(),
        ei=np.asarray(ei, dtype=np.float32),
        home_channel=home_channel,
        peak_channel=int(peak_channel),
        detect_channels=[(int(detect_channel), int(spike_times_detect.size))],
        spike_times=np.asarray(spike_times_home, dtype=np.int64),
        selected_channels=np.asarray(selected_channels, dtype=int),
        p2p=np.asarray(p2p, dtype=np.float32),
        gbm=float(gbm),
        snr=float(snr),
        tpeak=np.asarray(tpeak, dtype=np.int32),
    )
    return rec

# =====================================================================================
# Integrated Orchestrator (no callbacks) with cached FinderSession
# =====================================================================================

@dataclass
class ChannelStats:
    channel: int
    n_spikes_initial: int
    n_spikes_after_prescreen: int
    n_prescreen_templates_tried: int
    n_prescreen_spikes_peeled: int
    n_new_templates: int
    n_alias_merged: int


class FinderSession:
    """Stateful, cached candidate finder for a single detect channel."""
    def __init__(self, snips_master: np.ndarray, min_cluster_size: int = 20, max_bimodal_tries: int = 20):
        self.snips_master = snips_master              # [C,T,N0]
        self.N0 = int(snips_master.shape[2])
        self.alive_global = np.ones(self.N0, dtype=bool)
        self.min_cluster_size = int(min_cluster_size)
        self.max_bimodal_tries = int(max_bimodal_tries)
        self.cached_results: List[Dict[str, Any]] = []
        self.blacklist: set[int] = set()
        self.last_sub_idx_global: Optional[np.ndarray] = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        ei_pool = np.mean(self.snips_master, axis=2)
        pool_ids0 = np.arange(self.N0, dtype=np.int64)
        self.cached_results = per_channel_gmm_bimodality(
            ei_pool, self.snips_master, win40=3, n_top=self.max_bimodal_tries,
            min_cluster_size=self.min_cluster_size, pool_ids=pool_ids0
        )
        self._initialized = True

    def note_removed(self, orig_ids: np.ndarray):
        if orig_ids is None or len(orig_ids) == 0:
            return
        self.alive_global[np.asarray(orig_ids, dtype=int)] = False
        self.last_sub_idx_global = np.asarray(orig_ids, dtype=np.int64)

    def note_rejected(self, candidate_id: int):
        self.blacklist.add(int(candidate_id))

    def _maybe_refresh_candidate(self, cand: Dict[str, Any], snips_current: np.ndarray,
                                 pool_ids_current: np.ndarray) -> Dict[str, Any]:
        if self.last_sub_idx_global is None:
            return cand
        c_best = cand.get('chan', None); t_best = cand.get('t', None)
        if c_best is None or t_best is None:
            return cand
        cg = cand.get('cand_idx_global', None)
        if cg is None:
            return cand
        touched = np.intersect1d(np.asarray(cg, dtype=np.int64), self.last_sub_idx_global, assume_unique=False).size
        denom = max(1, np.asarray(cg).size)
        frac = touched / denom
        if frac < 0.30:
            return cand
        v_all = snips_current[int(c_best), int(t_best), :].reshape(-1, 1)
        try:
            gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0).fit(v_all)
            mu = gmm.means_.flatten(); labels = gmm.predict(v_all)
            comp = np.argmax(np.abs(mu))
            cand_local_mask = (labels == comp)
            new_global = pool_ids_current[np.where(cand_local_mask)[0]]
            cand['cand_idx_global'] = new_global.astype(np.int64)
        except Exception:
            pass
        return cand

    def next_candidate(self, snips_current: np.ndarray, pool_ids_current: np.ndarray,
                       cfg: OrchestratorConfig) -> Optional[Dict[str, Any]]:
        self.initialize()
        for cid, cand in enumerate(self.cached_results):
            if cid in self.blacklist:
                continue
            cg = cand.get('cand_idx_global', None)
            if cg is None:
                continue
            cg = np.asarray(cg, dtype=np.int64)
            alive_ids = np.where(self.alive_global)[0]
            cg_alive = np.intersect1d(cg, alive_ids, assume_unique=False)
            if cg_alive.size < self.min_cluster_size:
                continue
            cand_mask_local = np.in1d(pool_ids_current, cg_alive, assume_unique=False)
            if cand_mask_local.sum() < self.min_cluster_size:
                continue
            cand = self._maybe_refresh_candidate(cand, snips_current, pool_ids_current)
            snips_cand = snips_current[:, :, cand_mask_local]
            ei_cand = median_ei_adaptive(snips_cand)
            p2p = ei_cand.max(axis=1) - ei_cand.min(axis=1)
            snr = float(p2p.max() / max(np.median(p2p), 1e-9))
            if snr < 10.0:
                continue
            if cfg.P2P_THR is not None:
                chans = np.where(p2p >= cfg.P2P_THR)[0]
                if chans.size == 0:
                    chans = np.argsort(p2p)[-cfg.MIN_CHANNELS:]
            else:
                chans = np.argsort(p2p)[-cfg.MAX_CHANNELS:]
            if chans.size > cfg.MAX_CHANNELS:
                idx = np.argsort(p2p[chans])[::-1][:cfg.MAX_CHANNELS]
                chans = chans[idx]
            peak_ch = int(np.argmax(p2p))
            gbm = float(compute_global_baseline_mean(ei_cand))
            return dict(
                candidate_id=cid,
                ei=ei_cand,
                cand_mask_local=cand_mask_local,
                selected_channels=chans.astype(int),
                p2p=p2p.astype(np.float32),
                snr=snr,
                gbm=gbm,
                peak_channel=peak_ch,
            )
        return None


class DiscoveryOrchestrator:
    """Integrated orchestrator that owns prescreen, cached discovery, judging, peel, and merge."""

class DiscoveryOrchestrator:
    def __init__(self, raw_data, peak_idx_per_chan, template_index, cfg=None,
                 *, window=(-40, 80), edge_guard=200, max_n_per_chan=10000, top_frac=1.0,
                 ei_positions=None):      
        self.ei_positions = ei_positions
        self.raw = raw_data
        self.peaks = peak_idx_per_chan
        self.index = template_index
        self.cfg = cfg or OrchestratorConfig()
        self.window = window
        self.edge_guard = int(edge_guard)
        self.max_n = int(max_n_per_chan)
        self.top_frac = float(top_frac)
        self.n_channels = raw_data.shape[1]

    def _plot_ei_and_traces(self, *, ei, snips_acc, detect_channel, peak_channel, title_prefix=""):
        import matplotlib.pyplot as plt

        # EI waveform plot (nice geometry if available)
        if (_pew is not None) and (self.ei_positions is not None):
            try:
                plt.figure(figsize=(18, 8))
                _pew.plot_ei_waveforms(
                    ei, self.ei_positions,
                    ref_channel=detect_channel,
                    scale=90, box_height=1, box_width=50,
                    colors='black', aspect=0.5
                )
                plt.title(f"{title_prefix}EI (peak ch {peak_channel})")
                plt.show()
            except Exception as e:
                print(f"[plot] EI topography failed: {e}")
        else:
            # Fallback: just the peak-channel waveform
            plt.figure(figsize=(10, 2))
            plt.plot(ei[int(peak_channel)], lw=2)
            plt.title(f"{title_prefix}EI — peak ch {peak_channel}")
            plt.grid(True); plt.show()

        # Main-channel traces for accepted spikes
        tr = snips_acc[int(detect_channel)].T     # [n_acc, T]
        if tr.size:
            plt.figure(figsize=(12, 2))
            for row in tr:
                plt.plot(row, alpha=0.25)
            plt.plot(np.median(tr, axis=0), lw=2)
            plt.title(f"{title_prefix}Main-channel traces (n={tr.shape[0]})")
            plt.grid(True); plt.show()

        # Home-channel traces for accepted spikes
        tr = snips_acc[int(peak_channel)].T     # [n_acc, T]
        if tr.size:
            plt.figure(figsize=(12, 2))
            for row in tr:
                plt.plot(row, alpha=0.25)
            plt.plot(np.median(tr, axis=0), lw=2)
            plt.title(f"Peak-channel {peak_channel} traces (n={tr.shape[0]})")
            plt.grid(True); plt.show()


    def process_channel(self, Z: int) -> ChannelStats:
        Z = int(Z)
        snips_pool, spike_times_pool = self._build_local_pool(Z)
        N0 = snips_pool.shape[2]
        pool_ids = np.arange(N0, dtype=np.int64)

        finder = FinderSession(snips_pool, min_cluster_size=20, max_bimodal_tries=20)

        # -------- Prescreen with existing templates strong on this channel --------
        n_tried = 0
        n_peeled = 0
        if self.cfg.USE_PRESCREEN and N0 > 0:
            candidates = self.index.find_candidates_for_channel(Z, p2p_reuse_thr=self.cfg.P2P_REUSE_THR)
            for rec in candidates[: self.cfg.PRESCREEN_TOP_T]:
                if snips_pool.shape[2] == 0:
                    break
                n_tried += 1
                eval_res = evaluate_template_on_channel(rec, snips_pool, self.cfg)
                acc = eval_res.accept_mask
                n_acc = int(acc.sum())
                if n_acc >= self.cfg.MIN_PRESCREEN_ACCEPT:
                    times_Z = spike_times_pool[acc]
                    times_home = map_spike_times_to_home(times_Z, Z, rec.home_channel, rec.tpeak)
                    self.index.update_spike_times(rec.id, times_home, tolerance=5, detect_ch=Z)

                    if self.cfg.SHOW_PLOTS:
                        snips_acc_plot = snips_pool[:, :, acc].copy()
                        peak_ch_here = int(np.argmax(rec.p2p))  # or reuse rec.peak_channel if stored
                        self._plot_ei_and_traces(
                            ei=rec.ei, snips_acc=snips_acc_plot,
                            detect_channel=Z, peak_channel=peak_ch_here,
                            title_prefix=f"[reuse t{rec.id} on ch {Z}] "
                        )
                        # existing diagnostics (heatmap/lines/scatter)
                        post_accept_plots(eval_result=eval_res,
                                        title_prefix=f"[reuse t{rec.id} on ch {Z}] ",
                                        show_plots=True)

                    snips_pool, peeled_idx = peel_spikes_local(
                        snips_pool, rec.ei, acc, eval_res.best_lag_per_spike,
                        detect_channel=Z,
                        residual_sample=self.cfg.RESIDUAL_SAMPLE,
                        residual_keep_thr=self.cfg.RESIDUAL_KEEP_THR,
                    )
                    if peeled_idx.size > 0:
                        orig_ids = pool_ids[peeled_idx]
                        finder.note_removed(orig_ids)
                        keep_mask = np.ones(pool_ids.size, dtype=bool)
                        keep_mask[peeled_idx] = False
                        pool_ids = pool_ids[keep_mask]
                        spike_times_pool = spike_times_pool[keep_mask]
                        n_peeled += int(peeled_idx.size)

                    # if self.cfg.SHOW_PLOTS:
                    #     post_accept_plots(eval_result=eval_res,
                    #                       title_prefix=f"[reuse t{rec.id} on ch {Z}] ",
                    #                       show_plots=True)

                if snips_pool.shape[2] <= (1.0 - self.cfg.PRESCREEN_STOP_FRACTION) * N0:
                    break

        N_prescreen = snips_pool.shape[2]

        # -------- Discovery on residual using cached finder --------
        n_new = 0
        n_alias = 0
        while snips_pool.shape[2] >= 50:
            prop = finder.next_candidate(snips_pool, pool_ids, self.cfg)
            if prop is None:
                break

            rec_tmp = TemplateRecord(
                id=-1,
                ei=prop['ei'],
                home_channel=int(prop['peak_channel']),
                peak_channel=int(prop['peak_channel']),
                detect_channels=[(Z, int(prop['cand_mask_local'].sum()))],
                spike_times=spike_times_pool[prop['cand_mask_local']],
                selected_channels=np.asarray(prop['selected_channels'], dtype=int),
                p2p=np.asarray(prop['p2p'], dtype=np.float32),
                gbm=float(prop['gbm']),
                snr=float(prop['snr']),
                tpeak=compute_tpeak(prop['ei']),
            )

            eval_res = evaluate_template_on_channel(rec_tmp, snips_pool, self.cfg)
            acc = eval_res.accept_mask
            n_acc = int(acc.sum())
            if n_acc < self.cfg.MIN_PRESCREEN_ACCEPT:
                finder.note_rejected(prop['candidate_id'])
                continue

            snips_acc = snips_pool[:, :, acc]
            ei_final = median_ei_adaptive(snips_acc)
            p2p_final = ei_final.max(axis=1) - ei_final.min(axis=1)
            snr_final = float(p2p_final.max() / max(np.median(p2p_final), 1e-9))
            gbm_final = float(compute_global_baseline_mean(ei_final))
            peak_ch_final = int(np.argmax(p2p_final))

            rec_new = make_template_record(
                ei=ei_final,
                spike_times_detect=spike_times_pool[acc],
                detect_channel=Z,
                peak_channel=peak_ch_final,
                selected_channels=np.asarray(prop['selected_channels'], dtype=int),
                p2p=p2p_final.astype(np.float32),
                gbm=gbm_final,
                snr=snr_final,
                index=self.index,
            )

            matched = None
            winner = None
            for existing in list(self.index.accepted):
                same, diag = same_unit(existing, rec_new, cfg=self.cfg)
                if same:
                    matched = existing
                    winner = choose_canonical(existing, rec_new, cfg=self.cfg)
                    break

            if matched is not None:
                if winner is matched:
                    delta = int(rec_new.tpeak[rec_new.home_channel]) - int(matched.tpeak[matched.home_channel])
                    times_on_exist = rec_new.spike_times - np.int64(delta)
                    self.index.update_spike_times(matched.id, times_on_exist, tolerance=5, detect_ch=Z)
                    n_alias += 1
                else:
                    self.index.register(rec_new)
                    delta = int(matched.tpeak[matched.home_channel]) - int(rec_new.tpeak[rec_new.home_channel])
                    times_on_new = matched.spike_times - np.int64(delta)
                    self.index.update_spike_times(rec_new.id, times_on_new, tolerance=5)
                    self.index.mark_alias(matched.id, rec_new.id)
                    n_new += 1
            else:
                self.index.register(rec_new)
                n_new += 1

            if self.cfg.SHOW_PLOTS:
                snips_acc_plot = snips_acc.copy()  # we just computed snips_acc = snips_pool[:, :, acc]
                self._plot_ei_and_traces(
                    ei=ei_final, snips_acc=snips_acc_plot,
                    detect_channel=Z, peak_channel=peak_ch_final,
                    title_prefix=f"[discover ch {Z}] "
                )
                # existing diagnostics (heatmap/lines/scatter)
                post_accept_plots(eval_result=eval_res,
                                title_prefix=f"[discover ch {Z}] ",
                                show_plots=True)

            snips_pool, peeled_idx = peel_spikes_local(
                snips_pool, rec_new.ei, acc, eval_res.best_lag_per_spike,
                detect_channel=Z,
                residual_sample=self.cfg.RESIDUAL_SAMPLE,
                residual_keep_thr=self.cfg.RESIDUAL_KEEP_THR,
            )
            if peeled_idx.size > 0:
                orig_ids = pool_ids[peeled_idx]
                finder.note_removed(orig_ids)
                keep_mask = np.ones(pool_ids.size, dtype=bool)
                keep_mask[peeled_idx] = False
                pool_ids = pool_ids[keep_mask]
                spike_times_pool = spike_times_pool[keep_mask]

            # if self.cfg.SHOW_PLOTS:
            #     post_accept_plots(eval_result=eval_res,
            #                       title_prefix=f"[discover ch {Z}] ",
            #                       show_plots=True)

        return ChannelStats(
            channel=Z,
            n_spikes_initial=N0,
            n_spikes_after_prescreen=N_prescreen,
            n_prescreen_templates_tried=n_tried,
            n_prescreen_spikes_peeled=n_peeled,
            n_new_templates=n_new,
            n_alias_merged=n_alias,
        )

    # ----------------------- helpers -----------------------
    def _build_local_pool(self, Z: int) -> Tuple[np.ndarray, np.ndarray]:
        peak_times = np.asarray(self.peaks[Z], dtype=np.int64)
        amp_vals = self.raw[peak_times, Z]
        n_top = min(int(len(amp_vals) * self.top_frac), self.max_n)
        top_idx = np.argsort(amp_vals)[:n_top]
        sample_idx = peak_times[top_idx]
        left, right = int(self.window[0]), int(self.window[1])
        pad = max(self.edge_guard, -left, right)
        T_max = self.raw.shape[0]
        sample_idx = sample_idx[(pad <= sample_idx) & (sample_idx < T_max - pad)]
        C = self.n_channels
        all_channels = np.arange(C, dtype=int)
        snips, valid_times = axolotl_utils_ram.extract_snippets_fast_ram(
            raw_data=self.raw,
            spike_times=sample_idx,
            selected_channels=all_channels,
            window=self.window,
        )
        return snips, valid_times.astype(np.int64)
