# =====================================================================================
# Integrated Orchestrator (no callbacks) with cached FinderSession
# =====================================================================================
from dataclasses import dataclass
from typing import Any

# Extra imports needed for the integrated flow
import axolotl_utils_ram
from sklearn.mixture import GaussianMixture

# Pull a few additional helpers from collision_utils
try:
    from collision_utils import per_channel_gmm_bimodality, median_ei_adaptive, compute_global_baseline_mean
except Exception as e:
    per_channel_gmm_bimodality = None
    median_ei_adaptive = None
    compute_global_baseline_mean = None
    print("[warn] Missing helpers from collision_utils; integrated orchestrator may be limited:", e)


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
    """Stateful, cached candidate finder for a single detect channel.

    This encapsulates the user's per-channel GMM candidate discovery with lazy reuse
    of cached splits and minimal recomputation after pool changes.
    """
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
        if per_channel_gmm_bimodality is None:
            raise RuntimeError("per_channel_gmm_bimodality is required for FinderSession")
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
        """Optional targeted refresh: if many of cand's spikes were just removed, recompute
        the split for this (chan, t) on the *current* pool only. Update cand_idx_global from
        current local labels → global via pool_ids_current.
        """
        if self.last_sub_idx_global is None:
            return cand
        c_best = cand.get('chan', None); t_best = cand.get('t', None)
        if c_best is None or t_best is None:
            return cand
        # How many of cand's original indices were touched by last removal?
        cg = cand.get('cand_idx_global', None)
        if cg is None:
            return cand
        touched = np.intersect1d(np.asarray(cg, dtype=np.int64), self.last_sub_idx_global, assume_unique=False).size
        denom = max(1, np.asarray(cg).size)
        frac = touched / denom
        if frac < 0.30:
            return cand
        # Recompute 2-GMM at (c_best, t_best) on CURRENT pool
        # Map: current local values at that (c,t)
        v_all = snips_current[int(c_best), int(t_best), :].reshape(-1, 1)
        try:
            gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0).fit(v_all)
            mu = gmm.means_.flatten()
            labels = gmm.predict(v_all)
            # pick stronger amplitude component (larger |mean|) consistent with polarity of ei_pool
            comp = np.argmax(np.abs(mu))
            cand_local_mask = (labels == comp)
            # convert to global ids
            new_global = pool_ids_current[np.where(cand_local_mask)[0]]
            cand['cand_idx_global'] = new_global.astype(np.int64)
        except Exception:
            # keep old cand if GMM fails
            pass
        return cand

    def next_candidate(self, snips_current: np.ndarray, pool_ids_current: np.ndarray,
                       cfg: OrchestratorConfig) -> Optional[Dict[str, Any]]:
        self.initialize()
        # Iterate through cached candidates; skip blacklisted or too small-after-alive
        for cid, cand in enumerate(self.cached_results):
            if cid in self.blacklist:
                continue
            cg = cand.get('cand_idx_global', None)
            if cg is None:
                continue
            # Intersect with alive
            cg = np.asarray(cg, dtype=np.int64)
            alive_ids = np.where(self.alive_global)[0]
            cg_alive = np.intersect1d(cg, alive_ids, assume_unique=False)
            if cg_alive.size < self.min_cluster_size:
                continue
            # Map to current local indices
            cand_mask_local = np.in1d(pool_ids_current, cg_alive, assume_unique=False)
            if cand_mask_local.sum() < self.min_cluster_size:
                continue
            # Optional targeted refresh if heavily touched by last removal
            cand = self._maybe_refresh_candidate(cand, snips_current, pool_ids_current)
            # Build EI and cheap prefilters (SNR)
            if median_ei_adaptive is None:
                raise RuntimeError("median_ei_adaptive required for candidate formation")
            snips_cand = snips_current[:, :, cand_mask_local]
            ei_cand = median_ei_adaptive(snips_cand)
            p2p = ei_cand.max(axis=1) - ei_cand.min(axis=1)
            snr = float(p2p.max() / max(np.median(p2p), 1e-9))
            if snr < 10.0:
                continue
            # Selected channels for evaluation (limit to MAX_CHANNELS by p2p threshold if desired)
            if cfg.P2P_THR is not None:
                chans = np.where(p2p >= cfg.P2P_THR)[0]
                if chans.size == 0:
                    chans = np.argsort(p2p)[-cfg.MIN_CHANNELS:]
            else:
                chans = np.argsort(p2p)[-cfg.MAX_CHANNELS:]
            if chans.size > cfg.MAX_CHANNELS:
                # keep strongest by p2p
                idx = np.argsort(p2p[chans])[::-1][:cfg.MAX_CHANNELS]
                chans = chans[idx]
            peak_ch = int(np.argmax(p2p))
            gbm = float(compute_global_baseline_mean(ei_cand)) if compute_global_baseline_mean else 0.0

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

    def __init__(self, raw_data: np.ndarray, peak_idx_per_chan: Dict[int, np.ndarray],
                 template_index: TemplateIndex, cfg: Optional[OrchestratorConfig] = None,
                 *, window: Tuple[int, int] = (-40, 80), edge_guard: int = 200,
                 max_n_per_chan: int = 10000, top_frac: float = 1.0):
        self.raw = raw_data
        self.peaks = peak_idx_per_chan
        self.index = template_index
        self.cfg = cfg or OrchestratorConfig()
        self.window = window
        self.edge_guard = int(edge_guard)
        self.max_n = int(max_n_per_chan)
        self.top_frac = float(top_frac)
        self.n_channels = raw_data.shape[1]

    # ----------------------- public entry point -----------------------
    def process_channel(self, Z: int) -> ChannelStats:
        Z = int(Z)
        snips_pool, spike_times_pool = self._build_local_pool(Z)
        N0 = snips_pool.shape[2]
        pool_ids = np.arange(N0, dtype=np.int64)

        # Finder session with cached candidates
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
                    # Map accepted times to template's home and merge
                    times_Z = spike_times_pool[acc]
                    times_home = map_spike_times_to_home(times_Z, Z, rec.home_channel, rec.tpeak)
                    self.index.update_spike_times(rec.id, times_home, tolerance=5, detect_ch=Z)

                    # Local peel + update pool_ids and finder
                    snips_pool, peeled_idx = peel_spikes_local(
                        snips_pool, rec.ei, acc, eval_res.best_lag_per_spike,
                        detect_channel=Z,
                        residual_sample=self.cfg.RESIDUAL_SAMPLE,
                        residual_keep_thr=self.cfg.residual_keep_thr if hasattr(self.cfg, 'residual_keep_thr') else self.cfg.RESIDUAL_KEEP_THR,
                    )
                    if peeled_idx.size > 0:
                        orig_ids = pool_ids[peeled_idx]
                        finder.note_removed(orig_ids)
                        keep_mask = np.ones(pool_ids.size, dtype=bool)
                        keep_mask[peeled_idx] = False
                        pool_ids = pool_ids[keep_mask]
                        spike_times_pool = spike_times_pool[keep_mask]
                        n_peeled += int(peeled_idx.size)

                    # Post-accept plots (reuse)
                    if self.cfg.SHOW_PLOTS:
                        title = f"[reuse t{rec.id} on ch {Z}] "
                        post_accept_plots(eval_result=eval_res, title_prefix=title, show_plots=True)

                # Early exit if pool sufficiently reduced
                if snips_pool.shape[2] <= (1.0 - self.cfg.PRESCREEN_STOP_FRACTION) * N0:
                    break

        N_prescreen = snips_pool.shape[2]

        # -------- Discovery on residual using cached finder --------
        n_new = 0
        n_alias = 0
        while snips_pool.shape[2] >= 50:  # same safety floor you use
            prop = finder.next_candidate(snips_pool, pool_ids, self.cfg)
            if prop is None:
                break

            # Build a temporary TemplateRecord for evaluation
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

            # Recompute EI on the accepted subset (final EI)
            snips_acc = snips_pool[:, :, acc]
            ei_final = median_ei_adaptive(snips_acc) if median_ei_adaptive is not None else prop['ei']
            p2p_final = ei_final.max(axis=1) - ei_final.min(axis=1)
            snr_final = float(p2p_final.max() / max(np.median(p2p_final), 1e-9))
            gbm_final = float(compute_global_baseline_mean(ei_final)) if compute_global_baseline_mean else prop['gbm']
            peak_ch_final = int(np.argmax(p2p_final))

            # Create a proper record in the prospective home timebase
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

            # Dedup / merge against existing templates
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
                    # Merge new spikes into existing
                    delta = int(rec_new.tpeak[rec_new.home_channel]) - int(matched.tpeak[matched.home_channel])
                    times_on_exist = rec_new.spike_times - np.int64(delta)
                    self.index.update_spike_times(matched.id, times_on_exist, tolerance=5, detect_ch=Z)
                    n_alias += 1
                else:
                    # Register new as canonical, merge existing spikes into it
                    self.index.register(rec_new)
                    delta = int(matched.tpeak[matched.home_channel]) - int(rec_new.tpeak[rec_new.home_channel])
                    times_on_new = matched.spike_times - np.int64(delta)
                    self.index.update_spike_times(rec_new.id, times_on_new, tolerance=5)
                    self.index.mark_alias(matched.id, rec_new.id)
                    n_new += 1
            else:
                # No match → register
                self.index.register(rec_new)
                n_new += 1

            # Local peel of accepted spikes and notify finder
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

            # Post-accept plots (final state)
            if self.cfg.SHOW_PLOTS:
                post_accept_plots(eval_result=eval_res,
                                  title_prefix=f"[discover ch {Z}] ",
                                  show_plots=True)

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
        top_idx = np.argsort(amp_vals)[:n_top]  # most negative first
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
