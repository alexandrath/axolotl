import numpy as np
import h5py
import torch
from extract_data_snippets import extract_snippets
from refine_cluster import refine_cluster
from run_template_scoring_gpu import run_template_scoring_gpu
from compare_eis import compare_eis

def process_unit(cell_id, spike_times, dat_path, h5_out_path, params):
    print(f"[Unit {cell_id}] Processing...")

    # Open output file in append mode
    with h5py.File(h5_out_path, 'a') as h5:
        group = h5.require_group(f"unit_{cell_id}")

        # Skip if too few spikes
        if len(spike_times) < 100:
            print(f"[Unit {cell_id}] Skipped (only {len(spike_times)} spikes)")
            group.create_dataset('inds', data=np.array([len(spike_times)], dtype=np.int32))
            return

        # Step 1: Extract EI template
        snips = extract_snippets(
            dat_path=dat_path,
            spike_times=spike_times,
            window=(-20, 60),
            n_channels=512,
            dtype='int16'
        )
        ei_template = np.mean(snips, axis=2)
        ei_template = ei_template - ei_template[:, :5].mean(axis=1, keepdims=True)

        # Step 2: Select channels
        ei_peak2peak = ei_template.max(axis=1) - ei_template.min(axis=1)
        selected = np.where(ei_peak2peak > params['amplitude_threshold'])[0]
        if len(selected) > 80:
            top = np.argsort(ei_peak2peak)[-80:]
            selected = np.sort(top)
        elif len(selected) < 30:
            top = np.argsort(ei_peak2peak)[-30:]
            selected = np.sort(top)
        selected_channels = selected.astype(np.int32)

        # Save selected channels
        group.create_dataset('selected_channels', data=selected_channels)

        # Step 3: EI mask and norms
        mask = np.zeros_like(ei_template)
        mask[selected_channels, :] = 1
        ei_norms = np.linalg.norm(ei_template * mask, axis=1)

        # Step 4: Run template scoring on GPU
        mean_score, max_score, valid_score = run_template_scoring_gpu(
            dat_path=dat_path,
            ei_template=ei_template[selected_channels, :],
            ei_masks=mask[selected_channels, :],
            ei_norms=ei_norms[selected_channels],
            selected_channels=selected_channels,
            start_sample=0,
            total_samples=params['total_samples'],
            dtype='int16',
            block_size=None,
            device=torch.device('cuda:0')  # will be overridden in multi-GPU mode
        )

        # Step 5: Threshold selection
        masked_template = ei_template[selected_channels, :] * mask[selected_channels, :]
        dot_scores = np.sum(masked_template ** 2, axis=1)
        score_threshold = max(np.mean(dot_scores) / 2, 10000)
        from scipy.signal import argrelextrema
        peaks = argrelextrema(mean_score, np.greater_equal, order=1)[0]
        valid_inds = peaks[(mean_score[peaks] > score_threshold) & (valid_score[peaks] > 3)]
        final_spike_times = valid_inds + 19

        if len(final_spike_times) == 0:
            print(f"[Unit {cell_id}] No matches above threshold")
            group.create_dataset('inds', data=np.array([], dtype=np.int32))
            return

        # Step 6: Extract matching snippets
        snips_final = extract_snippets(
            dat_path=dat_path,
            spike_times=final_spike_times,
            window=(-20, 60),
            n_channels=512,
            dtype='int16'
        )

        # Step 7: Refine
        refined_inds = refine_cluster(
            snips=snips_final,
            ei_template=ei_template,
            selected_by_ei=selected_channels,
            k=8,
            inds=None,
            depth=0,
            threshold=params['similarity_threshold']
        )

        # Save final result
        inds_to_save = final_spike_times[refined_inds] if len(refined_inds) > 0 else np.array([], dtype=np.int32)
        group.create_dataset('inds', data=inds_to_save)

        print(f"[Unit {cell_id}] Done. Found {len(inds_to_save)} matches.")
