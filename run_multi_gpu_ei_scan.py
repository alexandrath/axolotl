import os
import numpy as np
import torch
import multiprocessing as mp
from scipy.io import loadmat
from run_scoring_block import run_scoring_block

def run_multi_gpu_ei_scan(ei_mat_path, dat_path, total_samples,
                          save_prefix='/tmp/ei_scan', dtype='int16',
                          block_size=None, baseline_start_sample=0,
                          channel_major=False):
    """
    Run EI template matching on a .dat file using multiple GPUs.

    Parameters:
        ei_mat_path : path to .mat file containing 'ei_template'
        dat_path : path to .dat file
        total_samples : number of samples to scan
        save_prefix : where to save intermediate GPU score files
        dtype : datatype of .dat file
        block_size : override GPU block size (optional)
        baseline_start_sample : start sample for estimating baseline (optional)

    Returns:
        mean_score, max_score, valid_score : arrays of match metrics
    """
    # --- Load and preprocess EI template ---
    ei_template = loadmat(ei_mat_path)['ei_template'].astype(np.float32)
    ei_template -= ei_template[:, :5].mean(axis=1, keepdims=True)

    # --- Select channels based on peak-to-peak range ---
    ei_peak = ei_template.max(axis=1) - ei_template.min(axis=1)
    sorted_idx = np.argsort(ei_peak)
    if np.sum(ei_peak > 0) >= 80:
        selected = sorted_idx[-80:]
    else:
        selected = sorted_idx[-30:]
    selected = np.sort(selected)
    ei_template_sel = ei_template[selected, :]

    # --- Prepare masks and norms ---
    mask = np.zeros_like(ei_template)
    mask[selected, :] = 1
    ei_masks = mask[selected, :]
    ei_norms = np.linalg.norm(ei_template_sel * ei_masks, axis=1)

    mp.set_start_method('spawn', force=True)

    # --- Run on multiple GPUs ---
    n_gpus = torch.cuda.device_count()

    snippet_len = ei_template.shape[1]

    n_timepoints = total_samples - snippet_len + 1
    samples_per_gpu = n_timepoints // n_gpus
    remainder = n_timepoints % n_gpus
    processes = []
    for gpu_id in range(n_gpus):
        score_start = gpu_id * samples_per_gpu + min(gpu_id, remainder)
        score_len = samples_per_gpu + (1 if gpu_id < remainder else 0)
        score_end = score_start + score_len

        sample_start = score_start
        sample_end = score_end + snippet_len - 1
        GPU_samples = sample_end - sample_start

        p = mp.Process(target=run_scoring_block, args=(
            gpu_id, sample_start, GPU_samples, save_prefix, dat_path,
            ei_template_sel, ei_masks, ei_norms, selected,
            dtype, block_size, baseline_start_sample,channel_major
        ))
        p.start()
        processes.append(p)



    # samples_per_gpu = total_samples // n_gpus
    # snippet_len = ei_template.shape[1]
    # processes = []
    # for gpu_id in range(n_gpus):
    #     start = gpu_id * samples_per_gpu
    #     samples = total_samples - start if gpu_id == n_gpus - 1 else samples_per_gpu + (snippet_len - 1)
    #     #samples = total_samples - start if gpu_id == n_gpus - 1 else samples_per_gpu

    #     p = mp.Process(target=run_scoring_block, args=(
    #         gpu_id, start, samples, save_prefix, dat_path,
    #         ei_template_sel, ei_masks, ei_norms, selected,
    #         dtype, block_size, baseline_start_sample,channel_major
    #     ))
    #     p.start()
    #     processes.append(p)

    for p in processes:
        p.join()

    # --- Merge outputs ---
    n_timepoints = total_samples - snippet_len + 1
    mean_score = np.zeros(n_timepoints, dtype=np.float32)
    max_score = np.zeros(n_timepoints, dtype=np.float32)
    valid_score = np.zeros(n_timepoints, dtype=np.int32)

    for gpu_id in range(n_gpus):
        part = np.load(f"{save_prefix}_gpu{gpu_id}.npy", allow_pickle=True).item()
        score_start = gpu_id * samples_per_gpu + min(gpu_id, remainder)
        start = score_start
        end = start + score_len
        mean_score[start:end] = part['mean']

        # start = gpu_id * samples_per_gpu
        # end = start + len(part['mean'])
        # mean_score[start:end] = part['mean']
        max_score[start:end] = part['max']
        valid_score[start:end] = part['valid']

    return mean_score, max_score, valid_score, selected, ei_template
