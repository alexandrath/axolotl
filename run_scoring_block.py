import os
import numpy as np
import torch
from run_template_scoring_gpu import run_template_scoring_gpu

def run_scoring_block(gpu_id, start_sample, GPU_samples, save_prefix,
                      dat_path, ei_template, ei_masks, ei_norms,
                      selected_channels, dtype, block_size, baseline_start_sample, channel_major=False):
    """
    Run template scoring on a GPU chunk with optional trimming at edges.
    """

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda:0')  # local GPU index 0 due to env pinning

    #print(f"[GPU {gpu_id}] Block: {start_sample} → {start_sample + total_samples}")
    #print(f"[GPU {gpu_id}] Using device: {torch.cuda.get_device_name(device)}")

    mean_score, max_score, valid_score = run_template_scoring_gpu(
        dat_path=dat_path,
        ei_template=ei_template,
        ei_masks=ei_masks,
        ei_norms=ei_norms,
        selected_channels=selected_channels,
        start_sample=start_sample,
        GPU_samples=GPU_samples,
        dtype=dtype,
        block_size=block_size,
        device=device,
        baseline_start_sample=baseline_start_sample,
        channel_major=channel_major
    )

    #print(f"[GPU {gpu_id}] Final score array length: {len(mean_score)}")


    out_path = f"{save_prefix}_gpu{gpu_id}.npy"
    np.save(out_path, {
        'mean': mean_score,
        'max': max_score,
        'valid': valid_score
    })

    #print(f"[GPU {gpu_id}] Done. Saved to {out_path}")
