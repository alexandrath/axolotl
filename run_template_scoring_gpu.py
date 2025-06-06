import numpy as np
import torch

def run_template_scoring_gpu(dat_path, ei_template, ei_masks, ei_norms, selected_channels,
                             start_sample, GPU_samples, dtype='int16',
                             block_size=None, device=torch.device('cuda'),baseline_start_sample=0,
                             channel_major=False, raw_data=None):

    n_channels = 512
    snippet_len = ei_template.shape[1]

    # Compute byte offset to start
    #offset_bytes = start_sample * n_channels * np.dtype(dtype).itemsize

    # Estimate safe block size dynamically

    if block_size is None:
        torch.cuda.empty_cache()
        free_bytes = torch.cuda.mem_get_info(device)[0]
        safety_factor = 0.6
        nC = len(selected_channels)
        est_bytes_per_sample = nC * snippet_len * 4 * 3
        max_samples = int((free_bytes * safety_factor) // est_bytes_per_sample)
        block_size = min(max_samples, GPU_samples)

        # --- Read baseline block during auto mode ---
        if raw_data is not None:
            baseline_block = raw_data[baseline_start_sample:baseline_start_sample + 1000, selected_channels].T
        elif channel_major:
            data = np.memmap(dat_path, dtype=dtype, mode='r')
            data = data.reshape((n_channels, -1))
            baseline_block = data[selected_channels, baseline_start_sample:baseline_start_sample + 1000]
        else:
            with open(dat_path, 'rb') as f:
                baseline_offset_bytes = baseline_start_sample * n_channels * np.dtype(dtype).itemsize
                f.seek(baseline_offset_bytes)
                raw_block = np.fromfile(f, dtype=dtype, count=n_channels * 1000)
                raw_block = raw_block.reshape((1000, n_channels)).T
                baseline_block = raw_block[selected_channels, :]
    else:
        # --- Explicit block_size, but we still need baseline ---
        if raw_data is not None:
            baseline_block = raw_data[baseline_start_sample:baseline_start_sample + 1000, selected_channels].T
        elif channel_major:
            data = np.memmap(dat_path, dtype=dtype, mode='r')
            data = data.reshape((n_channels, -1))
            baseline_block = data[selected_channels, baseline_start_sample:baseline_start_sample + 1000]
        else:
            with open(dat_path, 'rb') as f:
                baseline_offset_bytes = baseline_start_sample * n_channels * np.dtype(dtype).itemsize
                f.seek(baseline_offset_bytes)
                raw_block = np.fromfile(f, dtype=dtype, count=n_channels * 1000)
                raw_block = raw_block.reshape((1000, n_channels)).T
                baseline_block = raw_block[selected_channels, :]

    baseline_means = np.mean(baseline_block.astype(np.float32), axis=1)

    #with open(dat_path, 'rb') as f:
    #    baseline_offset_bytes = baseline_start_sample * n_channels * np.dtype(dtype).itemsize
    #    f.seek(baseline_offset_bytes)
    #    raw_block = np.fromfile(f, dtype=dtype, count=n_channels * 1000)
    #    raw_block = raw_block.reshape((1000, n_channels)).T
    #    baseline_block = raw_block[selected_channels, :]
    #    baseline_means = np.mean(baseline_block.astype(np.float32), axis=1)

        # Preallocate full output arrays (CPU)
    n_timepoints_total = GPU_samples - snippet_len + 1
    mean_score = np.zeros(n_timepoints_total, dtype=np.float32)
    max_score = np.zeros(n_timepoints_total, dtype=np.float32)
    valid_score = np.zeros(n_timepoints_total, dtype=np.int32)

    # Move static data to GPU
    ei_masks_torch = torch.from_numpy(ei_masks).to(device)                   # [C x T]
    ei_norms_torch = torch.from_numpy(ei_norms).to(device)                   # [C]
    masked_template_torch = torch.from_numpy(ei_template * ei_masks).to(device)  # [C x T]

    for b in range((GPU_samples + block_size - 1) // block_size):
        block_start = start_sample + b * block_size
        samples_remaining = GPU_samples - b * block_size
        samples_in_block = min(block_size, samples_remaining)


        if raw_data is not None:
            block = raw_data[block_start:block_start + samples_in_block, selected_channels].T
        elif channel_major:
            block = data[selected_channels, block_start:block_start + samples_in_block]
        else:
            with open(dat_path, 'rb') as f:
                offset_bytes = block_start * n_channels * np.dtype(dtype).itemsize
                f.seek(offset_bytes)
                raw = np.fromfile(f, dtype=dtype, count=n_channels * samples_in_block)
                block = raw.reshape((samples_in_block, n_channels)).T
                block = block[selected_channels, :]


        block = block.astype(np.float32) - baseline_means[:, None]  # [C x T]

        block_torch = torch.from_numpy(block).to(device)  # [C x T]
        if block_torch.shape[1] < snippet_len:
            continue  # skip block too small to process

        n_timepoints = block_torch.shape[1] - snippet_len + 1

        # Extract all timepoint snippets as [C x T x n_timepoints]
        snippets = block_torch.unfold(1, snippet_len, 1)  # shape: [C x n_timepoints x T]
        snippets = snippets.permute(0, 2, 1)              # [C x T x n_timepoints]

        # Masked snippets
        masked_snippets = snippets * ei_masks_torch[:, :, None]  # [C x T x n_timepoints]

        # Compute scale factors
        norms = torch.norm(masked_snippets, dim=1)        # [C x n_timepoints]
        scale = norms / ei_norms_torch[:, None]           # [C x n_timepoints]
        valid = (scale > 0.5) & (scale < 2.0)              # [C x n_timepoints]

        # Compute dot products
        dot = masked_snippets * masked_template_torch[:, :, None]  # [C x T x n_timepoints]
        dot_scores = dot.sum(dim=1)                                # [C x n_timepoints]

        # Mask invalid channels
        dot_scores[~valid] = float('nan')

        # Reduce across valid channels
        mean_dot = torch.nanmean(dot_scores, dim=0)        # [n_timepoints]
        # Replace NaNs with -inf before max
        dot_scores_maxsafe = dot_scores.clone()
        dot_scores_maxsafe[torch.isnan(dot_scores_maxsafe)] = float('-inf')
        max_dot = torch.max(dot_scores_maxsafe, dim=0).values

        valid_count = valid.sum(dim=0).to(torch.int32)     # [n_timepoints]

        # Save back to CPU
        start_idx = b * block_size
        end_idx = start_idx + n_timepoints
        mean_score[start_idx:end_idx] = mean_dot.cpu().numpy()
        max_score[start_idx:end_idx] = max_dot.cpu().numpy()
        valid_score[start_idx:end_idx] = valid_count.cpu().numpy()

    return mean_score, max_score, valid_score
