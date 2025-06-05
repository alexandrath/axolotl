import numpy as np
import torch

def run_template_scoring_gpu(dat_path, ei_template, ei_masks, ei_norms, selected_channels,
                             start_sample, GPU_samples, dtype='int16',
                             block_size=None, device=torch.device('cuda'),baseline_start_sample=0,
                             channel_major=False):

    n_channels = 512
    snippet_len = ei_template.shape[1]

    # Compute byte offset to start
    #offset_bytes = start_sample * n_channels * np.dtype(dtype).itemsize

        # Estimate safe block size dynamically
    if block_size is None:
        #torch.cuda.empty_cache()
        #stats = torch.cuda.mem_get_info(device)
        #free_mem = stats[0]

        torch.cuda.empty_cache()
        free_bytes = torch.cuda.mem_get_info(device)[0]
        safety_factor = 0.6  # use at most 60% of free memory
        nC = len(selected_channels)
        est_bytes_per_sample = nC * snippet_len * 4 * 3  # 3x float32 arrays
        max_samples = int((free_bytes * safety_factor) // est_bytes_per_sample)
        block_size = min(max_samples, GPU_samples)
        # print(f"Auto block_size set to {block_size}")

        if channel_major:
            data = np.memmap(dat_path, dtype=dtype, mode='r')
            data = data.reshape((n_channels, -1))  # reshape based on file size

            #data = np.memmap(dat_path, dtype=dtype, mode='r', shape=(n_channels, total_samples))
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
        #samples_in_block = min(block_size, total_samples - b * block_size)

        samples_remaining = GPU_samples - b * block_size
        samples_in_block = min(block_size, samples_remaining)


        if channel_major:
            block = data[selected_channels, block_start:block_start + samples_in_block]

            # if device.index == 0 and b == 0:
            #     print("DEBUG trace")
            #     ch_idx = np.where(selected_channels == 39)[0][0]
            #     print("Block start:", block_start)
            #     print("Selected ch idx for 39:", ch_idx)
            #     print("Block[ch_idx, :5]:", block[ch_idx, :5])
            #     print("Direct memmap[39, block_start:block_start+5]:", data[39, block_start:block_start+5])

            # if device.index == 0:
            #     import matplotlib.pyplot as plt

            #     if 39 in selected_channels:
            #         ch_idx = np.where(selected_channels == 39)[0][0]
            #         trace = block[ch_idx, :5000]

            #         plt.figure(figsize=(12, 3))
            #         plt.plot(trace, color='black')
            #         plt.title(f"GPU 3 - Block {b} - Channel 39 (first 5000 samples)")
            #         plt.xlabel("Sample")
            #         plt.ylabel("Amplitude")
            #         plt.grid(True)
            #         plt.tight_layout()
            #         plt.savefig(f"/Volumes/Lab/Users/alexth/axolotl/debug/gpu3_block{b}_ch39.png", dpi=150)
            #         plt.close()
            #     else:
            #         print("Channel 39 not in selected_channels for this block.")


        else:
            with open(dat_path, 'rb') as f:
                offset_bytes = block_start * n_channels * np.dtype(dtype).itemsize
                f.seek(offset_bytes)
                raw = np.fromfile(f, dtype=dtype, count=n_channels * samples_in_block)
                block = raw.reshape((samples_in_block, n_channels)).T
                block = block[selected_channels, :]

        #offset_bytes = block_start * n_channels * np.dtype(dtype).itemsize
        #f.seek(offset_bytes)

        #raw = np.fromfile(f, dtype=dtype, count=n_channels * samples_in_block)
        #block = raw.reshape((samples_in_block, n_channels)).T
        #block = block[selected_channels, :]
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
