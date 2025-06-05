import numpy as np
import torch

def run_template_scoring(dat_path, ei_template, ei_masks, ei_norms, selected_channels,
                         start_sample, total_samples, dtype='int16', block_size=1_000_000):

    n_channels = 512
    snippet_len = ei_template.shape[1]

    # Compute byte offset to start
    offset_bytes = start_sample * n_channels * np.dtype(dtype).itemsize

    with open(dat_path, 'rb') as f:
        f.seek(offset_bytes)
        raw_block = np.fromfile(f, dtype=dtype, count=n_channels * 1000)
        raw_block = raw_block.reshape((1000, n_channels)).T  # shape: [channels x time]
        baseline_block = raw_block[selected_channels, :]     # restrict to selected channels
        baseline_means = np.mean(baseline_block.astype(np.float32), axis=1)  # [channels]
        #print("Baseline means:", baseline_means[:10])

        # Preallocate score arrays
        n_timepoints_total = total_samples - snippet_len + 1
        mean_score = np.zeros(n_timepoints_total, dtype=np.float32)
        max_score = np.zeros(n_timepoints_total, dtype=np.float32)
        valid_score = np.zeros(n_timepoints_total, dtype=np.int32)

        # Masked template
        masked_template = ei_template * ei_masks  # shape: [channels x time]

        for b in range((total_samples + block_size - 1) // block_size):
            block_start = start_sample + b * block_size
            samples_in_block = min(block_size, total_samples - b * block_size)

            offset_bytes = block_start * n_channels * np.dtype(dtype).itemsize
            f.seek(offset_bytes)

            # Read and reshape the block
            raw = np.fromfile(f, dtype=dtype, count=n_channels * samples_in_block)
            block = raw.reshape((samples_in_block, n_channels)).T  # [channels x time]
            block = block[selected_channels, :]  # keep only selected
            block = block.astype(np.float32) - baseline_means[:, None]

            n_timepoints = block.shape[1] - snippet_len + 1
            block_scores_mean = np.zeros(n_timepoints, dtype=np.float32)
            block_scores_max = np.zeros(n_timepoints, dtype=np.float32)
            block_valid_count = np.zeros(n_timepoints, dtype=np.int32)

            # Convert block and templates to torch GPU tensors
            block_torch = torch.from_numpy(block).to('cuda')  # [channels x time]
            ei_masks_torch = torch.from_numpy(ei_masks).to('cuda')
            ei_norms_torch = torch.from_numpy(ei_norms).to('cuda')
            masked_template_torch = torch.from_numpy(masked_template).to('cuda')

            for t in range(n_timepoints):
                snippet = block_torch[:, t:t + snippet_len]
                masked_snippet = snippet * ei_masks_torch

                norms = torch.norm(masked_snippet, dim=1)
                scale = norms / ei_norms_torch
                valid = (scale > 0.5) & (scale < 2)

                if torch.any(valid):
                    dot_scores = torch.sum(masked_snippet[valid, :] * masked_template_torch[valid, :], dim=1)
                    block_scores_mean[t] = dot_scores.mean().item()
                    block_scores_max[t] = dot_scores.max().item()
                    block_valid_count[t] = valid.sum().item()


            # Store in preallocated arrays
            start_idx = b * block_size
            mean_score[start_idx:start_idx + n_timepoints] = block_scores_mean
            max_score[start_idx:start_idx + n_timepoints] = block_scores_max
            valid_score[start_idx:start_idx + n_timepoints] = block_valid_count

        return mean_score, max_score, valid_score
