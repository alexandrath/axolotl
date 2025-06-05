import numpy as np
import torch

def run_template_scoring_gpu_minibatch(dat_path, ei_template, ei_masks, ei_norms, selected_channels,
                                       start_sample, total_samples, dtype='int16',
                                       block_size=1_000_000, minibatch_size=10_000,
                                       device=torch.device('cuda:0')):

    n_channels = 512
    snippet_len = ei_template.shape[1]

    # Compute byte offset
    offset_bytes = start_sample * n_channels * np.dtype(dtype).itemsize

    with open(dat_path, 'rb') as f:
        f.seek(offset_bytes)
        raw_block = np.fromfile(f, dtype=dtype, count=n_channels * 1000)
        raw_block = raw_block.reshape((1000, n_channels)).T
        baseline_block = raw_block[selected_channels, :]
        baseline_means = np.mean(baseline_block.astype(np.float32), axis=1)

        # Preallocate
        n_timepoints_total = total_samples - snippet_len + 1
        mean_score = np.zeros(n_timepoints_total, dtype=np.float32)
        max_score = np.zeros(n_timepoints_total, dtype=np.float32)
        valid_score = np.zeros(n_timepoints_total, dtype=np.int32)

        # Preload static tensors to GPU
        ei_masks_t = torch.from_numpy(ei_masks).to(device)
        ei_norms_t = torch.from_numpy(ei_norms).to(device)
        masked_template_t = torch.from_numpy(ei_template * ei_masks).to(device)

        for b in range((total_samples + block_size - 1) // block_size):
            block_start = start_sample + b * block_size
            samples_in_block = min(block_size, total_samples - b * block_size)

            offset_bytes = block_start * n_channels * np.dtype(dtype).itemsize
            f.seek(offset_bytes)
            raw = np.fromfile(f, dtype=dtype, count=n_channels * samples_in_block)
            block = raw.reshape((samples_in_block, n_channels)).T
            block = block[selected_channels, :].astype(np.float32)
            block -= baseline_means[:, None]

            n_timepoints = block.shape[1] - snippet_len + 1
            block_mean = np.zeros(n_timepoints, dtype=np.float32)
            block_max = np.zeros(n_timepoints, dtype=np.float32)
            block_valid = np.zeros(n_timepoints, dtype=np.int32)

            # Slide over minibatches
            for t_start in range(0, n_timepoints, minibatch_size):
                t_end = min(t_start + minibatch_size, n_timepoints)
                mb_len = t_end - t_start

                # Build minibatch snippets: [C x T x mb]
                snippets = np.stack([
                    block[:, t:t+snippet_len] for t in range(t_start, t_end)
                ], axis=2)  # [C x T x mb]

                snippets_t = torch.from_numpy(snippets).to(device)
                masked_snips = snippets_t * ei_masks_t[:, :, None]

                norms = torch.norm(masked_snips, dim=1)                     # [C x mb]
                scale = norms / ei_norms_t[:, None]
                valid = (scale > 0.5) & (scale < 2.0)

                dot = masked_snips * masked_template_t[:, :, None]
                dot_scores = dot.sum(dim=1)                                # [C x mb]
                dot_scores[~valid] = float('nan')

                mean = torch.nanmean(dot_scores, dim=0).cpu().numpy()
                dot_scores_safe = dot_scores.clone()
                dot_scores_safe[torch.isnan(dot_scores_safe)] = float('-inf')
                max_ = torch.max(dot_scores_safe, dim=0).values.cpu().numpy()

                count = valid.sum(dim=0).cpu().numpy()

                block_mean[t_start:t_end] = mean
                block_max[t_start:t_end] = max_
                block_valid[t_start:t_end] = count.astype(np.int32)

                # Free GPU memory explicitly
                del snippets_t, masked_snips, norms, scale, valid, dot, dot_scores
                torch.cuda.empty_cache()

            # Store to full arrays
            start_idx = b * block_size
            end_idx = start_idx + n_timepoints
            mean_score[start_idx:end_idx] = block_mean
            max_score[start_idx:end_idx] = block_max
            valid_score[start_idx:end_idx] = block_valid

        return mean_score, max_score, valid_score
