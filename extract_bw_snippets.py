import numpy as np
from bin_spikes_by_triggers import bin_spikes_matlab_style


def extract_bw_snippets(spikes_sec, triggers_sec, generator, seed, rf_pixels,
                        weights_rgb, depth=15, offset=0, chunk_size=1000, refresh=2):
    """
    Extracts grayscale stimulus snippets aligned to spike times, using a custom RGB stimulus generator.

    Args:
        spikes_sec: array-like, spike times in seconds
        triggers_sec: array-like, trigger times in seconds for each stimulus frame
        generator: RGBFrameGenerator object, already configured
        seed: int, starting seed for deterministic stimulus generation
        rf_pixels: list of (x, y) pixel coordinates to extract (x = col, y = row)
        weights_rgb: array-like of shape (3,), RGB → BW weights (e.g. normalized STA peak)
        depth: int, number of frames to extract before each spike
        offset: int, offset relative to spike frame (default = 0)
        chunk_size: int, number of stimulus frames to generate at a time
        refresh: int, number of monitor refresh frames per stimulus frame (e.g. 2 for 60Hz on 120Hz display)

    Returns:
        snippets: np.ndarray of shape [N_spikes, depth, N_pixels]
    """
    frame_indices = bin_spikes_matlab_style(
        spikes_sec=np.asarray(spikes_sec, dtype=float),
        triggers_sec=np.asarray(triggers_sec, dtype=float),
        refresh=refresh
    )
    total_frames = np.max(frame_indices) + offset + 1

    H, W = generator.height, generator.width
    current_seed = seed

    # Generate all stimulus frames in chunks
    frames_buffer = []
    for chunk_start in range(0, total_frames, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_frames)
        n_frames = chunk_end - chunk_start

        frames, current_seed = generator.draw_frames_batch(current_seed, n_frames)
        frames = (frames.astype(np.float32) / 127.5) - 1.0  # Normalize to [-1, 1]
        frames_buffer.append(frames)

    # Combine into full stimulus sequence: [T, H, W, 3]
    full_frames = np.concatenate(frames_buffer, axis=0)

    # Convert relevant RF pixels to BW using weights
    rf_signals = []
    for (x, y) in rf_pixels:
        if not (0 <= x < W and 0 <= y < H):
            raise ValueError(f"Invalid pixel coordinate: ({x}, {y}) outside stimulus bounds")
        pixel_rgb = full_frames[:, y, x, :]         # shape: [T, 3]
        pixel_bw = pixel_rgb @ weights_rgb          # shape: [T,]
        rf_signals.append(pixel_bw)

    rf_signals = np.stack(rf_signals, axis=1)  # shape: [T, N_pixels]

    # Extract aligned snippets per spike
    snippets = []
    for spike_f in frame_indices:
        start = spike_f - depth + offset
        end = spike_f + offset
        if start >= 0 and end <= rf_signals.shape[0]:
            snippet = rf_signals[start:end, :]  # shape: [depth, N_pixels]
            snippets.append(snippet)

    if not snippets:
        raise RuntimeError("No valid snippets extracted. Check depth/offset and spike/frame alignment.")

    return np.stack(snippets)  # shape: [N_spikes, depth, N_pixels]
