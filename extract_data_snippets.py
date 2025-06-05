import numpy as np

def extract_snippets(dat_path, spike_times, window=(-20, 60), n_channels=512, dtype='int16'):
    """
    Extract snippets of raw data around given spike times.
    
    Parameters:
        dat_path: Path to the .dat file
        spike_times: array of spike center times (in samples)
        window: tuple (pre, post) in samples
        n_channels: number of electrodes (default: 512)
        dtype: e.g., 'int16'
        
    Returns:
        snips: numpy array of shape [n_channels x snippet_len x num_spikes]
    """
    snip_len = window[1] - window[0] + 1
    spike_count = len(spike_times)
    snips = np.zeros((n_channels, snip_len, spike_count), dtype=np.float32)

    with open(dat_path, 'rb') as f:
        f.seek(0, 2)
        file_len_bytes = f.tell()
        total_samples = file_len_bytes // (np.dtype(dtype).itemsize * n_channels)

        for i, center in enumerate(spike_times):
            t_start = center + window[0]
            t_end = center + window[1]

            if t_start < 0 or t_end >= total_samples:
                continue  # skip invalid spikes

            offset = t_start * n_channels * np.dtype(dtype).itemsize
            f.seek(offset, 0)
            raw = np.fromfile(f, dtype=dtype, count=n_channels * snip_len)
            snips[:, :, i] = raw.reshape((snip_len, n_channels)).T  # shape: [channels x time]
    
    return snips


def extract_snippets_blockwise(dat_path, spike_times, window=(-20, 60),
                               n_channels=512, dtype='int16', block_size=100000):
    """
    Faster snippet extractor that reads data in large blocks and slices in-memory.

    Parameters:
        dat_path: Path to the .dat file
        spike_times: array of spike center times (in samples)
        window: tuple (pre, post) in samples
        n_channels: number of electrodes (default: 512)
        dtype: e.g., 'int16'
        block_size: number of samples to read in one block (e.g. 100,000)

    Returns:
        snips: [n_channels x snippet_len x num_valid_spikes] array
    """
    spike_times = np.array(spike_times)
    snip_len = window[1] - window[0] + 1
    spike_count = len(spike_times)

    # Estimate total samples in file
    with open(dat_path, 'rb') as f:
        f.seek(0, 2)
        file_len_bytes = f.tell()
    total_samples = file_len_bytes // (np.dtype(dtype).itemsize * n_channels)

    # Sort spikes to group by time
    spike_times_sorted = np.sort(spike_times)
    snips = []
    current_idx = 0

    with open(dat_path, 'rb') as f:
        while current_idx < spike_count:
            # Define current block range
            t_block_start = spike_times_sorted[current_idx] + window[0]
            t_block_end = t_block_start + block_size
            t_block_end = min(t_block_end, total_samples - 1)
            block_spikes = []

            # Collect spikes that fall within this block
            while (current_idx < spike_count and
                   spike_times_sorted[current_idx] + window[1] < t_block_end):
                block_spikes.append(spike_times_sorted[current_idx])
                current_idx += 1

            if len(block_spikes) == 0:
                continue

            t0 = block_spikes[0] + window[0]
            t1 = block_spikes[-1] + window[1]
            read_start = max(t0, 0)
            read_end = min(t1, total_samples - 1)

            n_samples = read_end - read_start + 1
            offset = read_start * n_channels * np.dtype(dtype).itemsize

            f.seek(offset, 0)
            raw = np.fromfile(f, dtype=dtype, count=n_channels * n_samples)
            block_data = raw.reshape((n_samples, n_channels)).T  # [C x T]

            for t in block_spikes:
                local_start = t + window[0] - read_start
                local_end = local_start + snip_len
                if local_start < 0 or local_end > block_data.shape[1]:
                    continue  # Skip if out of bounds
                snip = block_data[:, local_start:local_end]
                snips.append(snip)

    if len(snips) == 0:
        return np.zeros((n_channels, snip_len, 0), dtype=np.float32)
    
    return np.stack(snips, axis=2).astype(np.float32)
