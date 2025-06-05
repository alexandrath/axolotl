import numpy as np
import os

def convert_time_to_channel_major(
    dat_path_in,
    dat_path_out,
    n_channels,
    dtype=np.int16,
    chunk_size_time=1_000_000
):
    """
    Convert a time-major .dat file (T x C) to a channel-major layout (C x T).
    
    Args:
        dat_path_in (str): Path to input time-major .dat file.
        dat_path_out (str): Path to output channel-major .dat file.
        n_channels (int): Number of channels.
        dtype (np.dtype): Data type (default: np.int16).
        chunk_size_time (int): Number of time samples to read at once.
    """
    print(f"Converting {dat_path_in} to channel-major format...")
    
    # Determine total number of samples
    bytes_per_sample = np.dtype(dtype).itemsize
    file_size_bytes = os.path.getsize(dat_path_in)
    total_samples = file_size_bytes // bytes_per_sample
    total_timepoints = total_samples // n_channels

    print(f"  Total timepoints: {total_timepoints}")
    print(f"  Number of channels: {n_channels}")
    
    # Prepare output array
    data_ch_major = np.memmap(dat_path_out, mode='w+', dtype=dtype,
                               shape=(n_channels, total_timepoints))

    # Read in chunks of time
    with open(dat_path_in, 'rb') as f:
        for start_time in range(0, total_timepoints, chunk_size_time):
            n_time = min(chunk_size_time, total_timepoints - start_time)
            print(f"  Reading timepoints {start_time} to {start_time + n_time}...")
            raw = np.fromfile(f, dtype=dtype, count=n_time * n_channels)
            raw = raw.reshape((n_channels, n_time), order='F')
            data_ch_major[:, start_time:start_time + n_time] = raw

    print(f"Conversion complete. Saved to: {dat_path_out}")
