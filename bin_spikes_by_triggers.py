import numpy as np

def bin_spikes_matlab_style(spikes_sec, triggers_sec, refresh=2):
    """
    Spike binning to match MATLAB: all in seconds, double precision.

    Parameters:
        spikes_sec: spike times in seconds (1D numpy array)
        triggers_sec: trigger times in seconds (1D numpy array)
        refresh: monitor refresh interval per stimulus frame

    Returns:
        frame_indices: stimulus frame index for each spike (0-based, like MATLAB output minus 1)
    """
    spikes = spikes_sec.astype(np.float64)
    triggers = triggers_sec.astype(np.float64)

    # Align spikes relative to first trigger, same as: spikes{i} - triggers(1)
    spikes = spikes - triggers[0]
    spikes = spikes[spikes > 0]

    mean_trigger_interval = np.mean(np.diff(triggers))  # seconds per 100 monitor frames

    # Convert to stimulus frame index as floating point
    spike_frame_pos = spikes / mean_trigger_interval * (100 / refresh)

    # Digitize with 1-based edges, then shift to 0-based frame indices
    frame_indices = np.floor(spike_frame_pos - 0.45).astype(int)


    return frame_indices
