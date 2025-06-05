import numpy as np
from scipy.io import loadmat
from compute_sta_from_spikes import compute_sta_chunked
from benchmark_c_rgb_generation import RGBFrameGenerator


def sta_test(spike_times,
                     triggers_mat_path,
                     lut=None,
                     sta_depth=20,
                     sta_offset=0,
                     sta_chunk_size=1000,
                     sta_refresh=2):

    if lut is None:
        lut = np.array([
            [255, 255, 255],
            [255, 255,   0],
            [255,   0, 255],
            [255,   0,   0],
            [0,   255, 255],
            [0,   255,   0],
            [0,     0, 255],
            [0,     0,   0]
        ], dtype=np.uint8).flatten()

    triggers_sec = loadmat(triggers_mat_path)['triggers'].flatten()

    generator = RGBFrameGenerator('/Volumes/Lab/Users/alexth/axolotl/sta/libdraw_rgb.so')
    generator.configure(width=20, height=40, lut=lut, noise_type=1, n_bits=3)

    # STA
    sta = compute_sta_chunked(
        spikes_sec=spike_times,
        triggers_sec=triggers_sec,
        generator=generator,
        seed=11111,
        depth=sta_depth,
        offset=sta_offset,
        chunk_size=sta_chunk_size,
        refresh=sta_refresh
    )

    print(sta.shape)


