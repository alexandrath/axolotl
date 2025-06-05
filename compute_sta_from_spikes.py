import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torch
import time
import ctypes
import scipy.io as sio
from bin_spikes_by_triggers import bin_spikes_matlab_style

# Define wrapper for calling C shared library
class RGBFrameGenerator:
    def __init__(self, lib_path):
        self.lib = ctypes.cdll.LoadLibrary(lib_path)
        self.lib.draw_rgb_frame.argtypes = [
            ctypes.c_longlong,               # seed
            ctypes.c_int, ctypes.c_int,      # width, height
            ctypes.POINTER(ctypes.c_ubyte),  # lut
            ctypes.c_int, ctypes.c_int,      # noise_type, n_bits
            ctypes.POINTER(ctypes.c_ubyte),  # output buffer
            ctypes.POINTER(ctypes.c_longlong) # updated seed
        ]
        self.lib.draw_rgb_frame_batch.argtypes = [
            ctypes.c_longlong, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_longlong)
        ]
        self.width = None
        self.height = None
        self.lut = None
        self.noise_type = None
        self.n_bits = None

    def configure(self, width, height, lut, noise_type, n_bits):
        self.width = width
        self.height = height
        self.lut = (ctypes.c_ubyte * len(lut))(*lut)
        self.noise_type = noise_type
        self.n_bits = n_bits

    def draw_frame(self, seed):
        size = self.width * self.height * 3
        output = (ctypes.c_ubyte * size)()
        seed_in = ctypes.c_longlong(seed)
        seed_out = ctypes.c_longlong()

        self.lib.draw_rgb_frame(
            seed_in,
            self.width, self.height,
            self.lut,
            self.noise_type, self.n_bits,
            output,
            ctypes.byref(seed_out)
        )

        frame = np.ctypeslib.as_array(output).reshape((self.width, self.height, 3)).transpose(1, 0, 2)
        return frame, seed_out.value

    def draw_frames_batch(self, seed, n_frames):
        size = self.width * self.height * 3 * n_frames
        output = (ctypes.c_ubyte * size)()
        seed_in = ctypes.c_longlong(seed)
        seed_out = ctypes.c_longlong()

        self.lib.draw_rgb_frame_batch(
            seed_in,
            self.width, self.height,
            self.lut,
            self.noise_type, self.n_bits,
            n_frames,
            output,
            ctypes.byref(seed_out)
        )

        frames = np.ctypeslib.as_array(output).reshape((n_frames, self.width, self.height, 3)).transpose(0, 2, 1, 3)
        return frames, seed_out.value



def compute_sta_chunked(spikes_sec, triggers_sec, generator, seed, depth,
                         offset, chunk_size=1000, refresh=2):
    frame_indices = bin_spikes_matlab_style(
        spikes_sec=spikes_sec,
        triggers_sec=triggers_sec,
        refresh=refresh
    )

    total_frames = np.max(frame_indices) + offset + 1

    h, w = generator.height, generator.width
    sta = np.zeros((h, w, 3, depth), dtype=np.float32)
    spike_counts = np.zeros(depth, dtype=np.int32)
    current_seed = seed

    # Pre-generate all stimulus frames
    all_frames = []
    for chunk_start in range(0, total_frames, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_frames)
        n_frames = chunk_end - chunk_start

        #print(f"Chunk {chunk_start}-{chunk_end}, seed in: {current_seed}")
        frames, current_seed = generator.draw_frames_batch(current_seed, n_frames)
        #print(f"Chunk {chunk_start}-{chunk_end}, updated seed: {current_seed}")


        # Normalize to [0, 1] for plotting
        frames = (frames.astype(np.float32) / 127.5) - 1.0
        all_frames.append(frames)

    frames = np.concatenate(all_frames, axis=0)  # shape: (total_frames, H, W, 3)

    #print(frame_indices)

    # Spike-based STA accumulation
    for spike_f in frame_indices:
        for d in range(depth):
            stim_f = spike_f - d + offset
            if 0 <= stim_f < total_frames:
                sta[:, :, :, d] += frames[stim_f]
                spike_counts[d] += 1

    # Normalize
    for d in range(depth):
        if spike_counts[d] > 0:
            sta[:, :, :, d] /= spike_counts[d]

    return sta
