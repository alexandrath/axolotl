import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torch
import time
import ctypes

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


def benchmark_c_rgb_generation(generator, seed, n_frames):
    start = time.time()
    _, seed = generator.draw_frames_batch(seed, n_frames)
    elapsed = time.time() - start
    print(f"Generated {n_frames} frames using C batch in {elapsed:.6f} seconds")
    return elapsed

def generate_and_plot_test_frames(seed, stim_size, noise_type, n_bits, lut, num_frames=5):
    generator = RGBFrameGenerator('/Volumes/Lab/Users/alexth/axolotl/sta/libdraw_rgb.so')
    generator.configure(stim_size[0], stim_size[1], lut, noise_type, n_bits)
    frames, _ = generator.draw_frames_batch(seed, num_frames)

    # Normalize to [0, 1] for plotting
    frames = (frames.astype(np.float32) / 255.0)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, num_frames, figsize=(15, 5))
    for i, ax in enumerate(axes):
        ax.imshow(frames[i].transpose(1, 0, 2))  # transpose to W x H x 3
        ax.set_title(f'Frame {i}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    return frames
