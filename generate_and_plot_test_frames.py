import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

class JavaRandom:
    def __init__(self, seed):
        self.seed = (seed ^ 0x5DEECE66D) & ((1 << 48) - 1)

    def next_nbit(self, n):
        self.seed = (self.seed * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        if n == 1:
            return self.seed >> 47
        elif n == 3:
            return self.seed >> 45
        elif n == 8:
            return self.seed >> 40
        else:
            return self.seed >> 32

def draw_random_frame(rand_gen, stim_size, noise_type, n_bits, lut):
    frame = np.zeros(stim_size, dtype=np.uint8)
    w, h, _ = stim_size

    if noise_type == 0:  # binary BW
        for y in range(h):
            for x in range(w):
                val = rand_gen.next_nbit(n_bits)
                base_idx = (val % 2) * 3
                rgb = lut[base_idx:base_idx+3]
                frame[x, y, :] = rgb

    elif noise_type == 1:  # binary RGB
        for y in range(h):
            for x in range(w):
                val = rand_gen.next_nbit(n_bits)
                base_idx = (val % 8) * 3
                rgb = lut[base_idx:base_idx+3]
                frame[x, y, :] = rgb

    elif noise_type == 2:  # gaussian BW
        for y in range(h):
            for x in range(w):
                val = rand_gen.next_nbit(n_bits)
                base_idx = (val % 256) * 3
                rgb = lut[base_idx:base_idx+3]
                frame[x, y, :] = rgb

    elif noise_type == 3:  # gaussian RGB
        for y in range(h):
            for x in range(w):
                for c in range(3):
                    val = rand_gen.next_nbit(n_bits)
                    frame[x, y, c] = lut[(val % 256) * 3 + c]

    return frame

def generate_and_plot_test_frames(seed, stim_size, noise_type, n_bits, lut, num_frames=5):
    rand = JavaRandom(seed)
    frames = []
    for i in range(num_frames):
        frame = draw_random_frame(rand, stim_size, noise_type, n_bits, lut)
        frames.append(frame)

    fig, axes = plt.subplots(1, num_frames, figsize=(15, 5))
    for i, ax in enumerate(axes):
        ax.imshow(frames[i].transpose(1, 0, 2))
        ax.set_title(f'Frame {i}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    return frames
