# sta_grouped_by_offset.py  — Python 3.9 compatible

import numpy as np
import ctypes
from bin_spikes_by_triggers import (
    bin_spikes_matlab_style_robust_autorepair as bin_spikes,
)

# -------- C-library frame generator (stixel-res) --------

class RGBFrameGenerator:
    def __init__(self, lib_path):
        self.lib = ctypes.cdll.LoadLibrary(lib_path)
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
        self.width = int(width)
        self.height = int(height)
        lut = np.asarray(lut, dtype=np.uint8).ravel()
        self.lut = (ctypes.c_ubyte * len(lut))(*lut)
        self.noise_type = int(noise_type)
        self.n_bits = int(n_bits)

    def draw_frames_batch(self, seed, n_frames):
        n_frames = int(n_frames)
        size = self.width * self.height * 3 * n_frames
        output = (ctypes.c_ubyte * size)()
        seed_in = ctypes.c_longlong(int(seed))
        seed_out = ctypes.c_longlong()
        self.lib.draw_rgb_frame_batch(
            seed_in,
            self.width, self.height,
            self.lut, self.noise_type, self.n_bits,
            n_frames, output, ctypes.byref(seed_out)
        )
        arr = np.ctypeslib.as_array(output)
        frames = arr.reshape((n_frames, self.width, self.height, 3)).transpose(0, 2, 1, 3)
        return frames, int(seed_out.value)

# -------- Java-style 48-bit LCG jitter (Photons-compatible) --------

_A = 0x5DEECE66D
_C = 0xB
_MASK48 = (1 << 48) - 1

def _java_init_seed(seed):
    return (int(seed) ^ _A) & _MASK48

def _lcg_combine(a, c, steps):
    mul, add = 1, 0
    while steps > 0:
        if steps & 1:
            add = (add * a + c) & _MASK48
            mul = (mul * a) & _MASK48
        c = (c * (a + 1)) & _MASK48
        a = (a * a) & _MASK48
        steps >>= 1
    return mul, add

def _lcg_advance(state, steps):
    mul, add = _lcg_combine(_A, _C, steps)
    return (mul * state + add) & _MASK48

def _random_uint16_from_state(state):
    s1 = (_A * state + _C) & _MASK48
    v = (s1 >> 32) & 0xFFFF
    return v, s1

def _jitter_for_frame_index(frame_idx, stixel_w, stixel_h, base_seed, centered=True):
    s0 = _java_init_seed(int(base_seed))
    s_base = _lcg_advance(s0, 2 * int(frame_idx))
    v1, s1 = _random_uint16_from_state(s_base)
    v2, _  = _random_uint16_from_state(s1)
    if centered:
        jx = int(v1 % stixel_w) - (stixel_w // 2)
        jy = int(v2 % stixel_h) - (stixel_h // 2)
    else:
        jx = int(v1 % stixel_w)
        jy = int(v2 % stixel_h)
    return jx, jy

def _j_offset_index(jx, jy, stixel_w, stixel_h, centered=True):
    if centered:
        jx_idx = jx + (stixel_w // 2)
        jy_idx = jy + (stixel_h // 2)
    else:
        jx_idx = jx
        jy_idx = jy
    return jy_idx * stixel_w + jx_idx

def _index_to_jxy(index, stixel_w, stixel_h, centered=True):
    jy_idx, jx_idx = divmod(int(index), stixel_w)
    if centered:
        jx = jx_idx - (stixel_w // 2)
        jy = jy_idx - (stixel_h // 2)
    else:
        jx = jx_idx
        jy = jy_idx
    return jx, jy

# -------- Grouped-by-offset STA --------

def compute_sta_chunked_grouped_by_offset(
    spikes_sec,
    triggers_sec,
    generator,
    seed,
    depth,
    offset,
    *,
    chunk_size=2048,
    refresh=2,
    jitter=None,
):
    """
    Exact STA with jitter, but fast: aggregate in stixel space per (lag, offset),
    upsample+shift once per bucket at the end.

    jitter: dict with keys
        'stixel_w', 'stixel_h', 'screen_w', 'screen_h', 'back_rgb' (0..255 or 0..1),
        optional 'seed' (default seed) and 'centered' (default True).
    """

    if jitter is None:
        raise ValueError("Grouping-by-offset is only meaningful when jitter is enabled.")

    # --- geometry & jitter config ---
    h0, w0 = int(generator.height), int(generator.width)
    stixel_w = int(jitter["stixel_w"])
    stixel_h = int(jitter["stixel_h"])
    screen_w = int(jitter["screen_w"])
    screen_h = int(jitter["screen_h"])
    if w0 * stixel_w != screen_w or h0 * stixel_h != screen_h:
        raise ValueError("screen must equal (w0*stixel_w, h0*stixel_h)")
    back_rgb = tuple(jitter.get("back_rgb", (127, 127, 127)))
    if all(0 <= v <= 1 for v in back_rgb):
        back_rgb = tuple(int(round(255 * v)) for v in back_rgb)
    bg_norm = (np.array(back_rgb, dtype=np.float32) / 127.5) - 1.0  # shape (3,)
    jitter_seed = int(jitter.get("seed", seed))
    centered = bool(jitter.get("centered", True))

    n_offsets = stixel_w * stixel_h

    # --- bin spikes to unique-frame indices ---
    frame_indices = bin_spikes(
        spikes_sec=spikes_sec,
        triggers_sec=triggers_sec,
        refresh=refresh,
        bin_shift=0.95,
    ).astype(np.int64)

    total_frames = int(frame_indices.max()) + int(offset) + 1

    # --- preallocate grouped accumulators ---
    # sums in stixel space (already normalized to [-1,1])
    accum_stix = np.zeros((depth, n_offsets, h0, w0, 3), dtype=np.float32)
    # how many weighted additions per (d, offset)
    weights_dj = np.zeros((depth, n_offsets), dtype=np.int64)
    # total normalizer per depth
    spike_counts = np.zeros(depth, dtype=np.int64)

    current_seed = int(seed)

    # --- stream frames in chunks and aggregate into buckets ---
    for chunk_start in range(0, total_frames, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_frames)
        n = chunk_end - chunk_start

        frames_u8, current_seed = generator.draw_frames_batch(current_seed, n)  # (n, h0, w0, 3), uint8

        # counts per depth for this chunk, and union of used frames
        counts_per_depth = []
        used_mask = np.zeros(n, dtype=bool)
        for d in range(depth):
            stim_f = frame_indices - d + offset
            m = (stim_f >= chunk_start) & (stim_f < chunk_end)
            if not np.any(m):
                counts = np.zeros(n, dtype=np.int64)
            else:
                local = (stim_f[m] - chunk_start).astype(np.int64)
                counts = np.bincount(local, minlength=n).astype(np.int64)
            spike_counts[d] += int(counts.sum())
            counts_per_depth.append(counts)
            used_mask |= (counts > 0)

        if not used_mask.any():
            continue

        used_idx = np.nonzero(used_mask)[0]
        # vector-friendly: precompute normalized stixel frames for used indices
        # (do it once, reuse across all lags)
        base_f = ((frames_u8[used_idx].astype(np.float32) / 127.5) - 1.0)  # (Nu, h0, w0, 3)

        # jitter bucket index per used frame (absolute frame index)
        j_idx_list = []
        for k, i_local in enumerate(used_idx):
            f_abs = chunk_start + int(i_local)
            jx, jy = _jitter_for_frame_index(f_abs, stixel_w, stixel_h, jitter_seed, centered=centered)
            j_idx_list.append(_j_offset_index(jx, jy, stixel_w, stixel_h, centered=centered))
        j_idx_arr = np.asarray(j_idx_list, dtype=np.int64)

        # accumulate into (depth, offset) buckets
        for d in range(depth):
            counts = counts_per_depth[d][used_idx]  # (Nu,)
            nz = np.nonzero(counts)[0]
            if nz.size == 0:
                continue
            # For each nonzero frame, add base_f[k] * count into the right bucket
            for k in nz:
                c = int(counts[k])
                j = int(j_idx_arr[k])
                accum_stix[d, j] += base_f[k] * float(c)
                weights_dj[d, j] += c

    # --- finalize: upsample+shift each bucket ONCE and add to full-res STA ---
    sta = np.zeros((screen_h, screen_w, 3, depth), dtype=np.float32)

    for d in range(depth):
        if spike_counts[d] == 0:
            continue

        for j in range(n_offsets):
            wsum = int(weights_dj[d, j])
            if wsum == 0:
                continue

            jx, jy = _index_to_jxy(j, stixel_w, stixel_h, centered=centered)
            base_sum = accum_stix[d, j]  # (h0, w0, 3) sum of normalized stixel frames

            # Upsample once (nearest): (screen_h, screen_w, 3)
            up = base_sum.repeat(stixel_h, axis=0).repeat(stixel_w, axis=1)

            # Compute overlap between shifted content and the screen
            x_src0 = 0 if jx >= 0 else -jx
            y_src0 = 0 if jy >= 0 else -jy
            x_dst0 = jx if jx >= 0 else 0
            y_dst0 = jy if jy >= 0 else 0
            x_len = screen_w - abs(jx)
            y_len = screen_h - abs(jy)

            # 1) background bands (outside overlap), weighted by wsum
            bg = bg_norm * float(wsum)  # shape (3,)
            if jy > 0:   # top band
                sta[0:jy, :, :, d] += bg
            elif jy < 0: # bottom band
                sta[screen_h+jy:screen_h, :, :, d] += bg
            if jx > 0:   # left band (only the vertical middle after top/bottom handled)
                sta[y_dst0:y_dst0+y_len, 0:jx, :, d] += bg
            elif jx < 0: # right band
                sta[y_dst0:y_dst0+y_len, screen_w+jx:screen_w, :, d] += bg

            # 2) overlapped region: add shifted upsampled sum
            sta[y_dst0:y_dst0+y_len, x_dst0:x_dst0+x_len, :, d] += \
                up[y_src0:y_src0+y_len, x_src0:x_src0+x_len, :]

        # normalize this lag
        sta[:, :, :, d] /= float(spike_counts[d])

    return sta
