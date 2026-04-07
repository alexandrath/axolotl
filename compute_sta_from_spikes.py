import numpy as np
import ctypes
from bin_spikes_by_triggers import (
    bin_spikes_matlab_style_robust_autorepair as bin_spikes,
)

# =========================
# C-library frame generator
# =========================

class RGBFrameGenerator:
    """
    Thin ctypes wrapper for libdraw_rgb.so that draws unique stimulus frames
    at the *stixel* resolution. The batch call advances the seed across frames.

    configure(width, height, lut, noise_type, n_bits)
    draw_frames_batch(seed, n_frames) -> (frames_u8[N,H,W,3], updated_seed)
    """
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
            self.lut,
            self.noise_type, self.n_bits,
            n_frames,
            output,
            ctypes.byref(seed_out)
        )

        arr = np.ctypeslib.as_array(output)
        frames = arr.reshape((n_frames, self.width, self.height, 3)).transpose(0, 2, 1, 3)
        return frames, int(seed_out.value)


# =======================
# Java-style RNG for jitter
# =======================

_A = 0x5DEECE66D
_C = 0xB
_MASK48 = (1 << 48) - 1

def _java_init_seed(seed):
    """Init_RNG_JavaStyle: state = (seed ^ 0x5DEECE66D) & ((1<<48)-1)."""
    return (int(seed) ^ _A) & _MASK48

def _lcg_combine(a, c, steps):
    """
    Return (mul, add) so that applying the LCG `steps` times equals:
        s_next = mul * s + add  (mod 2^48)
    """
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
    """Advance once and return (value16, new_state)."""
    s1 = (_A * state + _C) & _MASK48
    value16 = (s1 >> 32) & 0xFFFF
    return value16, s1

def _jitter_for_frame_index(frame_idx, stixel_w, stixel_h, base_seed, centered=True):
    """
    Deterministic jitter for absolute unique-stimulus frame index.
    Two random_uint16 draws per frame, like the MEX path. :contentReference[oaicite:0]{index=0}
    """
    s0 = _java_init_seed(int(base_seed))
    # Each frame consumes 2 draws; advance to start of this frame's draws
    s_base = _lcg_advance(s0, 2 * int(frame_idx))
    v1, s1 = _random_uint16_from_state(s_base)
    v2, _ = _random_uint16_from_state(s1)

    if centered:
        jx = int(v1 % stixel_w) - (stixel_w // 2)
        jy = int(v2 % stixel_h) - (stixel_h // 2)
    else:
        jx = int(v1 % stixel_w)
        jy = int(v2 % stixel_h)
    return jx, jy


# ============================================
# On-demand jitter + integer upscaling (single)
# ============================================

def _jitter_and_upscale_one(frame_u8,
                            stixel_w, stixel_h,
                            screen_w, screen_h,
                            back_rgb, jx, jy):
    """
    frame_u8: (h0, w0, 3) uint8 at stixel resolution.
    Returns   (screen_h, screen_w, 3) uint8 after integer upscaling + jitter.
    Mirrors Run_OnTheFly: upsample by integer factor, pad, jitter-shift, crop. :contentReference[oaicite:1]{index=1}
    """
    h0, w0, _ = frame_u8.shape
    up_h = h0 * stixel_h
    up_w = w0 * stixel_w
    if up_h != screen_h or up_w != screen_w:
        raise ValueError(
            "screen (%dx%d) must equal W*stixel_w x H*stixel_h = %dx%d"
            % (screen_w, screen_h, up_w, up_h)
        )

    # integer upsample (nearest)
    up = frame_u8.repeat(stixel_h, axis=0).repeat(stixel_w, axis=1)

    pad_h = stixel_h - 1
    pad_w = stixel_w - 1
    padded = np.empty((screen_h + 2 * pad_h, screen_w + 2 * pad_w, 3), dtype=np.uint8)
    padded[:] = back_rgb
    padded[pad_h:pad_h + screen_h, pad_w:pad_w + screen_w, :] = up

    y0 = pad_h + jy
    x0 = pad_w + jx
    return padded[y0:y0 + screen_h, x0:x0 + screen_w, :]

def _apply_jitter_and_upscale_fast(frames_u8, *,
                                   stixel_w, stixel_h,
                                   screen_w, screen_h,
                                   back_rgb=(127,127,127),
                                   rng=None,
                                   centered=True):
    """
    Fast vectorized jitter matching MATLAB Run_OnTheFly behavior.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    
    N, h0, w0, _ = frames_u8.shape
    
    # Upscale by integer repeats
    up = frames_u8.repeat(stixel_h, axis=1).repeat(stixel_w, axis=2)
    
    # Pad size needs to accommodate jitter range
    if centered:
        max_jitter = stixel_w // 2  # assuming square stixels
        pad_size = max_jitter + 1
    else:
        pad_size = max(stixel_w, stixel_h)
    
    # Pad all frames at once
    padded = np.pad(up, 
                    ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                    mode='constant', 
                    constant_values=back_rgb)
    
    # Generate jitter offsets
    if centered:
        jx = rng.randint(-stixel_w//2, stixel_w//2 + 1, size=N)
        jy = rng.randint(-stixel_h//2, stixel_h//2 + 1, size=N)
    else:
        jx = rng.randint(0, stixel_w, size=N)
        jy = rng.randint(0, stixel_h, size=N)
    
    # Extract windows (loop is fast for small N, or vectorize below)
    out = np.empty((N, screen_h, screen_w, 3), dtype=np.uint8)
    for i in range(N):
        y0 = pad_size + jy[i]
        x0 = pad_size + jx[i]
        out[i] = padded[i, y0:y0+screen_h, x0:x0+screen_w, :]
    
    return out

# =======================
# Chunked STA computation
# =======================

def compute_sta_chunked(
    spikes_sec,
    triggers_sec,
    generator,
    seed,
    depth,
    offset,
    chunk_size=2048,
    refresh=2,
    jitter=None,
):
    """
    Chunked STA that never holds all frames in memory.

    spikes_sec: 1D spike times (seconds)
    triggers_sec: 1D trigger times (seconds)
    generator: configured RGBFrameGenerator (stixel resolution)
    seed: initial RNG seed for stimulus
    depth: number of pre-spike frames to average (lag bins)
    offset: frame offset applied before lagging
    chunk_size: how many unique stimulus frames to generate per chunk
    refresh: frames-per-unique-stimulus-frame (Photons 'interval')
    jitter: None or dict with keys:
        'stixel_w', 'stixel_h', 'screen_w', 'screen_h', 'back_rgb' (0–255 or 0–1),
        optional 'seed' (default: seed) and 'centered' (default True).
    """

    # Bin spikes to unique-stimulus frame indices
    frame_indices = bin_spikes(
        spikes_sec=spikes_sec,
        triggers_sec=triggers_sec,
        refresh=refresh,
        bin_shift=0.95,
    ).astype(np.int64)

    total_frames = int(frame_indices.max()) + int(offset) + 1

    if jitter is None:
        H, W = generator.height, generator.width
        use_jitter = False
    else:
        stixel_w = int(jitter["stixel_w"])
        stixel_h = int(jitter["stixel_h"])
        screen_w = int(jitter["screen_w"])
        screen_h = int(jitter["screen_h"])
        back_rgb = tuple(jitter.get("back_rgb", (127, 127, 127)))
        if all(0 <= v <= 1 for v in back_rgb):
            back_rgb = tuple(int(round(255 * v)) for v in back_rgb)

        up_w = int(generator.width) * stixel_w
        up_h = int(generator.height) * stixel_h
        if up_w != screen_w or up_h != screen_h:
            raise ValueError(
                "screen (%dx%d) must equal W*stixel_w x H*stixel_h = %dx%d"
                % (screen_w, screen_h, up_w, up_h)
            )
        H, W = screen_h, screen_w
        use_jitter = True
        jitter_seed = int(jitter.get("seed", seed))
        centered = bool(jitter.get("centered", True))

    sta = np.zeros((H, W, 3, depth), dtype=np.float32)
    spike_counts = np.zeros(depth, dtype=np.int64)

    current_seed = int(seed)

    for chunk_start in range(0, total_frames, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_frames)
        n_frames = chunk_end - chunk_start

        # Generate this chunk of unique frames at stixel resolution
        frames_u8, current_seed = generator.draw_frames_batch(current_seed, n_frames)

        # Precompute, for each depth, how many spikes want each frame in this chunk
        counts_per_depth = []
        used_mask = np.zeros(n_frames, dtype=bool)
        for d in range(depth):
            stim_f = frame_indices - d + offset
            m = (stim_f >= chunk_start) & (stim_f < chunk_end)
            if not np.any(m):
                counts = np.zeros(n_frames, dtype=np.int64)
            else:
                local = (stim_f[m] - chunk_start).astype(np.int64)
                counts = np.bincount(local, minlength=n_frames).astype(np.int64)
            spike_counts[d] += int(counts.sum())
            counts_per_depth.append(counts)
            used_mask |= (counts > 0)

        if not used_mask.any():
            continue

        used_indices = np.nonzero(used_mask)[0]

        # For each used frame in this chunk, build image once and reuse per depth
        for i_local in used_indices:
            f_abs = chunk_start + int(i_local)
            base = frames_u8[i_local]  # (h0, w0, 3) uint8

            if use_jitter:
                jx, jy = _jitter_for_frame_index(
                    f_abs, stixel_w, stixel_h, jitter_seed, centered=centered
                )
                img_u8 = _jitter_and_upscale_one(
                    base,
                    stixel_w=stixel_w,
                    stixel_h=stixel_h,
                    screen_w=screen_w,
                    screen_h=screen_h,
                    back_rgb=back_rgb,
                    jx=jx,
                    jy=jy,
                )
                frame_f = (img_u8.astype(np.float32) / 127.5) - 1.0
            else:
                frame_f = (base.astype(np.float32) / 127.5) - 1.0

            for d in range(depth):
                cnt = counts_per_depth[d][i_local]
                if cnt:
                    sta[:, :, :, d] += frame_f * float(cnt)

    # Normalize
    for d in range(depth):
        if spike_counts[d] > 0:
            sta[:, :, :, d] /= float(spike_counts[d])

    return sta
