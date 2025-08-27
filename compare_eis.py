import numpy as np
from scipy.signal import correlate

from scipy.signal import correlate
import numpy as np

def compare_eis(eis, ei_template=None, max_lag=30, thr=30.0):
    """
    If `ei_template` is None  → full pair-wise similarity matrix  [k,k]
    If `ei_template` given     → column vector of similarities  [k,1]
    Similarity is cosine over **signal channels only**
    (channels whose P2P > `thr` in either EI).
    """

    if isinstance(eis, list):
        eis = np.stack(eis, axis=0)
    k, C, T = eis.shape
    ptp = eis.ptp(axis=2)                     # [k, C]

    # ----------------------------------------------------------
    # 1) stack-vs-stack   (symmetric [k,k])
    # ----------------------------------------------------------
    if ei_template is None:
        sim = np.zeros((k, k), dtype=np.float32)
        for i in range(k):
            ei_i = eis[i]
            dom_i = np.argmax(ptp[i])
            trace_i = ei_i[dom_i]

            for j in range(i, k):
                ei_j = eis[j]
                dom_j = np.argmax(ptp[j])
                trace_j = ei_j[dom_j]

                # align by x-corr of the dominant traces
                lags = np.arange(-max_lag, max_lag + 1)
                xc = correlate(trace_i, trace_j, mode="full", method="auto")
                center = len(xc) // 2
                shift = lags[np.argmax(xc[center - max_lag:center + max_lag + 1])]

                ei_j_aligned = np.roll(ei_j, shift, axis=1)

                mask = (ptp[i] > thr) | (ptp[j] > thr)
                if mask.any():
                    a = ei_i[mask].ravel()
                    b = ei_j_aligned[mask].ravel()
                    val = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
                else:
                    val = 0.0

                sim[i, j] = sim[j, i] = val
        return sim

    # ----------------------------------------------------------
    # 2) stack-vs-single template   (column [k,1])
    # ----------------------------------------------------------
    else:
        sim = np.zeros((k, 1), dtype=np.float32)

        # dominant channel and P2P mask for template
        ptp_t   = ei_template.ptp(axis=1)
        dom_t   = int(np.argmax(ptp_t))
        trace_t = ei_template[dom_t]

        for i in range(k):
            ei_i   = eis[i]
            dom_i  = int(np.argmax(ptp[i]))
            trace_i = ei_i[dom_i]

            lags = np.arange(-max_lag, max_lag + 1)
            xc   = correlate(trace_i, trace_t, mode="full", method="auto")
            center = len(xc) // 2
            shift  = lags[np.argmax(xc[center - max_lag:center + max_lag + 1])]

            ei_t_aligned = np.roll(ei_template, shift, axis=1)

            mask = (ptp[i] > thr) | (ptp_t > thr)
            if mask.any():
                a = ei_i[mask].ravel()
                b = ei_t_aligned[mask].ravel()
                sim[i, 0] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            else:
                sim[i, 0] = 0.0

        return sim
