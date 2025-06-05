import numpy as np
from scipy.signal import correlate

def compare_eis(eis, ei_template=None, max_lag=3):
    """
    Compare a list of EIs to each other or to a template.

    Parameters:
        eis         : list of [C x T] arrays (cluster EIs)
        ei_template : optional [C x T] template EI
        max_lag     : max lag to align on dominant channel

    Returns:
        sim : [k x k] similarity matrix if ei_template is None
              [k x 1] similarity vector if template is given
    """
    k = len(eis)
    if ei_template is not None:
        sim = np.zeros((k, 1))
        for i in range(k):
            ei_i = eis[i]
            dom_chan = np.argmax(np.max(np.abs(ei_i), axis=1))
            trace_i = ei_i[dom_chan, :]
            trace_t = ei_template[dom_chan, :]

            lags = np.arange(-max_lag, max_lag + 1)
            xc = correlate(trace_i, trace_t, mode='full', method='auto')
            center = len(xc) // 2
            xc_window = xc[center - max_lag:center + max_lag + 1]
            shift = lags[np.argmax(xc_window)]

            aligned_t = np.roll(ei_template, shift, axis=1)
            sim[i] = np.dot(ei_i.flatten(), aligned_t.flatten()) / (
                np.linalg.norm(ei_i) * np.linalg.norm(aligned_t))
        return sim

    else:
        sim = np.zeros((k, k))
        for i in range(k):
            ei_i = eis[i]
            dom_chan = np.argmax(np.max(np.abs(ei_i), axis=1))
            trace_i = ei_i[dom_chan, :]

            for j in range(i, k):
                ei_j = eis[j]
                trace_j = ei_j[dom_chan, :]

                lags = np.arange(-max_lag, max_lag + 1)
                xc = correlate(trace_i, trace_j, mode='full', method='auto')
                center = len(xc) // 2
                xc_window = xc[center - max_lag:center + max_lag + 1]
                shift = lags[np.argmax(xc_window)]

                aligned_j = np.roll(ei_j, shift, axis=1)
                val = np.dot(ei_i.flatten(), aligned_j.flatten()) / (
                    np.linalg.norm(ei_i) * np.linalg.norm(aligned_j))
                sim[i, j] = val
                sim[j, i] = val
        return sim
