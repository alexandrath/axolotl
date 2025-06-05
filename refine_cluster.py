import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from compare_eis import compare_eis
import torch

def baseline_correct(ei, n_baseline=5):
    return ei - ei[:, :n_baseline].mean(axis=1, keepdims=True)

def refine_cluster(snips, ei_template, selected_by_ei, k=8, inds=None, depth=0, threshold=0.80):
    """
    Recursively refine a cluster using EI similarity and k-means on PCA.

    Parameters:
        snips           : [C x T x N] snippets (full spike set)
        ei_template     : [512 x T] template EI
        selected_by_ei  : list of channel indices used in PCA
        k               : initial number of k-means clusters (default 8)
        inds            : indices into snips to refine (optional, default: all)
        depth           : recursion depth (for logging/debugging)
        threshold       : EI cosine similarity threshold for merging

    Returns:
        final_inds : list of spike indices matching refined cluster
    """
    #print("running")

    if inds is None:
        inds = np.arange(snips.shape[2])
    if len(inds) < 100:
        return inds

    # Step 1: Subset and flatten for PCA
    snips_sel = snips[selected_by_ei][:, :, inds]
    snips_centered = snips_sel - snips_sel.mean(axis=1, keepdims=True)
    n_spikes = snips_sel.shape[2]
    snips_flat = snips_centered.transpose(2, 0, 1).reshape(n_spikes, -1)

    pca = PCA(n_components=10)
    pcs = pca.fit_transform(snips_flat)

    # Step 2: k-means clustering
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pcs)

    # Step 3: Compute EIs for each cluster
    ei_per_cluster = []
    for i in range(k):
        idx = np.where(labels == i)[0]
        if len(idx) == 0:
            ei = np.zeros_like(ei_template)
        else:
            snips_i = snips[:, :, inds[idx]]  # full EI, not just PCA channels
            snips_i = torch.from_numpy(snips_i)
            ei = torch.mean(snips_i, dim=2).numpy()
            ei = baseline_correct(ei)
        ei_per_cluster.append(ei)

    # Step 4: EI similarity to template
    sim = compare_eis(ei_per_cluster, ei_template)

    #np.set_printoptions(precision=2, suppress=True)
    #print(sim)

    matched = [i for i in range(k) if sim[i, i] > threshold]

    if len(matched) == 0:
        return np.array([], dtype=int)

    if len(matched) == 1:
        # Just one match — recurse
        idx = np.where(labels == matched[0])[0]
        return refine_cluster(snips, ei_template, selected_by_ei, k=4, inds=inds[idx], depth=depth+1)

    # Multiple matched — merge them
    keep = np.isin(labels, matched)
    return inds[keep]
