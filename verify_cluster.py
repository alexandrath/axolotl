import numpy as np
import torch
import matplotlib.pyplot as plt
from extract_data_snippets import extract_snippets
from compare_eis import compare_eis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx

def baseline_correct(ei, n_baseline=5):
    return ei - ei[:, :n_baseline].mean(axis=1, keepdims=True)

def compute_ei(snips):
    snips_torch = torch.from_numpy(snips)
    ei = torch.mean(snips_torch, dim=2).numpy()
    return baseline_correct(ei)

def select_channels(ei, min_chan=30, max_chan=80, threshold=15):
    p2p = ei.max(axis=1) - ei.min(axis=1)
    selected = np.where(p2p > threshold)[0]
    if len(selected) > max_chan:
        selected = np.argsort(p2p)[-max_chan:]
    elif len(selected) < min_chan:
        selected = np.argsort(p2p)[-min_chan:]
    return np.sort(selected)

def find_merge_groups(sim, threshold):
    G = nx.Graph()
    k = sim.shape[0]
    G.add_nodes_from(range(k))  # ensure isolated clusters are counted
    for i in range(k):
        for j in range(i + 1, k):
            if sim[i, j] > threshold:
                G.add_edge(i, j)
    return list(nx.connected_components(G))

def plot_kmeans_pca(pcs, labels):
    plt.figure(figsize=(6, 5))
    for i in np.unique(labels):
        cluster_points = pcs[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i+1}", s=10)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA on all concatenated waveforms")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def verify_cluster(spike_times, dat_path, params):

    window = params.get('window', (-20, 60))

    if isinstance(dat_path, np.ndarray):
        snips = dat_path
    elif isinstance(dat_path, str):
        snips = extract_snippets(dat_path, spike_times, window)
    else:
        print("Unknown type:", type(dat_path))


    
    min_spikes = params.get('min_spikes', 100)
    k_start = params.get('k_start', 8)
    k_refine = params.get('k_refine', 2)
    ei_sim_threshold = params.get('ei_sim_threshold', 0.95)

    
    full_inds = np.arange(snips.shape[2])

    cluster_pool = [{'inds': full_inds, 'depth': 0}]
    final_clusters = []
    max_depth = params.get('max_depth', 10)

    while cluster_pool:

        #print(f"[verify_cluster] {len(final_clusters)} finalized, {len(cluster_pool)} pending")
        cl = cluster_pool.pop(0)
        inds = cl['inds']
        depth = cl['depth']

        if cl['depth'] >= max_depth:
            #print("Reached max depth ({max_depth}) — finalizing.")
            final_clusters.append({
                'inds': inds,
                'ei': compute_ei(snips[:, :, inds]),
                'channels': select_channels(compute_ei(snips[:, :, inds]))
            })
            continue
        #print(f"  Refining cluster with {len(inds)} spikes at depth {depth}")

        if len(inds) < min_spikes:
            #print("  Cluster too small — skipping.")
            continue

        k = k_start if depth == 0 else k_refine

        snips_cl = snips[:, :, inds]
        ei = compute_ei(snips_cl)
        selected = select_channels(ei)
        snips_sel = snips[np.ix_(selected, np.arange(snips.shape[1]), inds)]

        snips_centered = snips_sel - snips_sel.mean(axis=1, keepdims=True)
        flat = snips_centered.transpose(2, 0, 1).reshape(len(inds), -1)
        pcs = PCA(n_components=10).fit_transform(flat)
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(pcs)

        #plot_kmeans_pca(pcs, labels)

        subclusters = []
        for i in range(k):
            idx = np.where(labels == i)[0]
            if len(idx) > 0:
                sub_inds = inds[idx]
                ei_i = compute_ei(snips[:, :, sub_inds])
                subclusters.append({
                    'inds': sub_inds,
                    'ei': ei_i,
                })

        #print(f"  KMeans produced {len(subclusters)} subclusters")

        if len(subclusters) <= 1:
            #print("  Only one subcluster — finalizing as-is")
            final_clusters.append({
                'inds': inds,
                'ei': ei,
                'channels': selected
            })
            continue

        large_subclusters = [(i, cl) for i, cl in enumerate(subclusters) if len(cl['inds']) >= min_spikes]
        #print(f"  {len(large_subclusters)} subclusters ≥ {min_spikes} spikes")

        if len(large_subclusters) == 1:
            #print("  Only one large cluster — finalizing.")
            final_clusters.append({
                'inds': large_subclusters[0][1]['inds'],
                'ei': large_subclusters[0][1]['ei'],
                'channels': select_channels(large_subclusters[0][1]['ei'])
            })
            continue

        if len(large_subclusters) > 1:
            eis_large = [cl['ei'] for _, cl in large_subclusters]
            sim_large = compare_eis(eis_large, None)
            #print("  Pairwise similarity (large clusters):\n", np.round(sim_large, 3))
            merge_groups = find_merge_groups(sim_large, ei_sim_threshold)
            #print(f"  Found {len(merge_groups)} merge groups")
            if len(merge_groups) == 1:
                #print("  All large subclusters are mergeable — aborting split")
                final_clusters.append({
                    'inds': inds,
                    'ei': ei,
                    'channels': selected
                })
                continue

        eis = [cl['ei'] for _, cl in large_subclusters]
        sim = compare_eis(eis, None)
        #print("  Similarity matrix (all large clusters):\n", np.round(sim, 3))

        groups = find_merge_groups(sim, ei_sim_threshold)
        #print(f"  {len(groups)} merge groups will be processed")

        if len(groups) == 0:
            #print("  No valid subclusters after merge — keeping current cluster.")
            final_clusters.append({
                'inds': inds,
                'ei': ei,
                'channels': selected
            })
            continue

        for group in groups:
            group = list(group)
            all_inds = np.concatenate([large_subclusters[i][1]['inds'] for i in group])
            #print(f"    Group of size {len(all_inds)}")
            if len(all_inds) >= min_spikes:
                cluster_pool.append({
                    'inds': all_inds,
                    'depth': depth + 1
                })
            #else:
                #print("    Discarding — too small")

    if len(final_clusters) == 1:
        #print("  Only one final cluster — skipping deduplication.")
        return final_clusters

    all_final_eis = [compute_ei(snips[:, :, cl['inds']]) for cl in final_clusters]
    sim = compare_eis(all_final_eis, None)
    groups = find_merge_groups(sim, ei_sim_threshold)



    kept = []
    for group in groups:
        group = list(group)
        all_inds = np.concatenate([final_clusters[i]['inds'] for i in group])
        ei_final = compute_ei(snips[:, :, all_inds])
        chans_final = select_channels(ei_final)
        kept.append({
            'inds': all_inds,
            'ei': ei_final,
            'channels': chans_final
        })

    all_eis = [cl['ei'] for cl in kept]
    sim = compare_eis(all_eis, None)

    print("Final EI similarity matrix:\n", np.round(sim, 2))

    print(f"[verify_cluster] Merged down to {len(kept)} final clusters.")
    return kept
