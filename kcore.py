"""Shared k-core dataset filtering for the GNNML3 and LightGCN runners.

Both experiment entry points (bipartite_experiment.py and the LightGCN
dataloader staged in modified_LightGCN_code/) use these functions so that
run_comparison.py --k-core K feeds the exact same filtered graph to each
model. The k-core of a graph is unique and the remapping below is
deterministic, so identical input files yield identical filtered graphs.
"""

import networkx as nx


def k_core_filter(edges, k):
    """Keep only interactions in the k-core: users and items that still have
    at least k interactions after recursively dropping nodes with fewer than
    k connections (standard k-core dataset filtering)."""
    # Tag nodes so user id i and item id i don't collide in the same graph.
    G = nx.Graph()
    G.add_edges_from((("u", u), ("i", i)) for u, i in edges)
    core_nodes = set(nx.k_core(G, k))
    return [(u, i) for u, i in edges
            if ("u", u) in core_nodes and ("i", i) in core_nodes]


def remap_k_core(train_edges, test_edges):
    """Remap the users/items surviving k-core filtering to contiguous ids.

    Test interactions involving filtered-out users or items are dropped.
    Returns (train_edges, test_edges, n_users, n_items) with the new ids.
    """
    keep_u = sorted({u for u, _ in train_edges})
    keep_i = sorted({i for _, i in train_edges})
    umap = {u: n for n, u in enumerate(keep_u)}
    imap = {i: n for n, i in enumerate(keep_i)}
    train = [(umap[u], imap[i]) for u, i in train_edges]
    test = [(umap[u], imap[i]) for u, i in test_edges if u in umap and i in imap]
    return train, test, len(keep_u), len(keep_i)
