"""
Shared benchmark result IO for the GNNML3 vs LightGCN comparison.

Both harnesses (bipartite_experiment.py and run_lightgcn.py) emit one result
record per run into a JSON-lines file using this module, and run_comparison.py
reads them back for aggregation. JSONL is used (rather than a fixed-column CSV)
so the metric columns can vary with the requested top-K list without schema
juggling.

Canonical record fields:
    model, dataset, seed, epochs, embed_dim, layers, lr, decay, device,
    recall@<k>, precision@<k>, ndcg@<k> (one each per evaluated k),
    setup_time_s, train_time_s, time_to_best_s, peak_mem_mb, n_params,
    best_epoch
plus any extra model-specific fields (e.g. nfreq/dv/k_svd/recfield for gnnml3).
"""

import json
import os


def append_jsonl(path, row):
    """Append a single dict as one JSON line, creating parent dirs as needed."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def read_jsonl(path):
    """Read a JSON-lines file into a list of dicts (empty list if missing)."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
