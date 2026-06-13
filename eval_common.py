"""
Shared evaluation metrics for the GNNML3 vs LightGCN comparison.

This module is the single source of truth for Recall@K / Precision@K / NDCG@K so
that both models are scored with byte-identical math. The three core functions
(`RecallPrecision_ATk`, `NDCGatK_r`, `getLabel`) are copied verbatim from the
reference LightGCN implementation (LightGCN-PyTorch/code/utils.py), and `score`
reproduces the per-user aggregation done in LightGCN's `Procedure.Test` (sum the
per-user contributions, then divide by the number of evaluated test users).

Usage:
    from eval_common import score
    metrics = score(ranked_topk, ground_truth, ks=[20, 50])
    # -> {"recall@20": ..., "precision@20": ..., "ndcg@20": ..., "recall@50": ...}
"""

import numpy as np


# ──────────────────────────────────────────────────────────────
#  Verbatim LightGCN metric functions (LightGCN-PyTorch/code/utils.py)
# ──────────────────────────────────────────────────────────────

def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


# ──────────────────────────────────────────────────────────────
#  Aggregation wrapper (mirrors Procedure.Test)
# ──────────────────────────────────────────────────────────────

def score(ranked_topk, ground_truth, ks=(20,)):
    """Compute mean Recall/Precision/NDCG@K over all evaluated users.

    Args:
        ranked_topk: sequence (len = #test users) of ranked predicted item ids,
            each of length >= max(ks), best first. Training items must already
            be excluded by the caller.
        ground_truth: sequence (len = #test users) of held-out positive item
            ids (list/array/set-like) for each corresponding user. Users with no
            ground-truth items must be filtered out by the caller (LightGCN keys
            its test dict only on users that have test interactions).
        ks: iterable of cutoff values.

    Returns:
        dict mapping "recall@k" / "precision@k" / "ndcg@k" to floats, each
        averaged over the number of users in ``ground_truth``.
    """
    ks = list(ks)
    n_users = len(ground_truth)
    if n_users == 0:
        return {f"{m}@{k}": 0.0 for k in ks for m in ("recall", "precision", "ndcg")}

    # Ground truth must be index-able by position for getLabel/NDCG.
    gt = [list(g) for g in ground_truth]
    sorted_items = np.asarray([list(p) for p in ranked_topk])
    r = getLabel(gt, sorted_items)

    out = {}
    for k in ks:
        rp = RecallPrecision_ATk(gt, r, k)
        ndcg = NDCGatK_r(gt, r, k)
        out[f"recall@{k}"] = float(rp["recall"] / n_users)
        out[f"precision@{k}"] = float(rp["precision"] / n_users)
        out[f"ndcg@{k}"] = float(ndcg / n_users)
    return out


# ──────────────────────────────────────────────────────────────
#  Equivalence sanity check
# ──────────────────────────────────────────────────────────────

def _selftest():
    """Tiny hand-checkable example confirming the aggregation matches a manual
    computation of LightGCN's Test() formulas."""
    # Two users.
    #   user 0: ground truth {1, 3}, ranked [1, 5, 3, 9, ...]
    #           -> hit positions (0-indexed) 0 and 2
    #   user 1: ground truth {7},    ranked [2, 7, 4, 8, ...]
    #           -> hit position 1
    ranked = [
        [1, 5, 3, 9, 0],
        [2, 7, 4, 8, 6],
    ]
    gt = [[1, 3], [7]]
    k = 5

    res = score(ranked, gt, ks=[k])

    # Manual Recall@5: user0 = 2/2 = 1.0, user1 = 1/1 = 1.0 -> mean 1.0
    exp_recall = (2 / 2 + 1 / 1) / 2
    # Manual Precision@5: user0 = 2/5, user1 = 1/5 -> sum/k then /n_users
    exp_precision = ((2 + 1) / 5) / 2
    # Manual NDCG@5:
    #   user0 dcg = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    #          idcg = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
    #          ndcg = 1.5 / 1.6309
    #   user1 dcg = 1/log2(3) = 0.6309 ; idcg = 1/log2(2) = 1 ; ndcg = 0.6309
    u0 = (1 / np.log2(2) + 1 / np.log2(4)) / (1 / np.log2(2) + 1 / np.log2(3))
    u1 = (1 / np.log2(3)) / (1 / np.log2(2))
    exp_ndcg = (u0 + u1) / 2

    assert abs(res[f"recall@{k}"] - exp_recall) < 1e-9, res
    assert abs(res[f"precision@{k}"] - exp_precision) < 1e-9, res
    assert abs(res[f"ndcg@{k}"] - exp_ndcg) < 1e-9, res

    # Cross-check against calling the verbatim functions exactly as
    # Procedure.Test does (single "batch"), to prove score() reproduces it.
    r = getLabel(gt, np.asarray(ranked))
    rp = RecallPrecision_ATk(gt, r, k)
    ndcg = NDCGatK_r(gt, r, k)
    assert abs(res[f"recall@{k}"] - rp["recall"] / len(gt)) < 1e-12
    assert abs(res[f"precision@{k}"] - rp["precision"] / len(gt)) < 1e-12
    assert abs(res[f"ndcg@{k}"] - ndcg / len(gt)) < 1e-12

    print("eval_common self-test passed:", {k2: round(v, 6) for k2, v in res.items()})


if __name__ == "__main__":
    _selftest()
