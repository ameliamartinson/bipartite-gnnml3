import networkx as nx
import numpy as np
import scipy.sparse as sp

from graph_loader import normalized_biadjacency


def phi(x, b=5, f_s=0.5):
    return np.exp(-b * (x - f_s) ** 2)


def h(x, b=5, f_s=0.5):  # even part
    return 0.5 * (phi(x, b, f_s) + phi(-x, b, f_s))


def g(x, b=5, f_s=1):  # odd part
    return 0.5 * (phi(x, b, f_s) - phi(-x, b, f_s))


def _dense_mask(M):
    if sp.issparse(M):
        return M.toarray()
    return np.asarray(M)


def _sorted_truncated_svd(B: sp.spmatrix, k: int):
    min_dim = min(B.shape)
    if min_dim <= 1:
        U, S, Vt = np.linalg.svd(B.toarray(), full_matrices=False)
        return U, S, Vt

    k = max(1, min(int(k), min_dim - 1))
    U, S, Vt = sp.linalg.svds(B, k=k)
    order = np.argsort(S)[::-1]
    return U[:, order], S[order], Vt[order, :]


def spectral_design_full(G: nx.Graph, M: sp.csr_matrix, s, phi, h, g):
    B = normalized_biadjacency(G).toarray()
    m, n = B.shape

    M = _dense_mask(M)
    M_11 = M[0:m, 0:m]
    M_12 = M[0:m, m:]
    M_21 = M[m:, 0:m]
    M_22 = M[m:, m:]

    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    lambda_max = float(np.max(S))

    s_vals = np.linspace(-lambda_max, lambda_max, s)

    ret_arr = []
    for f_s in s_vals:
        h_S = np.diag(h(S, f_s=f_s))
        g_S = np.diag(g(S, f_s=f_s))

        UU = M_11 * (U @ h_S @ U.T)
        UV = M_12 * (U @ g_S @ Vt)
        VU = M_21 * (Vt.T @ g_S @ U.T)
        VV = M_22 * (Vt.T @ h_S @ Vt)

        ret_arr.append(sp.csr_matrix(np.block([[UU, UV], [VU, VV]])))

    return ret_arr


def spectral_design_trunc(G: nx.Graph, M: sp.csr_matrix, s, k, phi, h, g):
    B = normalized_biadjacency(G)
    m, n = B.shape

    M = _dense_mask(M)
    M_11 = M[0:m, 0:m]
    M_12 = M[0:m, m:]
    M_21 = M[m:, 0:m]
    M_22 = M[m:, m:]

    U, S, Vt = _sorted_truncated_svd(B, k=k)
    lambda_max = float(np.max(S))

    s_vals = np.linspace(-lambda_max, lambda_max, s)

    ret_arr = []
    for f_s in s_vals:
        h_S = np.diag(h(S, f_s=f_s))
        g_S = np.diag(g(S, f_s=f_s))

        UU = M_11 * (U @ h_S @ U.T)
        UV = M_12 * (U @ g_S @ Vt)
        VU = M_21 * (Vt.T @ g_S @ U.T)
        VV = M_22 * (Vt.T @ h_S @ Vt)

        ret_arr.append(sp.csr_matrix(np.block([[UU, UV], [VU, VV]])))

    return ret_arr
