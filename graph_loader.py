import networkx as nx
from networkx.algorithms import bipartite
from pathlib import Path
import numpy as np
import scipy.sparse as sp

#########
# Paths: include test.txt and train.txt
# datasets/amazon-book/
# datasets/gowalla/
# datasets/yelp2018/

amazon_train = Path("datasets/amazon-book/train.txt")
amazon_test = Path("datasets/amazon-book/test.txt")

gowalla_train = Path("datasets/gowalla/train.txt")
gowalla_test = Path("datasets/gowalla/test.txt")

yelp2018_train = Path("datasets/yelp2018/train.txt")
yelp2018_test = Path("datasets/yelp2018/train.txt")

def matrix_to_graph(B: sp.csr_matrix): # B scipy sparse matrix
    G = nx.Graph()
    B_coo = B.tocoo()
    M,N = B.shape

    for u_idx in range(M):
        G.add_node(f"user_{u_idx}", node_type="user")
    
    for i_idx in range(N):
        G.add_node(f"item_{i_idx}", node_type="item")
        
    for u_idx, i_idx, weight in zip(B_coo.row, B_coo.col, B_coo.data):
        G.add_edge(f"user_{u_idx}", f"item_{i_idx}", weight=weight)
        
    return G

def load_bipartite_graph(txt_path):
    G = nx.Graph()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            user_id = int(parts[0])
            item_ids = map(int, parts[1:])

            u = f"user_{user_id}"
            G.add_node(u, node_type="user")

            for item_id in item_ids:
                i = f"item_{item_id}"
                G.add_node(i, node_type="item")
                G.add_edge(u,i)
    return G

def numeric_id(node_name):
    return int(node_name.split("_", 1)[1])

def biadjacency(G: nx.Graph):
    users = sorted(
        [n for n, d in G.nodes(data=True) if d["node_type"] == "user"],
        key=numeric_id
    )
    items = sorted(
        [n for n, d in G.nodes(data=True) if d["node_type"] == "item"],
        key=numeric_id
    )
    B = bipartite.biadjacency_matrix(G, row_order=users, column_order=items, dtype=np.float32)
    return B

def normalized_biadjacency(G: nx.Graph):
    B = biadjacency(G)
    row_sums = np.array(B.sum(axis=1).flatten())
    col_sums = np.array(B.sum(axis=0).flatten())

    d_inv_sqrt_row = np.zeros_like(row_sums)
    d_inv_sqrt_col = np.zeros_like(col_sums)

    np.power(row_sums, -0.5, where=(row_sums != 0), out=d_inv_sqrt_row)
    np.power(col_sums, -0.5, where=(col_sums != 0), out=d_inv_sqrt_col)

    d_inv_sqrt_row[row_sums == 0] = 0
    d_inv_sqrt_col[col_sums == 0] = 0

    D_u_inv_sqrt = sp.diags(d_inv_sqrt_row)
    D_v_inv_sqrt = sp.diags(d_inv_sqrt_col)

    return D_u_inv_sqrt @ B @ D_v_inv_sqrt

def adjacency(G: nx.Graph):
    B = biadjacency(G)
    return sp.bmat([[None, B],[B.T, None]], format='csr')

#B_test = sp.csr_matrix(([[1,0,1],[0,1,1]]))
#G = matrix_to_graph(B_test)
#print(normalized_biadjacency(G).todense())
#
#G = load_bipartite_graph(amazon_train)
#print(B)

