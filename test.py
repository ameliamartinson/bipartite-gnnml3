import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

# Import from provided local files
from graph_loader import load_bipartite_graph, adjacency, normalized_biadjacency
import scipy.sparse.linalg as spla
from spectral import spectral_design_full, phi, h, g

# ==========================================
# 1. GNNML3 PyTorch Geometric Layer
# ==========================================
class GNNML3Layer(MessagePassing):
    def __init__(self, in_channels, out_channels_conv, out_channels_mul, num_supports, **kwargs):
        """
        GNNML3 Layer following Equations 5 and 6 of "Breaking the Limits of Message Passing Graph Neural Networks"
        """
        super(GNNML3Layer, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels_conv = out_channels_conv
        self.out_channels_mul = out_channels_mul
        self.S = num_supports

        # Eq 5: MLPs for edge features / convolution supports
        # mlp1, mlp2, mlp3: R^S -> R^S with Sigmoid activation
        self.mlp1 = nn.Sequential(nn.Linear(self.S, self.S), nn.Sigmoid())
        self.mlp2 = nn.Sequential(nn.Linear(self.S, self.S), nn.Sigmoid())
        self.mlp3 = nn.Sequential(nn.Linear(self.S, self.S), nn.Sigmoid())
        
        # mlp4 maps the concatenation to R^S. R^2S -> R^S with ReLU
        self.mlp4 = nn.Sequential(nn.Linear(2 * self.S, self.S), nn.ReLU())

        # Trainable parameters W for each support
        # Shape: [S, in_channels, out_channels_conv]
        self.W = nn.Parameter(torch.Tensor(self.S, in_channels, out_channels_conv))

        # Eq 6: MLPs for node features self-interaction
        self.mlp5 = nn.Sequential(nn.Linear(in_channels, out_channels_mul))
        self.mlp6 = nn.Sequential(nn.Linear(in_channels, out_channels_mul))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        for m in [self.mlp1, self.mlp2, self.mlp3, self.mlp4, self.mlp5, self.mlp6]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x, edge_index, C_prime):
        """
        x: Node feature matrix [N, in_channels]
        edge_index: Graph connectivity [2, E]
        C_prime: Extracted edge features from spectral prep [E, S]
        """
        # --- Step 1: Edge features transformation (Eq 5) ---
        out1 = self.mlp1(C_prime) # [E, S]
        out2 = self.mlp2(C_prime) # [E, S]
        out3 = self.mlp3(C_prime) # [E, S]

        # Concat mlp1 output and (mlp2 element-wise mlp3)
        concat_feat = torch.cat([out1, out2 * out3], dim=-1) # [E, 2S]
        C_tilde = self.mlp4(concat_feat) # [E, S] - Dynamically weighted convolution supports

        # --- Step 2: Linear transformation of Node Features ---
        # xW will have shape [S, N, out_channels_conv]
        xW = torch.matmul(x.unsqueeze(0), self.W) 
        
        # Flatten the features to pass through PyG's propagate function
        # Shape: [N, S * out_channels_conv]
        xW_reshaped = xW.transpose(0, 1).reshape(x.size(0), -1)

        # --- Step 3: Message passing over neighborhoods ---
        aggr_out = self.propagate(edge_index, x=xW_reshaped, C_tilde=C_tilde)

        # --- Step 4: Node feature self-interaction ---
        node_mul = self.mlp5(x) * self.mlp6(x) # [N, out_channels_mul]

        # --- Step 5: Combine & Activation (Eq 6) ---
        out = torch.cat([aggr_out, node_mul], dim=-1) # [N, out_channels_conv + out_channels_mul]
        return torch.relu(out)

    def message(self, x_j, C_tilde):
        # x_j: [E, S * out_channels_conv]
        # C_tilde: [E, S]
        E = C_tilde.size(0)
        
        # Reshape x_j back to compute individual support messages [E, S, out_channels_conv]
        x_j = x_j.view(E, self.S, self.out_channels_conv)

        # Scale transformed node features by the learned convolution edge weights
        msg = C_tilde.unsqueeze(-1) * x_j # [E, S, out_channels_conv]

        # Sum messages across all supports
        msg = msg.sum(dim=1) # [E, out_channels_conv]
        return msg


# ==========================================
# 2. Complete GNNML3 Model
# ==========================================
class GNNML3Model(nn.Module):
    def __init__(self, in_channels, hidden_channels_conv, hidden_channels_mul, out_channels, num_layers, num_supports):
        super(GNNML3Model, self).__init__()
        self.layers = nn.ModuleList()
        curr_in = in_channels
        
        for _ in range(num_layers):
            self.layers.append(
                GNNML3Layer(curr_in, hidden_channels_conv, hidden_channels_mul, num_supports)
            )
            # Output of a layer is the concatenation of the two paths
            curr_in = hidden_channels_conv + hidden_channels_mul

        self.readout = nn.Linear(curr_in, out_channels)

    def forward(self, x, edge_index, C_prime):
        # Pass through MPNN layers
        for layer in self.layers:
            x = layer(x, edge_index, C_prime)
        
        # Prediction head
        return self.readout(x)



def preprocess_gnnml3_data(G, S_supports=3, k_svd=6):
    """
    Memory-safe calculation of GNNML3 spectral edge features.
    Calculates features ONLY at the non-zero indices of the receptive mask.
    """
    print("Extracting bipartite structure...")
    # 1. Get the sparse biadjacency matrix
    B = normalized_biadjacency(G)
    m, n = B.shape
    N = m + n
    
    # 2. Receptive field mask: M = A + I
    A = adjacency(G)
    M = A + sp.eye(N, format='csr')
    M_coo = M.tocoo()
    
    # Edge indices from the mask
    rows = M_coo.row
    cols = M_coo.col
    E = len(rows)
    edge_index = np.vstack((rows, cols))

    print(f"Computing Truncated SVD (k={k_svd})...")
    # 3. Use sparse SVD (Requires significantly less RAM)
    U, S_vals, Vt = spla.svds(B, k=k_svd)
    V = Vt.T # Shape: (n, k_svd)
    
    lambda_max = S_vals.max()
    s_vals = np.linspace(-lambda_max, lambda_max, S_supports)
    
    print("Computing sparse spectral edge features...")
    # Pre-allocate C' [E, S]
    C_prime = np.zeros((E, S_supports), dtype=np.float32)
    
    # 4. Create boolean masks to identify which bipartite block each edge belongs to
    mask_11 = (rows < m)  & (cols < m)
    mask_12 = (rows < m)  & (cols >= m)
    mask_21 = (rows >= m) & (cols < m)
    mask_22 = (rows >= m) & (cols >= m)

    # 5. Calculate the spectral response element-wise for valid edges only
    for s_idx, f_s in enumerate(s_vals):
        # h() and g() from spectral.py applied to singular values
        h_S = h(S_vals, f_s=f_s) # Shape: (k_svd,)
        g_S = g(S_vals, f_s=f_s) # Shape: (k_svd,)
        
        # Block 11 (User-User)
        if np.any(mask_11):
            r, c = rows[mask_11], cols[mask_11]
            C_prime[mask_11, s_idx] = np.sum(U[r, :] * h_S * U[c, :], axis=1)
            
        # Block 12 (User-Item)
        if np.any(mask_12):
            r, c = rows[mask_12], cols[mask_12] - m
            C_prime[mask_12, s_idx] = np.sum(U[r, :] * g_S * V[c, :], axis=1)
            
        # Block 21 (Item-User)
        if np.any(mask_21):
            r, c = rows[mask_21] - m, cols[mask_21]
            C_prime[mask_21, s_idx] = np.sum(V[r, :] * g_S * U[c, :], axis=1)
            
        # Block 22 (Item-Item)
        if np.any(mask_22):
            r, c = rows[mask_22] - m, cols[mask_22] - m
            C_prime[mask_22, s_idx] = np.sum(V[r, :] * h_S * V[c, :], axis=1)

    print("Preprocessing complete.")
    return torch.tensor(edge_index, dtype=torch.long), torch.tensor(C_prime, dtype=torch.float32)

# ==========================================
# 4. Usage Example
# ==========================================
if __name__ == "__main__":
    # 1. Load data via graph_loader.py
    # NOTE: Using a small path mock to show it executes seamlessly
    import os 
    if os.path.exists("datasets/gowalla/train.txt"):
        G = load_bipartite_graph("datasets/gowalla/train.txt")
    else:
        # Fallback to random connected graph for demonstration if path is missing
        import networkx as nx
        G = nx.erdos_renyi_graph(20, 0.3)
        for i in G.nodes():
            G.nodes[i]['node_type'] = 'user' if i < 10 else 'item'

    S_supports = 3
    
    # 2. Compute edge_index and spectral features C'
    edge_index, C_prime = preprocess_gnnml3_data(G, S_supports=S_supports)
    N_total = edge_index.max().item() + 1
    
    # 3. Node features initialization 
    # Paper mentions setting H^(0) = 1 for structure-only graphs
    x = torch.ones((N_total, 1))

    # 4. Instantiate Model
    model = GNNML3Model(
        in_channels=1, 
        hidden_channels_conv=16, 
        hidden_channels_mul=16, 
        out_channels=10, 
        num_layers=2, 
        num_supports=S_supports
    )

    # 5. Forward Pass
    out = model(x, edge_index, C_prime)
    print(f"Forward Pass Output Shape: {out.shape}")  # Expected: [N_total, 10]