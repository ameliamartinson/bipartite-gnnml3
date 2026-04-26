import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx
from torch_geometric.nn import MessagePassing

# Import from your local files
from graph_loader import load_bipartite_graph, normalized_biadjacency, adjacency
from spectral import phi, h, g

# ==========================================
# 1. Memory-Safe Preprocessing (OOM Fix)
# ==========================================
def preprocess_gnnml3_data(G, S_supports=3, k_svd=6):
    print("Extracting bipartite structure...")
    B = normalized_biadjacency(G)
    m, n = B.shape
    N = m + n
    
    A = adjacency(G)
    M = A + sp.eye(N, format='csr')
    M_coo = M.tocoo()
    
    rows, cols = M_coo.row, M_coo.col
    E = len(rows)
    edge_index = np.vstack((rows, cols))

    print(f"Computing Truncated SVD (k={k_svd}) to save RAM...")
    U, S_vals, Vt = spla.svds(B, k=k_svd)
    V = Vt.T 
    
    lambda_max = S_vals.max() if len(S_vals) > 0 else 1.0
    s_vals = np.linspace(-lambda_max, lambda_max, S_supports)
    
    print("Computing sparse spectral edge features...")
    C_prime = np.zeros((E, S_supports), dtype=np.float32)
    
    mask_11 = (rows < m)  & (cols < m)
    mask_12 = (rows < m)  & (cols >= m)
    mask_21 = (rows >= m) & (cols < m)
    mask_22 = (rows >= m) & (cols >= m)

    for s_idx, f_s in enumerate(s_vals):
        h_S = h(S_vals, f_s=f_s) 
        g_S = g(S_vals, f_s=f_s) 
        
        if np.any(mask_11):
            r, c = rows[mask_11], cols[mask_11]
            C_prime[mask_11, s_idx] = np.sum(U[r, :] * h_S * U[c, :], axis=1)
        if np.any(mask_12):
            r, c = rows[mask_12], cols[mask_12] - m
            C_prime[mask_12, s_idx] = np.sum(U[r, :] * g_S * V[c, :], axis=1)
        if np.any(mask_21):
            r, c = rows[mask_21] - m, cols[mask_21]
            C_prime[mask_21, s_idx] = np.sum(V[r, :] * g_S * U[c, :], axis=1)
        if np.any(mask_22):
            r, c = rows[mask_22] - m, cols[mask_22] - m
            C_prime[mask_22, s_idx] = np.sum(V[r, :] * h_S * V[c, :], axis=1)

    print("Preprocessing complete.")
    return torch.tensor(edge_index, dtype=torch.long), torch.tensor(C_prime, dtype=torch.float32)

# ==========================================
# 2. GNNML3 Architecture 
# ==========================================
class GNNML3Layer(MessagePassing):
    def __init__(self, in_channels, out_channels_conv, out_channels_mul, num_supports, **kwargs):
        super(GNNML3Layer, self).__init__(aggr='add', **kwargs)
        self.S = num_supports
        self.out_channels_conv = out_channels_conv

        self.mlp1 = nn.Sequential(nn.Linear(self.S, self.S), nn.Sigmoid())
        self.mlp2 = nn.Sequential(nn.Linear(self.S, self.S), nn.Sigmoid())
        self.mlp3 = nn.Sequential(nn.Linear(self.S, self.S), nn.Sigmoid())
        self.mlp4 = nn.Sequential(nn.Linear(2 * self.S, self.S), nn.ReLU())

        self.W = nn.Parameter(torch.Tensor(self.S, in_channels, out_channels_conv))

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
        out1 = self.mlp1(C_prime)
        out2 = self.mlp2(C_prime)
        out3 = self.mlp3(C_prime)

        concat_feat = torch.cat([out1, out2 * out3], dim=-1)
        C_tilde = self.mlp4(concat_feat) 

        xW = torch.matmul(x.unsqueeze(0), self.W) 
        xW_reshaped = xW.transpose(0, 1).reshape(x.size(0), -1)

        aggr_out = self.propagate(edge_index, x=xW_reshaped, C_tilde=C_tilde)
        node_mul = self.mlp5(x) * self.mlp6(x) 

        out = torch.cat([aggr_out, node_mul], dim=-1) 
        return torch.relu(out)

    def message(self, x_j, C_tilde):
        E = C_tilde.size(0)
        x_j = x_j.view(E, self.S, self.out_channels_conv)
        msg = C_tilde.unsqueeze(-1) * x_j 
        return msg.sum(dim=1)

class GNNML3Model(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_supports):
        super(GNNML3Model, self).__init__()
        self.layers = nn.ModuleList()
        curr_in = in_channels
        
        for _ in range(num_layers):
            self.layers.append(GNNML3Layer(curr_in, hidden_channels, hidden_channels, num_supports))
            curr_in = hidden_channels * 2

        self.readout = nn.Linear(curr_in, out_channels)

    def forward(self, x, edge_index, C_prime):
        for layer in self.layers:
            x = layer(x, edge_index, C_prime)
        return self.readout(x)

# ==========================================
# 3. Training Loop
# ==========================================
def train_run():
    # 1. Load Data
    print("Loading Graph...")
    try:
        # Tries to load your real dataset if the path exists
        G = load_bipartite_graph("datasets/amazon-book/train.txt")
    except FileNotFoundError:
        print("Dataset not found! Generating a random bipartite graph for testing...")
        G = nx.erdos_renyi_graph(1000, 0.05)
        for i in G.nodes():
            G.nodes[i]['node_type'] = 'user' if i < 500 else 'item'

    S_supports = 3
    num_epochs = 50
    lr = 0.01

    # 2. Preprocess
    edge_index, C_prime = preprocess_gnnml3_data(G, S_supports=S_supports, k_svd=32)
    
    # 3. Setup Device & Features
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using device: {device}")
    
    num_nodes = edge_index.max().item() + 1
    
    # Paper explicitly mentions using node degrees or 1s as initial features
    x = torch.ones((num_nodes, 1), dtype=torch.float32) 
    
    # Dummy Task: Predict if a node is a User (0) or Item (1)
    # In a real scenario, replace this with your actual target labels
    labels = torch.zeros(num_nodes, dtype=torch.long)
    for idx, (node, data) in enumerate(G.nodes(data=True)):
        if data.get('node_type') == 'item':
            labels[idx] = 1

    # Move data to GPU/CPU
    model = GNNML3Model(
        in_channels=1, 
        hidden_channels=16, 
        out_channels=2, # 2 Classes (User vs Item)
        num_layers=2, 
        num_supports=S_supports
    ).to(device)

    x = x.to(device)
    edge_index = edge_index.to(device)
    C_prime = C_prime.to(device)
    labels = labels.to(device)

    # 4. Setup Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 5. Execute Training
    print("\n--- Starting Training Run ---")
    model.train()
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        
        # Forward Pass
        out = model(x, edge_index, C_prime)
        
        # Calculate Loss
        loss = criterion(out, labels)
        
        # Backward Pass & Optimize
        loss.backward()
        optimizer.step()
        
        # Calculate Accuracy
        preds = out.argmax(dim=1)
        acc = (preds == labels).sum().item() / num_nodes
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}")

    print("Training Complete!")

if __name__ == "__main__":
    train_run()