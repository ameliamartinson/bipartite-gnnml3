import torch
from gnnml3_pyg import build_pyg_data_from_txt, GNNML3

data = build_pyg_data_from_txt(
    "datasets/amazon-book/train.txt",
    x=None,                  # defaults to all-ones node features
    num_supports=4,          # total supports, including identity if enabled
    spectral_mode="full",    # or "trunc"
    include_identity_support=True,
)

model = GNNML3(
    in_channels=data.x.size(-1),
    hidden_channels=[64, 64, 64],
    num_supports=data.edge_attr.size(-1),
    dropout=0.1,
)

z = model(data)  # node embeddings
print(z.shape)