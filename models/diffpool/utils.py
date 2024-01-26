import torch
from models.baseline import TextEncoder
import torch.nn as nn
from torch_geometric.nn.models import GAT
from torch_geometric.nn import global_mean_pool

class GATEncoder(nn.Module):
    def __init__(
        self,
        d_features,
        d_out,
        d_hidden_dim=600,
        num_layers=3,
        num_heads=3,
        d_linear_layers=[
            1000,
            500,
        ],  # In addition to the first layer num_heads * d_hidden_dim -> d_linear_layers[0] and the last layer d_linear_layers[-1] -> d_out
        dropout=0.1,
        activation="ReLU",
    ):
        super(GATEncoder, self).__init__()
        self.num_node_features = d_features
        self.nout = d_out
        self.d_hidden_dim = d_hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_linear_layers = d_linear_layers
        self.dropout = dropout
        self.activation = activation

        self.heads = []
        for _ in range(num_heads):
            self.heads.append(
                GAT(
                    in_channels=d_features,
                    hidden_channels=d_hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    v2=True,
                    act=activation,
                    norm="BatchNorm",
                )
            )
        self.heads = nn.ModuleList(self.heads)

        self.linear_layers = [nn.Linear(num_heads * d_hidden_dim, d_linear_layers[0])]
        for i in range(1, len(d_linear_layers)):
            self.linear_layers.append(
                nn.Linear(d_linear_layers[i - 1], d_linear_layers[i])
            )
        self.linear_layers.append(nn.Linear(d_linear_layers[-1], d_out))
        self.linear_layers = nn.ModuleList(self.linear_layers)

        if isinstance(activation, str):
            self.activation = getattr(nn, activation)()
        else:
            self.activation = activation

    def forward(self, x, adj, batch):
        edge_index = adj.nonzero().t().contiguous()

        num_nodes = x.shape[0]

        output = torch.zeros(
            num_nodes, self.num_heads * self.d_hidden_dim, device=x.device
        )

        for i, head in enumerate(self.heads):
            output[:, i * self.d_hidden_dim : (i + 1) * self.d_hidden_dim] = head(
                x, edge_index, batch=batch
            )

        for i, layer in enumerate(self.linear_layers):
            output = layer(output)
            if i < len(self.linear_layers) - 1:
                output = self.activation(output)
            output = nn.Dropout(self.dropout)(output)

        return output

def get_segment_indices(T):
    if type(T) is torch.Tensor:
        T = T.tolist()
    output = {}
    for i,t in enumerate(T):
        if output.get(t) is None:
            output[t] = [i]
        else:
            if output[t][-1] != i and len(output[t]) > 1:
                output[t].pop()
            output[t].append(i)
    return output

def diffpool(S, A, X, idx = None, n_batches = 32, size = None):
    S = torch.softmax(S, dim=-1)

    if idx is not None :
        # If idx is not None, it means that we are passing a batched S, A and X with different number of nodes per element.
        segment_indices = get_segment_indices(idx)
        X_out, A_out = [], []
        for k,v in segment_indices.items():
            if len(v) == 1:
                a, b = v[0], v[0]+1
            else :
                assert len(v) == 2
                a, b = v[0], v[1]+1
            S_, X_, A_ = S[a:b], X[a:b], A[a:b, a:b]
            X_out.append(torch.mm(S_.transpose(0, 1), X_))
            A_out.append(torch.mm(torch.mm(S_.transpose(0, 1), A_), S_))
        X_out = torch.stack(X_out, dim=0) #[Batch x Size x Dim]
        A_out = torch.stack(A_out, dim=0) #[Batch x Size x Size]
    else :
        assert size is not None
        X_out = torch.mm(S.transpose(0, 1), X)
        A_out = torch.mm(torch.mm(S.transpose(0, 1), A), S)
    return X_out, A_out
