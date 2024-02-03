import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import GAT

from models.baseline import TextEncoder


class GATEncoder2(nn.Module):
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
        super(GATEncoder2, self).__init__()
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
    for i, t in enumerate(T):
        if output.get(t) is None:
            output[t] = [i]
        else:
            if output[t][-1] != i and len(output[t]) > 1:
                output[t].pop()
            output[t].append(i)
    return output


def diffpool(S, A, X, idx=None, n_batches=32, size=None):
    S = torch.softmax(S, dim=-1)

    if idx is not None:
        # If idx is not None, it means that we are passing a batched S, A and X with different number of nodes per element.
        segment_indices = get_segment_indices(idx)
        X_out, A_out = [], []
        for k, v in segment_indices.items():
            if len(v) == 1:
                a, b = v[0], v[0] + 1
            else:
                assert len(v) == 2
                a, b = v[0], v[1] + 1
            S_, X_, A_ = S[a:b], X[a:b], A[a:b, a:b]
            X_out.append(torch.mm(S_.transpose(0, 1), X_))
            A_out.append(torch.mm(torch.mm(S_.transpose(0, 1), A_), S_))
        X_out = torch.stack(X_out, dim=0)  # [Batch x Size x Dim]
        A_out = torch.stack(A_out, dim=0)  # [Batch x Size x Size]
    else:
        assert size is not None
        X_out = torch.mm(S.transpose(0, 1), X)
        A_out = torch.mm(torch.mm(S.transpose(0, 1), A), S)
    return X_out, A_out


class DiffPoolEncoder(nn.Module):
    def __init__(
        self, num_node_features, nout, hidden_dim=600, embed_dim=500, dropout=0.1
    ):
        super(DiffPoolEncoder, self).__init__()
        self.gat_pool = []
        self.gat_embed = []

        self.gat_pool1 = GATEncoder2(num_node_features, 10)
        self.gat_embed1 = GATEncoder2(num_node_features, num_node_features)

        self.gat_pool2 = GATEncoder2(num_node_features, 4)
        self.gat_embed2 = GATEncoder2(num_node_features, num_node_features)

        self.final_gat = GATEncoder2(num_node_features, hidden_dim)

        self.fc1 = nn.Linear(num_node_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, nout)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_batch):
        X = graph_batch.x  # nodes features (nb_nodes, embeddings_size)
        edge_index = graph_batch.edge_index  # 'adjacency matrix' (2, nb_edges_in_batch)
        idx = graph_batch.batch  # in what graph is each node (nb_nodes)

        if edge_index.ndim == 1:
            raise ValueError("edge_index should be of shape (2, nb_edges_in_batch)")

        A = torch.zeros((X.shape[0], X.shape[0]), device=X.device)
        A[edge_index[0], edge_index[1]] = 1

        n_batches = idx[-1].item() + 1

        S = self.gat_pool1(X, A, idx)
        X = self.gat_embed1(X, A, idx)

        X, A = diffpool(
            S, A, X, idx
        )  # X : [Batch x Size x Dim] ; A : [Batch x Size x Size]

        X = torch.cat([X[i] for i in range(len(X))])
        A = torch.block_diag(*[A[i] for i in range(len(A))])
        idx = [i for i in range(n_batches) for _ in range(10)]
        idx = torch.tensor(idx)

        S = self.gat_pool2(X, A, idx)
        X = self.gat_embed2(X, A, idx)

        X, A = diffpool(S, A, X, idx)

        X = torch.cat([X[i] for i in range(len(X))])
        A = torch.block_diag(*[A[i] for i in range(len(A))])
        idx = [i for i in range(n_batches) for _ in range(4)]
        idx = torch.tensor(idx).to(X.device)

        out = self.final_gat(X, A, idx)

        out = global_mean_pool(X, idx)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        out = self.dropout(out)

        return out.to(X.device)


class DiffPoolModel(nn.Module):
    def __init__(
        self,
        model_name,
        num_node_features,
        nout,
    ):
        super(DiffPoolModel, self).__init__()
        self.graph_encoder = DiffPoolEncoder(
            num_node_features,
            nout,
        )
        self.text_encoder = TextEncoder(model_name)

    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded

    def get_text_encoder(self):
        return self.text_encoder

    def get_graph_encoder(self):
        return self.graph_encoder
