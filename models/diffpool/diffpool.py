import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from models.baseline import TextEncoder

from ..baseline import TextEncoder
from .encoder import GATEncoder


def get_segment_indices(batch):
    """
    Given a batch, returns a list of tuples (a, b) where a is the index of the first element of the segment and b is the index of the first element of the next segment.
    """
    x = batch[0]
    start = 0

    output = []
    for end, y in enumerate(batch[1:]):
        if y != x:
            output.append((start, end + 1))
            x = y
            start = end
    output.append((start, len(batch)))
    return output


# This is not the most efficient way to do this, but it's the easiest to implement
def diffpool(S, Z, A, batch):
    segment_indices = get_segment_indices(batch)
    X_out = torch.zeros(
        len(segment_indices), S.shape[1], Z.shape[1], device=S.device
    )  # [Batch x Size x Dim]
    A_out = torch.zeros(
        len(segment_indices), S.shape[1], S.shape[1], device=S.device
    )  # [Batch x Size x Size]

    for i, (a, b) in enumerate(segment_indices):
        X_out[i] = torch.mm(S[a:b].transpose(0, 1), Z[a:b])
        A_out[i] = torch.mm(torch.mm(S[a:b].transpose(0, 1), A[a:b, a:b]), S[a:b])

    X_out = X_out.reshape(-1, Z.shape[1])
    A_out = torch.block_diag(*A_out)
    return X_out, A_out


class DiffPoolEncoder(nn.Module):
    def __init__(
        self,
        d_features,
        d_out,
        d_pooling_layers=[10, 4, 1],
        d_linear=600,
        dropout=0.1,
    ):
        super(DiffPoolEncoder, self).__init__()
        self.pooling_sizes = d_pooling_layers

        self.pooling_layers = nn.ModuleList()
        self.embedding_layers = nn.ModuleList()

        for size in d_pooling_layers:
            self.pooling_layers.append(
                GATEncoder(
                    d_features,
                    size,
                    d_hidden_dim=150,
                    d_linear_layers=[300],
                    dropout=dropout,
                )
            )
            self.embedding_layers.append(
                GATEncoder(
                    d_features,
                    d_features,
                    d_hidden_dim=300,
                    d_linear_layers=[600],
                    dropout=dropout,
                )
            )

        self.linear = nn.Sequential(
            nn.Linear(d_features, d_linear),
            nn.ReLU(),
            nn.Linear(d_linear, d_out),
            nn.Dropout(dropout),
        )

    def forward(self, graph_batch):
        X = graph_batch.x
        A = torch.zeros((X.shape[0], X.shape[0]), device=X.device)
        A[graph_batch.edge_index[0], graph_batch.edge_index[1]] = 1

        batch = graph_batch.batch
        batch_size = torch.max(batch).item() + 1

        for i, size in enumerate(self.pooling_sizes):
            S = torch.softmax(
                self.pooling_layers[i](X, A, batch),
                dim=-1,
            )
            Z = self.embedding_layers[i](X, A, batch)

            # Apply diffpool to get new X and A
            X, A = diffpool(S, Z, A, batch)

            # Update batch for next pooling layer. Assumes nodes are ordered by graph.
            batch = torch.tensor(
                [i for i in range(batch_size) for _ in range(size)]
            ).to(X.device)

        # If the last pooling layer is 1, then the global mean pooling doesn't do anything
        return self.linear(global_mean_pool(X, batch))


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
