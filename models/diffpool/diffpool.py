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


def batch_diffpool(S, Z, A):
    """
    Computes diffpool operations for batches of S, Z and A. Using this function supposes that all graphs in the batch have the same number of nodes. This is the case for all diffpool layers except the first one.

    Parameters:
    -----------
    S : torch.Tensor
        Batch of S matrices of shape (batch_size x n_l x n_{l+1})
    Z : torch.Tensor
        Batch of Z matrices of shape (batch_size x n_l x d)
    A : torch.Tensor
        Batch of A matrices of shape (batch_size x n_l x n_l)

    Returns:
    --------
    X_out : torch.Tensor
        Batch of X matrices of shape (batch_size * n_{l+1} x d)
    A_out : torch.Tensor
        Batch of A matrices of shape (batch_size * n_{l+1} x batch_size * n_{l+1})
    """

    batch_size, size = S.shape[0], S.shape[2]

    X_out = torch.bmm(S.transpose(1, 2), Z)
    A_out = torch.bmm(torch.bmm(S.transpose(1, 2), A), S)

    return X_out.reshape(batch_size * size, -1), torch.block_diag(*A_out)


def ankward_diffpool(S, Z, A, batch):
    """
    Computes diffpool operations for batched S, Z and A. Using this function supposes that all graphs in the batch have different numbers of nodes and have been batched together. This is the case for the first diffpool layer.

    Parameters:
    -----------
    S : torch.Tensor
        Matrix S of shape (num_nodes x n_{l+1})
    Z : torch.Tensor
        Matrix Z of shape (num_nodes x d)
    A : torch.Tensor
        Matrix A of shape (num_nodes x num_nodes)
    batch : torch.Tensor
        Batch index of shape (num_nodes)

    Returns:
    --------
    X_out : torch.Tensor
        Matrix X of shape (num_nodes x d)
    A_out : torch.Tensor
        Matrix A of shape (num_nodes x num_nodes)
    """
    segment_indices = get_segment_indices(batch)
    X_out = torch.zeros(len(segment_indices), S.shape[1], Z.shape[1], device=S.device)
    A_out = torch.zeros(len(segment_indices), S.shape[1], S.shape[1], device=S.device)

    for i, (a, b) in enumerate(segment_indices):
        X_out[i] = torch.mm(S[a:b].transpose(0, 1), Z[a:b])
        A_out[i] = torch.mm(torch.mm(S[a:b].transpose(0, 1), A[a:b, a:b]), S[a:b])

    X_out = X_out.reshape(-1, Z.shape[1])
    A_out = torch.block_diag(*A_out)
    return X_out, A_out


def extract_blocks(A, size, batch_size):
    """
    A: (batch_size * size x batch_size * size)
    Returns: (batch_size x size x size)
    """

    # Extract the batch_size diagonal blocks of shape size x size
    # without for loop
    return A.reshape(batch_size, size, batch_size, size)[
        torch.arange(batch_size), :, torch.arange(batch_size), :
    ]


class DiffPoolEncoder(nn.Module):
    def __init__(
        self,
        d_features,
        d_out,
        d_pooling_layers,
        d_encoder_hidden_dims,
        d_encoder_linear_layers,
        d_encoder_num_heads,
        d_encoder_num_layers,
        d_linear,
        dropout,
    ):
        super(DiffPoolEncoder, self).__init__()
        self.pooling_sizes = d_pooling_layers

        self.pooling_layers = nn.ModuleList()
        self.embedding_layers = nn.ModuleList()

        for pooling_size, hidden_dim, linear_layers, num_heads, num_layers in zip(
            d_pooling_layers,
            d_encoder_hidden_dims,
            d_encoder_linear_layers,
            d_encoder_num_heads,
            d_encoder_num_layers,
        ):
            self.pooling_layers.append(
                GATEncoder(
                    d_features,
                    pooling_size,
                    d_hidden_dim=hidden_dim,
                    d_linear_layers=linear_layers,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            )
            self.embedding_layers.append(
                GATEncoder(
                    d_features,
                    d_features,
                    d_hidden_dim=hidden_dim,
                    d_linear_layers=linear_layers,
                    num_heads=num_heads,
                    num_layers=num_layers,
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

        # ========= First pooling layer =========
        S = torch.softmax(
            self.pooling_layers[0](X, A, batch),
            dim=-1,
        )
        Z = self.embedding_layers[0](X, A, batch)

        X, A = ankward_diffpool(S, Z, A, batch)

        # Update batch for next pooling layer. Assumes nodes are ordered by graph.
        batch = torch.tensor(
            [k for k in range(batch_size) for _ in range(self.pooling_sizes[0])]
        ).to(X.device)

        # ========= Other pooling layers =========
        for i, size in enumerate(self.pooling_sizes[1:]):
            S = self.pooling_layers[i + 1](X, A, batch).reshape(batch_size, -1, size)
            S = torch.softmax(
                S,
                dim=-1,
            )
            Z = self.embedding_layers[i + 1](X, A, batch).reshape(
                batch_size, self.pooling_sizes[i], -1
            )

            X, A = batch_diffpool(
                S,
                Z,
                extract_blocks(A, self.pooling_sizes[i], batch_size),
            )

            # Update batch for next pooling layer. Assumes nodes are ordered by graph.
            batch = torch.tensor(
                [k for k in range(batch_size) for _ in range(size)]
            ).to(X.device)

        # If the last pooling layer is 1, then the global mean pooling doesn't do anything
        return self.linear(global_mean_pool(X, batch))


class DiffPoolModel(nn.Module):
    def __init__(
        self,
        model_name,
        num_node_features,
        nout,
        d_pooling_layers=[15, 5, 1],
        d_encoder_hidden_dims=[600, 600, 600],
        d_encoder_linear_layers=[[300], [300], [300]],
        d_encoder_num_heads=[3, 3, 3],
        d_encoder_num_layers=[3, 3, 2],
        d_linear=1000,
        dropout=0.05,
    ):
        super(DiffPoolModel, self).__init__()
        self.graph_encoder = DiffPoolEncoder(
            num_node_features,
            nout,
            d_pooling_layers,
            d_encoder_hidden_dims,
            d_encoder_linear_layers,
            d_encoder_num_heads,
            d_encoder_num_layers,
            d_linear,
            dropout,
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
