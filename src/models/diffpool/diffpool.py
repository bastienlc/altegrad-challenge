import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from models.text_encoder import TextEncoder

from ..text_encoder import TextEncoder
from .encoder import GATEncoder
from .utils import ankward_diffpool, batch_diffpool, extract_blocks


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
