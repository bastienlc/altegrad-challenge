import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import GAT

from ..baseline import TextEncoder


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

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch

        num_nodes = x.shape[0]

        output = torch.zeros(
            num_nodes, self.num_heads * self.d_hidden_dim, device=x.device
        )

        for i, head in enumerate(self.heads):
            output[:, i * self.d_hidden_dim : (i + 1) * self.d_hidden_dim] = head(
                x, edge_index, batch=batch
            )

        output = global_mean_pool(output, batch)
        for i, layer in enumerate(self.linear_layers):
            output = layer(output)
            if i < len(self.linear_layers) - 1:
                output = self.activation(output)
            output = nn.Dropout(self.dropout)(output)

        return output


class GATModel(nn.Module):
    def __init__(
        self,
        model_name,
        num_node_features,
        nout,
        d_hidden_dim=600,
        num_layers=3,
        num_heads=5,
        d_linear_layers=[
            1000,
        ],
        dropout=0.1,
        activation="ReLU",
    ):
        super(GATModel, self).__init__()
        self.graph_encoder = GATEncoder(
            num_node_features,
            nout,
            d_hidden_dim=d_hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            d_linear_layers=d_linear_layers,
            dropout=dropout,
            activation=activation,
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