import torch
from torch import nn

from ..baseline import TextEncoder
from .library import PathNN


class PathNNEncoder(nn.Module):
    def __init__(
        self,
        num_node_features,
        nout,
        hidden_dim=150,
        dropout=0.1,
    ):
        super(PathNNEncoder, self).__init__()
        self.pathnn = PathNN(
            num_node_features,
            hidden_dim,
            10,  # make sure to set the same cutoff when preprocessing the data
            nout,
            dropout,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            residuals=True,
            encode_distances=False,  # set to False if using shortest paths
            l2_norm=False,
            predict=True,
        )

    def forward(self, graph_batch):
        return self.pathnn(graph_batch)


class PathNNModel(nn.Module):
    def __init__(
        self,
        model_name,
        num_node_features,
        nout,
    ):
        super(PathNNModel, self).__init__()
        self.graph_encoder = PathNNEncoder(
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
