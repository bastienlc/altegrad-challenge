from torch import nn
from torch_geometric.nn import global_mean_pool

from ..baseline import TextEncoder
from .library import Graphormer


class GraphormerEncoder(nn.Module):
    def __init__(
        self,
        num_node_features,
        nout,
        nhid,
        graph_hidden_channels,
        layers=3,
        nheads=10,
    ):
        super(GraphormerEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.graphormer = Graphormer(
            layers,
            num_node_features,
            graph_hidden_channels,
            graph_hidden_channels,
            nheads,
            5,
            5,
            5,
        )
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = self.graphormer(graph_batch)
        x = global_mean_pool(x, graph_batch.batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x


class GraphormerModel(nn.Module):
    def __init__(
        self,
        model_name,
        num_node_features,
        nout,
        nhid,
        graph_hidden_channels,
        layers=3,
        nheads=10,
    ):
        super(GraphormerModel, self).__init__()
        self.graph_encoder = GraphormerEncoder(
            num_node_features,
            nout,
            nhid,
            graph_hidden_channels,
            layers=layers,
            nheads=nheads,
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
