import torch
from torch import nn
from torch_geometric.nn import global_mean_pool

from ..baseline import TextEncoder
from .pyGAT import GAT


class GATEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        super(GATEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.gat = GAT(num_node_features, nhid, graph_hidden_channels, 0.1, 0.01, 10)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x  # nodes features (nb_nodes, embeddings_size)
        edge_index = graph_batch.edge_index  # 'adjacency matrix' (2, nb_edges_in_batch)
        batch = graph_batch.batch  # in what graph is each node (nb_nodes)

        adj = torch.zeros((x.shape[0], x.shape[0]), device=x.device)
        adj[edge_index[0], edge_index[1]] = 1

        x = self.gat(x, adj)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x


class GATModel(nn.Module):
    def __init__(
        self, model_name, num_node_features, nout, nhid, graph_hidden_channels
    ):
        super(GATModel, self).__init__()
        self.graph_encoder = GATEncoder(
            num_node_features, nout, nhid, graph_hidden_channels
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