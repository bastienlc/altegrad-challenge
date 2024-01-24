import torch
from torch import nn
from torch_geometric.nn import global_mean_pool

from ..baseline import TextEncoder
from .library import GAT


class DeepGATEncoder(nn.Module):
    def __init__(
        self,
        num_node_features,
        nout,
        mlp_hid=600,
        att_hidden_dim=300,
        att_out_dim=600,
        nheads=10,
        dropout=0.1,
        alpha=0.02,
        attention_depth=3,
    ):
        super(DeepGATEncoder, self).__init__()
        self.nhid = mlp_hid
        self.nout = nout
        self.gat = GAT(
            num_node_features,
            att_hidden_dim,
            att_out_dim,
            dropout,
            alpha,
            nheads,
            attention_depth,
        )
        self.mol_hidden1 = nn.Linear(att_out_dim, mlp_hid)
        self.mol_hidden2 = nn.Linear(mlp_hid, nout)

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


class DeepGATModel(nn.Module):
    def __init__(
        self,
        model_name,
        num_node_features,
        nout,
        attention_depth=3,
    ):
        super(DeepGATModel, self).__init__()
        self.graph_encoder = DeepGATEncoder(
            num_node_features, nout, attention_depth=attention_depth
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