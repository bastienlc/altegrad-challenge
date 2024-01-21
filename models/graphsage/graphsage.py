from torch import nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import GraphSAGE

from ..baseline import TextEncoder


class SAGEncoder(nn.Module):
    def __init__(
        self,
        num_node_features,
        nout,
        hidden_dim=150,
        num_layers=15,
        dropout=0.1,
    ):
        super(SAGEncoder, self).__init__()
        self.nout = nout
        self.graph_encoder = GraphSAGE(
            num_node_features,
            hidden_dim,
            num_layers,
            out_channels=nout,
            dropout=dropout,
        )

    def forward(self, graph_batch):
        x = graph_batch.x  # nodes features (nb_nodes, embeddings_size)
        edge_index = graph_batch.edge_index  # 'adjacency matrix' (2, nb_edges_in_batch)
        batch = graph_batch.batch  # in what graph is each node (nb_nodes)

        x = self.graph_encoder.forward(x, edge_index, batch=batch)
        x = global_mean_pool(x, batch)
        return x


class SAGEModel(nn.Module):
    def __init__(
        self,
        model_name,
        num_node_features,
        nout,
    ):
        super(SAGEModel, self).__init__()
        self.graph_encoder = SAGEncoder(
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
