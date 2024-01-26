import torch
from models.baseline import TextEncoder
import torch.nn as nn
from torch_geometric.nn.models import GAT
from torch_geometric.nn import global_mean_pool
from models.diffpool.utils import GATEncoder, diffpool


class DiffPoolEncoder(nn.Module):
    def __init__(self, num_node_features, nout, hidden_dim = 600, embed_dim = 500, dropout = 0.1):
        super(DiffPoolEncoder, self).__init__()
        self.gat_pool = []
        self.gat_embed = []

        self.gat_pool1 = GATEncoder(num_node_features, 10)
        self.gat_embed1 = GATEncoder(num_node_features, num_node_features)

        self.gat_pool2 = GATEncoder(num_node_features, 4)
        self.gat_embed2 = GATEncoder(num_node_features, num_node_features)

        self.final_gat = GATEncoder(num_node_features, hidden_dim)

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

        X, A = diffpool(S, A, X, idx) # X : [Batch x Size x Dim] ; A : [Batch x Size x Size]

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
