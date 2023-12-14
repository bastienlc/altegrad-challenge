import torch
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import AutoModel


class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        super(GraphEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        # print(encoded_text.last_hidden_state.size())
        return encoded_text.last_hidden_state[:, 0, :]


class BaselineModel(nn.Module):
    def __init__(
        self, model_name, num_node_features, nout, nhid, graph_hidden_channels
    ):
        super(BaselineModel, self).__init__()
        self.graph_encoder = GraphEncoder(
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


def get_embeddings(
    graph_encoder: nn.Module,
    text_encoder: nn.Module,
    test_loader: DataLoader,
    test_text_loader: DataLoader,
    device: torch.device,
):
    graph_embeddings = []
    for batch in test_loader:
        for output in graph_encoder(batch.to(device)):
            graph_embeddings.append(output.tolist())

    text_embeddings = []
    for batch in test_text_loader:
        for output in text_encoder(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ):
            text_embeddings.append(output.tolist())

    return graph_embeddings, text_embeddings
