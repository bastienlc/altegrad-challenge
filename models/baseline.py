import torch
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import AutoModel, BitsAndBytesConfig


def mean_pooling(embeddings: torch.Tensor, attention_mask: torch.Tensor):
    """Mean Pooling - Take attention mask into account for correct averaging
    source : https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

    model_output: (batch_size, sequence_length, hidden_size)
    attention_mask: (batch_size, sequence_length)
    """
    hidden_size = embeddings.shape[2]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(-1, -1, hidden_size).float()
    )
    return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


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
        x = graph_batch.x  # nodes features (nb_graphs, embeddings_size)
        edge_index = graph_batch.edge_index  # 'adjacency matrix' (2, nb_edges_in_batch)
        batch = graph_batch.batch  # in what graph is each node (nb_graphs)
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
    def __init__(self, model_name, quantisize=False):
        super(TextEncoder, self).__init__()
        self.quantisize = quantisize
        self.use_sentence_transformer = model_name in [
            "sentence-transformers/all-MiniLM-L6-v2"
        ]
        if (
            quantisize
        ):  # Quantisizing speeds up inference and training, but I would not recommend it as it degrades the training quality too much
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                load_in_4bit=True,
                torch_dtype=torch.bfloat16,
            )
        else:
            self.model = AutoModel.from_pretrained(
                model_name,
            )

    def forward(self, input_ids, attention_mask):
        encoded_text = self.model(input_ids, attention_mask=attention_mask)

        if self.use_sentence_transformer:
            # In this case we can only use mean pooling
            return mean_pooling(encoded_text.last_hidden_state, attention_mask)
        else:
            return encoded_text.last_hidden_state[
                :, 0, :
            ]  # This is the CLS token i.e. it is the embedding of the whole sentence. Mean pooling gives similar results.


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
    graph_encoder.eval()
    text_encoder.eval()
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
