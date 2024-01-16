import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.loader import DataLoader as GeometricDataLoader

CE = nn.CrossEntropyLoss()


def solution_from_embeddings(graph_embeddings, text_embeddings, save_to="solution.csv"):
    similarity = cosine_similarity(text_embeddings, graph_embeddings)

    solution = pd.DataFrame(similarity)
    solution["ID"] = solution.index
    solution = solution[["ID"] + [col for col in solution.columns if col != "ID"]]
    solution.to_csv(f"outputs/{save_to}", index=False)


def contrastive_loss(v1, v2):
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)


def get_metrics(
    model: nn.Module,
    loader: GeometricDataLoader,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    model.eval()
    graph_embeddings = []
    text_embeddings = []
    loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch.input_ids
            batch.pop("input_ids")
            attention_mask = batch.attention_mask
            batch.pop("attention_mask")
            graph_batch = batch

            x_graph, x_text = model(
                graph_batch.to(device),
                input_ids.to(device),
                attention_mask.to(device),
            )

            loss += contrastive_loss(x_graph, x_text).item()

            for output in x_graph:
                graph_embeddings.append(output.tolist())
            for output in x_text:
                text_embeddings.append(output.tolist())

    similarity = cosine_similarity(text_embeddings, graph_embeddings)

    return (
        loss / len(loader),
        label_ranking_average_precision_score(np.eye(len(similarity)), similarity),
    )
