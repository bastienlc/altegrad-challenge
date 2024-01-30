from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import CircleLoss, ContrastiveLoss, PNPLoss
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.loader import DataLoader as GeometricDataLoader

CEL = nn.CrossEntropyLoss()
CL = ContrastiveLoss(pos_margin=1, neg_margin=0, distance=CosineSimilarity())
PNP = PNPLoss(distance=CosineSimilarity())
CL = CircleLoss(m=0.4, gamma=80, distance=CosineSimilarity())


def cross_entropy_loss(v1, v2):
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CEL(logits, labels) + CEL(torch.transpose(logits, 0, 1), labels)


def contrastive_loss(v1, v2):
    labels = torch.arange(v1.shape[0], device=v1.device)
    return CL(torch.cat((v1, v2)), torch.cat((labels, labels)))


def pnp_loss(v1, v2):
    labels = torch.arange(v1.shape[0], device=v1.device)
    return PNP(torch.cat((v1, v2)), torch.cat((labels, labels)))


def circle_loss(v1, v2):
    labels = torch.arange(v1.shape[0], device=v1.device)
    return CL(torch.cat((v1, v2)), torch.cat((labels, labels)))


class Metrics:
    """Class used to compute the loss and score of the model, and to generate the solutions based on these metrics.

    Let's distinguish between two things :
    - The similarity, which is a function that takes two tensors and returns a scalar describing how similar they are (1) or not (0).
    - The loss, which takes a similarity matrix and true labels and returns a scalar describing how good the model is on this batch.
    """

    def __init__(self, loss: Union[str, None] = None) -> None:
        if loss == None or loss == "cross_entropy":
            self.loss = cross_entropy_loss
            self.similarity = cosine_similarity
        elif loss == "contrastive":
            self.loss = contrastive_loss
            self.similarity = CosineSimilarity()
        elif loss == "pnp":
            self.loss = pnp_loss
            self.similarity = CosineSimilarity()
        elif loss == "circle":
            self.loss = circle_loss
            self.similarity = CosineSimilarity()
        else:
            raise ValueError(f"Loss '{loss}' not implemented")

    def get(
        self,
        model: nn.Module,
        loader: GeometricDataLoader,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        return_loss=True,
        return_score=True,
    ):
        model.eval()

        graph_embeddings = []
        text_embeddings = []
        loss = 0
        result = {}

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

                if return_loss:
                    loss += self.loss(x_graph, x_text).item()

                for output in x_graph:
                    graph_embeddings.append(output.tolist())
                for output in x_text:
                    text_embeddings.append(output.tolist())

        if return_score:
            graph_embeddings, text_embeddings = (
                torch.Tensor(np.array(graph_embeddings)),
                torch.Tensor(np.array(text_embeddings)),
            )
            similarity = self.similarity(text_embeddings, graph_embeddings)
            result["score"] = label_ranking_average_precision_score(
                np.eye(len(similarity)), similarity
            )
        if return_loss:
            result["loss"] = loss / len(loader)

        return result

    def solution_from_embeddings(
        self,
        graph_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        save_to="solution.csv",
    ):
        similarity = self.similarity(text_embeddings, graph_embeddings)

        solution = pd.DataFrame(similarity)
        solution["ID"] = solution.index
        solution = solution[["ID"] + [col for col in solution.columns if col != "ID"]]
        solution.to_csv(f"outputs/{save_to}", index=False)
