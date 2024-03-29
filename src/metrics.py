from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_metric_learning.distances import BaseDistance, CosineSimilarity
from pytorch_metric_learning.losses import CircleLoss, ContrastiveLoss, PNPLoss
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm


def cross_entropy_loss(v1: torch.Tensor, v2: torch.Tensor, **kwargs):
    CEL = nn.CrossEntropyLoss(**kwargs)
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CEL(logits, labels) + CEL(torch.transpose(logits, 0, 1), labels)


def contrastive_loss(
    v1: torch.Tensor,
    v2: torch.Tensor,
    pos_margin: float = 1.0,
    neg_margin: float = 0.0,
    distance: BaseDistance = CosineSimilarity(),
    **kwargs,
):
    CL = ContrastiveLoss(
        pos_margin=pos_margin,
        neg_margin=neg_margin,
        distance=distance,
        **kwargs,
    )
    labels = torch.arange(v1.shape[0], device=v1.device)
    return CL(torch.cat((v1, v2)), torch.cat((labels, labels)))


def pnp_loss(
    v1: torch.Tensor,
    v2: torch.Tensor,
    distance: BaseDistance = CosineSimilarity(),
    **kwargs,
):
    PNP = PNPLoss(distance=distance, **kwargs)
    labels = torch.arange(v1.shape[0], device=v1.device)
    return PNP(torch.cat((v1, v2)), torch.cat((labels, labels)))


def circle_loss(
    v1: torch.Tensor,
    v2: torch.Tensor,
    m: float = 0.1,
    gamma: float = 1.0,
    distance: BaseDistance = CosineSimilarity(),
    **kwargs,
):
    CL = CircleLoss(m=m, gamma=gamma, distance=distance, **kwargs)
    labels = torch.arange(v1.shape[0], device=v1.device)
    return CL(torch.cat((v1, v2)), torch.cat((labels, labels)))


class Metrics:
    def __init__(self, loss: Union[str, None] = None, **kwargs) -> None:
        if loss == None or loss == "cross_entropy":
            self.loss = lambda v1, v2: cross_entropy_loss(v1, v2, **kwargs)
            self.similarity = cosine_similarity
        elif loss == "contrastive":
            self.loss = lambda v1, v2: contrastive_loss(v1, v2, **kwargs)
            self.similarity = CosineSimilarity()
        elif loss == "pnp":
            self.loss = lambda v1, v2: pnp_loss(v1, v2, **kwargs)
            self.similarity = CosineSimilarity()
        elif loss == "circle":
            self.loss = lambda v1, v2: circle_loss(v1, v2, **kwargs)
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
        return_loss: bool = True,
        return_score: bool = True,
        verbose: bool = False,
    ):
        model.eval()

        graph_embeddings = []
        text_embeddings = []
        loss = 0
        result = {}

        with torch.no_grad():
            for batch in tqdm(loader, disable=not verbose):
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
        save_to: str = "solution.csv",
    ):
        similarity = self.similarity(text_embeddings, graph_embeddings)

        solution = pd.DataFrame(similarity)
        solution["ID"] = solution.index
        solution = solution[["ID"] + [col for col in solution.columns if col != "ID"]]
        solution.to_csv(f"outputs/{save_to}", index=False)
