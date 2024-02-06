import numpy as np
import torch


def compute_similarities(metrics, skip, split):
    for k, metric in enumerate(metrics):
        if skip[k]:
            continue
        else:
            print(f"Processing model {k+1}")

        graph_embeddings = torch.load(
            f"./outputs/embeddings/{split}/graph_embeddings{k}.pt"
        )
        text_embeddings = torch.load(
            f"./outputs/embeddings/{split}/text_embeddings{k}.pt"
        )

        similarity = np.array(metric.similarity(text_embeddings, graph_embeddings))

        torch.save(similarity, f"./outputs/similarities/{split}/similarity{k}.pt")


def load_similarities(split, models_indices):
    similarities = []
    for k in models_indices:
        similarity = torch.load(f"./outputs/similarities/{split}/similarity{k}.pt")
        similarities.append(similarity)
    return np.array(similarities)
