from functools import cmp_to_key
from typing import Callable, Union

import numpy as np
import torch
from tqdm import tqdm


def rank_norm(similarities: torch.Tensor):
    N = similarities.shape[1]

    ranks = N - similarities.argsort(axis=2).argsort(axis=2)

    return 1 - (ranks - 1) / N


def strange_norm(similarities: torch.Tensor):
    return (similarities - similarities.min(axis=1)[:, None, :]) / (
        similarities.max(axis=1)[:, None, :] - similarities.min(axis=1)[:, None, :]
    )


def min_max_norm(similarities: torch.Tensor):
    return (similarities - similarities.min(axis=2)[:, :, None]) / (
        similarities.max(axis=2)[:, :, None] - similarities.min(axis=2)[:, :, None]
    )


def sum_norm(similarities: torch.Tensor):
    return (similarities - similarities.min(axis=2)[:, :, None]) / (
        similarities.sum(axis=2)[:, :, None] - similarities.min(axis=2)[:, :, None]
    )


def zmuv_norm(similarities: torch.Tensor):
    return (similarities - similarities.mean(axis=2)[:, :, None]) / similarities.std(
        axis=2
    )[:, :, None]


def max_norm(similarities: torch.Tensor):
    return similarities / similarities.max(axis=2)[:, :, None]


def identity_norm(similarities: torch.Tensor):
    return similarities


def condorcet_fuse(
    similarities: torch.Tensor,
    weights: Union[np.ndarray, None] = None,
    norm: Callable = strange_norm,
):
    if weights is None:
        weights = np.ones(similarities.shape[0])

    N = similarities.shape[1]
    M = similarities.shape[0]
    aggregation = np.zeros((N, N))

    similarities = norm(similarities)

    for document in tqdm(range(N), leave=False):
        other_documents = similarities[:, document, :]

        def condorcet_compare(a, b):
            count = 0
            for model in range(M):
                if other_documents[model, a] > other_documents[model, b]:
                    count += weights[model]
                elif other_documents[model, a] < other_documents[model, b]:
                    count -= weights[model]
            if count > 0:
                return -1
            elif count < 0:
                return 1
            else:
                return 0

        # document from best to worst according to condorcet
        condorcet_sort = np.array(
            sorted(list(range(N)), key=cmp_to_key(condorcet_compare))
        )

        aggregation[document, condorcet_sort] = np.linspace(1, 0, N)

    return aggregation


def mean_fuse(
    similarities: torch.Tensor,
    weights: Union[np.ndarray, None] = None,
    norm: Callable = strange_norm,
):
    if weights is None:
        weights = np.ones(similarities.shape[0])

    similarities = norm(similarities)

    return np.average(similarities, axis=0, weights=weights)


def reciprocal_rank_fuse(
    similarities: torch.Tensor,
    weights: Union[np.ndarray, None] = None,
    norm: Callable = strange_norm,
    k: int = 10,
):
    if weights is None:
        weights = np.ones(similarities.shape[0])

    N = similarities.shape[1]
    aggregation = np.zeros((N, N))

    similarities = norm(similarities)

    for document in tqdm(range(N), leave=False):
        other_documents = -similarities[:, document, :]
        order = other_documents.argsort(axis=1)
        ranks = order.argsort(axis=1)

        aggregation[document, :] = np.average(1 / (ranks + k), axis=0, weights=weights)

    return aggregation


def prod_fuse(
    similarities: torch.Tensor,
    weights: Union[np.ndarray, None] = None,
    norm: Callable = strange_norm,
):
    if weights is None:
        weights = np.ones(similarities.shape[0])

    similarities = norm(similarities)

    return np.prod(similarities * weights[:, None, None], axis=0)
