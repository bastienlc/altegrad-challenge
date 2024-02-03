from functools import cmp_to_key

import numpy as np
from tqdm import tqdm


def condorcet_fuse(similarities, weights=None):
    if weights is None:
        weights = np.ones(similarities.shape[0])

    N = similarities.shape[1]
    M = similarities.shape[0]
    aggregation = np.zeros((N, N))

    similarities = (similarities - similarities.min(axis=1)[:, None, :]) / (
        similarities.max(axis=1)[:, None, :] - similarities.min(axis=1)[:, None, :]
    )

    for document in tqdm(range(N)):
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


def mean_fuse(similarities, weights=None):
    if weights is None:
        weights = np.ones(similarities.shape[0])

    similarities = (similarities - similarities.min(axis=1)[:, None, :]) / (
        similarities.max(axis=1)[:, None, :] - similarities.min(axis=1)[:, None, :]
    )

    return np.average(similarities, axis=0, weights=weights)


def reciprocal_rank_fuse(similarities, weights=None, k=10):
    if weights is None:
        weights = np.ones(similarities.shape[0])

    N = similarities.shape[1]
    aggregation = np.zeros((N, N))

    similarities = (similarities - similarities.min(axis=1)[:, None, :]) / (
        similarities.max(axis=1)[:, None, :] - similarities.min(axis=1)[:, None, :]
    )

    for document in tqdm(range(N)):
        other_documents = -similarities[:, document, :]
        order = other_documents.argsort(axis=1)
        ranks = order.argsort(axis=1)

        aggregation[document, :] = np.average(1 / (ranks + k), axis=0, weights=weights)

    return aggregation
