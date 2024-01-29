# This code originates from https://github.com/gasmichel/PathNNs_expressive/
# and is licensed under the MIT license.
# ==========================================================================

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx


# This class defines custom mini batching
class DistancesData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if "index" in key:
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if "index" in key:
            return 1
        else:
            return 0


def add_shortest_distances(data, max_distance=1000):
    G = to_networkx(data, to_undirected=True)
    n = G.number_of_nodes()
    distances = dict(nx.shortest_path_length(G))
    # use two flat tensors to store the distances so that it can be stacked in batches
    flat_distances = torch.zeros((n * n), dtype=torch.int16)
    flat_distances_index = torch.zeros((2, n * n), dtype=torch.int)
    i = 0
    for v1 in range(n):
        for v2 in range(n):
            flat_distances_index[0, i] = v1
            flat_distances_index[1, i] = v2
            try:
                flat_distances[i] = distances[v1][v2]
            except KeyError:
                flat_distances[i] = max_distance
            i += 1
    setattr(data, f"distances_index", flat_distances_index)
    setattr(data, f"distances", flat_distances)

    return DistancesData(**data.stores[0])
