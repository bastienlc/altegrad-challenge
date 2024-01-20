# This code originates from https://github.com/gasmichel/PathNNs_expressive/
# and is licensed under the MIT license.
# ==========================================================================

import igraph as ig
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx


class ModifData(Data):
    def __init__(self, edge_index=None, x=None, *args, **kwargs):
        super().__init__(x=x, edge_index=edge_index, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if "index" in key or "face" in key or "path" in key:
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if "index" in key or "face" in key:  # or "path" in key or "indicator" in key:
            return 1
        else:
            return 0


def fast_generate_paths2(g, cutoff, path_type, weights=None, undirected=True):
    """
    Generates paths for all nodes in the graph, based on specified path type. This function uses igraph rather than networkx
    to generate paths as it gives a more than 10x speedup.
    """
    if undirected and g.is_directed():
        g.to_undirected()

    path_length = np.array(g.distances())
    if path_type != "all_simple_paths":
        diameter = g.diameter(directed=False)
        diameter = diameter + 1 if diameter + 1 < cutoff else cutoff

    else:
        diameter = cutoff

    X = [[] for i in range(cutoff - 1)]
    sp_dists = [[] for i in range(cutoff - 1)]

    for n1 in range(g.vcount()):
        if path_type == "all_simple_paths":
            paths_ = g.get_all_simple_paths(n1, cutoff=cutoff - 1)

            for path in paths_:
                idx = len(path) - 2
                if len(path) > 0:
                    X[idx].append(path)
                    # Adding geodesic distance
                    sp_dist = []
                    for node in path:
                        sp_dist.append(path_length[n1, node])
                    sp_dists[idx].append(sp_dist)

        else:
            valid_ngb = [
                i
                for i in np.where(
                    (path_length[n1] <= cutoff - 1) & (path_length[n1] > 0)
                )[0]
                if i > n1
            ]
            for n2 in valid_ngb:
                if path_type == "shortest_path":
                    paths_ = g.get_shortest_paths(n1, n2, weights=weights)
                elif path_type == "all_shortest_paths":
                    paths_ = g.get_all_shortest_paths(n1, n2, weights=weights)

                for path in paths_:
                    idx = len(path) - 2
                    X[idx].append(path)
                    X[idx].append(list(reversed(path)))

    return X, diameter, sp_dists


def add_pathnn_data(data, path_type="all_simple_paths", cutoff=10):
    G = ig.Graph.from_networkx(to_networkx(data, to_undirected=False))
    setattr(data, f"path_2", data.edge_index.T.flip(1))

    graph_info = fast_generate_paths2(G, cutoff, path_type, undirected=False)

    if path_type == "all_simple_paths":
        if graph_info[2][0] == []:
            setattr(
                data,
                f"sp_dists_2",
                torch.zeros_like(data.edge_index.T.flip(1), dtype=torch.long),
            )
        else:
            setattr(data, f"sp_dists_2", torch.LongTensor(graph_info[2][0]).flip(1))

    for jj in range(1, cutoff - 1):
        paths = torch.LongTensor(graph_info[0][jj]).view(-1, jj + 2)
        distances = torch.LongTensor(graph_info[2][jj])

        setattr(data, f"path_{jj+2}", paths.flip(1))

        if path_type == "all_simple_paths":
            if distances.shape == torch.Size([0]):
                setattr(
                    data, f"sp_dists_{jj+2}", torch.zeros_like(paths, dtype=torch.long)
                )
            else:
                setattr(data, f"sp_dists_{jj+2}", distances.flip(1))

    return ModifData(**data.stores[0])
