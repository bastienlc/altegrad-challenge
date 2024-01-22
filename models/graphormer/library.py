# This code originates from https://github.com/leffff/graphormer-pyg/
# and is licensed under the MIT license. Modified to remove edge features
# and fix multiple issues.
# =======================================================================

from typing import Tuple

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import degree


def decrease_to_max_value(x, max_value):
    x[x > max_value] = max_value
    return x


# Discard in/out degrees since we have undirected graphs
class CentralityEncoding(nn.Module):
    def __init__(self, max_degree: int, node_dim: int):
        """
        :param max_degree: max degree of nodes
        :param node_dim: hidden dimensions of node features
        """
        super().__init__()
        self.max_degree = max_degree
        self.node_dim = node_dim
        self.z = nn.Parameter(torch.randn((max_degree, node_dim)))

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_index: edge_index of graph (adjacency list)
        :return: torch.Tensor, node embeddings after Centrality encoding
        """
        num_nodes = x.shape[0]

        deg = decrease_to_max_value(
            degree(index=edge_index[0], num_nodes=num_nodes).long(), self.max_degree - 1
        )

        x += self.z[deg]

        return x


class SpatialEncoding(nn.Module):
    def __init__(self, max_path_distance: int):
        """
        :param max_path_distance: max pairwise distance between nodes
        """
        super().__init__()
        self.max_path_distance = max_path_distance

        self.b = nn.Parameter(torch.randn(self.max_path_distance))

    def forward(
        self, x: torch.Tensor, distances: torch.Tensor, distances_index: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param distances: (num_nodes * num_nodes) tensor of pairwise distances between nodes
        :param distances_index: (2, num_nodes * num_nodes) index of pairwise distances between nodes
        :return: torch.Tensor, spatial Encoding matrix
        """
        spatial_matrix = torch.zeros((x.shape[0], x.shape[0])).to(x.device)

        spatial_matrix[distances_index[0, :], distances_index[1, :]] = self.b[
            torch.clamp(distances, max=self.max_path_distance - 1).type(torch.int)
        ]

        return spatial_matrix


def dot_product(x1, x2) -> torch.Tensor:
    return (x1 * x2).sum(dim=1)


class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        """
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        """
        super().__init__()

        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        b: torch.Tensor,
        ptr,
    ) -> torch.Tensor:
        """
        :param query: node feature matrix
        :param key: node feature matrix
        :param value: node feature matrix
        :param b: spatial Encoding matrix
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after attention operation
        """
        batch_mask_zeros = ptr[:, None] == ptr[None, :]

        batch_mask_neg_inf = torch.full(
            size=(query.shape[0], query.shape[0]), fill_value=-1e6
        ).to(next(self.parameters()).device)

        batch_mask_neg_inf[batch_mask_zeros] = 1

        query = self.q(query)
        key = self.k(key)
        value = self.v(value)

        a = query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5
        a = (a + b) * batch_mask_neg_inf
        softmax = torch.softmax(a, dim=-1) * batch_mask_zeros
        x = softmax.mm(value)
        return x


# FIX: sparse attention instead of regular attention, due to specificity of GNNs(all nodes in batch will exchange attention)
class GraphormerMultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        dim_in: int,
        dim_q: int,
        dim_k: int,
    ):
        """
        :param num_heads: number of attention heads
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, x: torch.Tensor, b: torch.Tensor, ptr) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param b: spatial Encoding matrix
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after all attention heads
        """
        return self.linear(
            torch.cat(
                [attention_head(x, x, x, b, ptr) for attention_head in self.heads],
                dim=-1,
            )
        )


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim, n_heads):
        """
        :param node_dim: node feature matrix input number of dimension
        :param n_heads: number of attention heads
        """
        super().__init__()

        self.node_dim = node_dim
        self.n_heads = n_heads

        self.attention = GraphormerMultiHeadAttention(
            dim_in=node_dim,
            dim_k=node_dim,
            dim_q=node_dim,
            num_heads=n_heads,
        )
        self.ln_1 = nn.LayerNorm(node_dim)
        self.ln_2 = nn.LayerNorm(node_dim)
        self.ff = nn.Linear(node_dim, node_dim)

    def forward(
        self, x: torch.Tensor, b: torch, ptr
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h′(l) = MHA(LN(h(l−1))) + h(l−1)
        h(l) = FFN(LN(h′(l))) + h′(l)

        :param x: node feature matrix
        :param b: spatial Encoding matrix
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after Graphormer layer operations
        """
        x_prime = self.attention(self.ln_1(x), b, ptr) + x
        x_new = self.ff(self.ln_2(x_prime)) + x_prime

        return x_new


class Graphormer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_node_dim: int,
        node_dim: int,
        output_dim: int,
        n_heads: int,
        max_degree: int,
        max_path_distance: int,
    ):
        """
        :param num_layers: number of Graphormer layers
        :param input_node_dim: input dimension of node features
        :param node_dim: hidden dimensions of node features
        :param output_dim: number of output node features
        :param n_heads: number of attention heads
        :param max_degree: max degree of nodes
        :param max_path_distance: max pairwise distance between two nodes
        """
        super().__init__()

        self.num_layers = num_layers
        self.input_node_dim = input_node_dim
        self.node_dim = node_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.max_degree = max_degree
        self.max_path_distance = max_path_distance

        self.node_in_lin = nn.Linear(self.input_node_dim, self.node_dim)

        self.centrality_encoding = CentralityEncoding(
            max_degree=self.max_degree,
            node_dim=self.node_dim,
        )

        self.spatial_encoding = SpatialEncoding(
            max_path_distance=max_path_distance,
        )

        self.layers = nn.ModuleList(
            [
                GraphormerEncoderLayer(
                    node_dim=self.node_dim,
                    n_heads=self.n_heads,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.node_out_lin = nn.Linear(self.node_dim, self.output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """
        :param data: input graph of batch of graphs
        :return: torch.Tensor, output node embeddings
        """
        x = data.x.float()
        edge_index = data.edge_index.long()
        ptr = data.batch

        x = self.node_in_lin(x)

        x = self.centrality_encoding(x, edge_index)
        b = self.spatial_encoding(x, data.distances, data.distances_index)
        for layer in self.layers:
            x = layer(x, b, ptr)
        x = self.node_out_lin(x)

        return x
