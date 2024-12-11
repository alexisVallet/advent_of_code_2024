from pathlib import Path
from dataclasses import dataclass
from itertools import product

import networkx as nx
from simple_parsing import ArgumentParser


@dataclass
class Args:
    input: Path


node = tuple[int, int]


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, "args")
    args: Args = parser.parse_args().args

    with args.input.open() as input_file:
        topo_map: list[list[int]] = [
            [int(c) for c in line.strip()] for line in input_file if len(line) > 1
        ]

    # Build up a graph
    num_rows = len(topo_map)
    num_cols = len(topo_map[0])

    neighbors: list[tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    edge_list: list[tuple[node, node]] = []
    start_nodes: list[node] = []

    for src_node in product(range(num_rows), range(num_cols)):
        src_i, src_j = src_node
        src_height = topo_map[src_i][src_j]
        if src_height == 0:
            start_nodes.append(src_node)
        for di, dj in neighbors:
            tgt_i = src_i + di
            tgt_j = src_j + dj
            if (
                0 <= tgt_i < num_rows
                and 0 <= tgt_j < num_cols
                and topo_map[tgt_i][tgt_j] - src_height == 1
            ):
                edge_list.append(
                    (
                        src_node,
                        (tgt_i, tgt_j),
                    )
                )
    graph = nx.DiGraph(edge_list)
    sum_scores: int = 0
    sum_ratings: int = 0

    for start_node in start_nodes:
        tree: nx.DiGraph = nx.dfs_tree(graph, start_node)
        end_nodes: list[node] = [
            (ni, nj) for ni, nj in tree.nodes if topo_map[ni][nj] == 9
        ]
        score = len(end_nodes)
        all_paths: list[list[node]] = list(
            nx.all_simple_paths(graph, start_node, end_nodes)
        )
        rating = len(all_paths)
        sum_scores += score
        sum_ratings += rating

    print(f"Question 1 answer: {sum_scores}")
    print(f"Question 2 answer: {sum_ratings}")


if __name__ == "__main__":
    main()
