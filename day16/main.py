from pathlib import Path
from dataclasses import dataclass
from itertools import product, chain, pairwise

import networkx as nx
from simple_parsing import ArgumentParser


@dataclass
class Args:
    input: Path


Vec = tuple[int, int]
Node = tuple[int, int, Vec]


def rot90(v: Vec) -> Vec:
    i, j = v
    return (-j, i)


def rot_minus90(v: Vec) -> Vec:
    i, j = v
    return (j, -i)


def show_nodes(map_array: list[str], nodes: frozenset[Vec]):
    num_rows = len(map_array)
    num_cols = len(map_array[0])

    for i in range(num_rows):
        for j in range(num_cols):
            if (i, j) in nodes:
                print("O", end="")
            else:
                print(map_array[i][j], end="")
        print("")


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, "args")
    args: Args = parser.parse_args().args

    with args.input.open() as input_file:
        map_array: list[str] = [line.strip() for line in input_file if len(line) > 1]

    edge_list: list[tuple[Node, Node, int]] = []
    start: None | Node = None
    end: list[Node] = []

    num_rows = len(map_array)
    num_cols = len(map_array[0])

    for i, j in product(range(num_rows), range(num_cols)):
        if map_array[i][j] == "#":
            continue

        for dir in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # can always turn 90 degree for a cost of 1000
            for trans in [rot90, rot_minus90]:
                edge_list.append(((i, j, dir), (i, j) + (trans(dir),), 1000))
            # if next node is not a wall or out of bounds, can
            # move forward for a cost of 1
            di, dj = dir
            ti = i + di
            tj = j + dj
            if 0 <= ti < num_rows and 0 <= tj < num_cols and map_array[ti][tj] != "#":
                edge_list.append(((i, j, dir), (ti, tj, dir), 1))
            if map_array[i][j] == "E":
                end.append((i, j, dir))
        if map_array[i][j] == "S":
            start = (i, j, (0, 1))

    assert start is not None
    assert len(end) == 4

    graph = nx.DiGraph()
    for n1, n2, w in edge_list:
        graph.add_edge(n1, n2, weight=w)
    all_shortest_paths: list[list[Node]] = list(
        chain(
            *[
                nx.all_shortest_paths(
                    graph,
                    source=start,
                    target=e,
                    weight="weight",
                )
                for e in end
            ]
        )
    )
    edge_weights: dict[tuple[Node, Node], int] = nx.get_edge_attributes(graph, "weight")
    all_shortest_paths_with_length: list[tuple[list[Node], int]] = [
        (path, sum(edge_weights[e] for e in pairwise(path)))
        for path in all_shortest_paths
    ]
    min_score: int = min(all_shortest_paths_with_length, key=lambda t: t[1])[1]
    print(f"Question 1 answer: {min_score}")

    shortest_path_nodes = frozenset(
        [
            (i, j)
            for (i, j, _) in chain(
                *[p for (p, l) in all_shortest_paths_with_length if l == min_score]
            )
        ]
    )
    show_nodes(map_array, shortest_path_nodes)
    print(f"Question 2 answer: {len(shortest_path_nodes)}")


if __name__ == "__main__":
    main()
