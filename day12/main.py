from pathlib import Path
from dataclasses import dataclass
from itertools import product

import networkx as nx
from simple_parsing import ArgumentParser


@dataclass
class Args:
    input: Path


Node = tuple[int, int]


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, "args")
    args: Args = parser.parse_args().args

    with args.input.open() as input_file:
        plot_maps: list[str] = [line.strip() for line in input_file if len(line) > 1]

    edge_list: list[tuple[Node, Node]] = []
    num_rows = len(plot_maps)
    num_cols = len(plot_maps[0])

    neighbors: list[tuple[int, int]] = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]

    for src_i, src_j in product(range(num_rows), range(num_cols)):
        src_plant = plot_maps[src_i][src_j]
        # self-loop to ensure single nodes are also added
        edge_list.append(((src_i, src_j), (src_i, src_j)))
        for di, dj in neighbors:
            tgt_i = src_i + di
            tgt_j = src_j + dj

            if (
                0 <= tgt_i < num_rows
                and 0 <= tgt_j < num_cols
                and plot_maps[tgt_i][tgt_j] == src_plant
            ):
                edge_list.append((((src_i, src_j), (tgt_i, tgt_j))))

    graph = nx.DiGraph(edge_list)

    plots = list(nx.strongly_connected_components(graph))

    total_price_q1: int = 0
    total_price_q2: int = 0

    def num_fences(n: Node) -> int:
        return (10 - graph.degree(n)) // 2

    def rot90(d: tuple[int, int]) -> tuple[int, int]:
        di, dj = d
        return (dj, -di)

    def get_plot_type(t: Node) -> str:
        i, j = t
        if 0 <= i < num_rows and 0 <= j < num_cols:
            return plot_maps[i][j]
        else:
            return "O"

    def add(t1: tuple[int, int], t2: tuple[int, int]) -> tuple[int, int]:
        return (t1[0] + t2[0], t1[1] + t2[1])

    def num_sides_added(n: Node) -> int:
        ni, nj = n
        plot_type = plot_maps[ni][nj]

        def base_num_sides():
            match num_fences(n):
                case 4:
                    return 4
                case 3:
                    return 2
                case 2:
                    is_pipe: bool = False

                    for d in neighbors:
                        t = add(n, d)
                        di, dj = d
                        if get_plot_type(t) != plot_type:
                            if get_plot_type((ni - di, nj - dj)) != plot_type:
                                is_pipe = True
                            break
                    return 0 if is_pipe else 1
                case _:
                    return 0

        diag = [-1, 1]
        top = [-1, 0]
        right = [0, 1]
        num_diag_corners: int = 0

        for _ in range(4):
            diag_ = add(n, diag)
            top_ = add(n, top)
            right_ = add(n, right)

            if (
                get_plot_type(diag_) != plot_type
                and get_plot_type(top_) == plot_type
                and get_plot_type(right_) == plot_type
            ):
                num_diag_corners += 1
            diag = rot90(diag)
            top = rot90(top)
            right = rot90(right)
        _base_num_sides = base_num_sides()
        return _base_num_sides + num_diag_corners

    for plot in plots:
        plot: set[Node]
        area = len(plot)

        perimeter = sum(num_fences(node) for node in plot)
        num_sides = sum(num_sides_added(node) for node in plot)
        i, j = next(iter(plot))
        plot_type = plot_maps[i][j]
        total_price_q1 += area * perimeter
        total_price_q2 += area * num_sides

    print(f"Question 1 answer: {total_price_q1}")
    print(f"Question 2 answer: {total_price_q2}")


if __name__ == "__main__":
    main()
