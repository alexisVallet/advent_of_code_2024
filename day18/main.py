from pathlib import Path
from dataclasses import dataclass
from itertools import product

import networkx as nx
from simple_parsing import ArgumentParser
from parsita import ParserContext, reg, rep1sep


@dataclass
class Args:
    input: Path
    coord_range: int
    num_bytes: int


class Parser(ParserContext):
    number = reg(r"[0-9]+") > int
    byte_position = number << "," & number
    byte_positions = rep1sep(byte_position, "\n") << "\n" > (
        lambda ls: [tuple(l) for l in ls]
    )


Node = tuple[int, int]


def shortest_path_length(
    byte_positions: list[tuple[int, int]], coord_range: int
) -> int | None:
    neighbors: list[tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    edge_list: list[tuple[Node, Node]] = []
    byte_positions_set = frozenset(byte_positions)

    for i, j in product(range(coord_range + 1), range(coord_range + 1)):
        if (i, j) in byte_positions_set:
            continue
        for di, dj in neighbors:
            t = (i + di, j + dj)
            if t not in byte_positions_set:
                edge_list.append(((i, j), t))

    start = (0, 0)
    end = (coord_range, coord_range)
    graph = nx.DiGraph(edge_list)
    try:
        return nx.shortest_path_length(graph, source=start, target=end)
    except nx.exception.NetworkXNoPath:
        return None


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, "args")
    args: Args = parser.parse_args().args

    with args.input.open() as input_file:
        input_text = input_file.read()
    byte_positions: list[tuple[int, int]] = Parser.byte_positions.parse(
        input_text
    ).unwrap()

    print(
        f"Question 1 answer: {shortest_path_length(byte_positions[:args.num_bytes], args.coord_range)}"
    )

    for i in range(1, len(byte_positions)):
        if shortest_path_length(byte_positions[:i], args.coord_range) is None:
            print(f"Question 2 answer: {byte_positions[i-1]}")
            return


if __name__ == "__main__":
    main()
