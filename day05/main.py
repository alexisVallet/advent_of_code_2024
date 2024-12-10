from pathlib import Path
from dataclasses import dataclass

import networkx as nx
from simple_parsing import ArgumentParser
from parsita import ParserContext, reg, rep1sep


@dataclass
class Args:
    input: Path


@dataclass
class Update:
    seq: list[int]
    to_index: dict[int, int]


class Parser(ParserContext):
    number = reg(r"[0-9]+") > int
    rule = number << "|" & number
    rules = rep1sep(rule, "\n")
    update = rep1sep(number, ",") > (
        lambda ns: Update(seq=ns, to_index={n: i for i, n in enumerate(ns)})
    )
    updates = rep1sep(update, "\n")
    main = rules << "\n\n" & updates << "\n"


def check_rule(rule: tuple[int, int], update: Update) -> bool:
    before, after = rule
    if before not in update.to_index or after not in update.to_index:
        return True
    return update.to_index[before] < update.to_index[after]


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, "args")
    args: Args = parser.parse_args().args

    with args.input.open() as input_file:
        input_text = input_file.read()
    rules: list[tuple[int, int]]
    updates: list[Update]
    rules, updates = Parser.main.parse(input_text).unwrap()

    total_middle: int = 0
    incorrect_updates: list[Update] = []

    for update in updates:
        if all(check_rule(r, update) for r in rules):
            total_middle += update.seq[len(update.seq) // 2]
        else:
            incorrect_updates.append(update)

    print(f"Question 1 answer is: {total_middle}")

    fixed_total_middle: int = 0

    for update in incorrect_updates:
        # We build the entire graph of the ordering rules and
        # compute a topological ordering. Then we sort the updates by looking
        # up the layer of each node in that topological ordering.
        rules_graph = nx.DiGraph(
            [(i, j) for i, j in rules if i in update.to_index and j in update.to_index]
        )
        node_to_layer: dict[int, int] = {}

        for layer, nodes in enumerate(nx.topological_generations(rules_graph)):
            for node in nodes:
                node_to_layer[node] = layer

        ordered = sorted(update.seq, key=lambda i: node_to_layer[i])
        fixed_total_middle += ordered[len(ordered) // 2]

    print(f"Question 2 answer is: {fixed_total_middle}")


if __name__ == "__main__":
    main()
