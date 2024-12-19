from pathlib import Path
from dataclasses import dataclass
from functools import cache

from simple_parsing import ArgumentParser
from parsita import ParserContext, reg, rep1sep


@dataclass
class Args:
    input: Path


@dataclass
class Input:
    towels: list[str]
    designs: list[str]


class Parser(ParserContext):
    colors = reg(r"[w|u|b|r|g]+")
    towels = rep1sep(colors, ", ")
    designs = rep1sep(colors, "\n")
    input = towels << "\n\n" & designs << "\n" > (
        lambda towels_and_designs: Input(towels_and_designs[0], towels_and_designs[1])
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, "args")
    args: Args = parser.parse_args().args

    with args.input.open() as input_file:
        input_text = input_file.read()
    input: Input = Parser.input.parse(input_text).unwrap()

    @cache
    def search_first_recursive(remaining_design: str) -> list[str] | None:
        if remaining_design == "":
            return []
        for towel in input.towels:
            if remaining_design.startswith(towel):
                recurse_out = search_first_recursive(remaining_design[len(towel) :])
                if recurse_out is not None:
                    return [towel] + recurse_out

    num_possible = len(
        [
            design
            for design in input.designs
            if search_first_recursive(design) is not None
        ]
    )

    print(f"Question 1 answer: {num_possible=}")

    @cache
    def num_solutions(remaining_design: str) -> int:
        if remaining_design == "":
            return 1
        total_num_solutions = 0
        for towel in input.towels:
            if remaining_design.startswith(towel):
                total_num_solutions += num_solutions(remaining_design[len(towel) :])
        return total_num_solutions

    print(
        f"Question 2 answer: {sum(num_solutions(design) for design in input.designs)}"
    )


if __name__ == "__main__":
    main()
