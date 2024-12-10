from pathlib import Path
from dataclasses import dataclass
from operator import add, mul
from typing import Callable

from simple_parsing import ArgumentParser
from parsita import ParserContext, reg, rep1sep


@dataclass
class Args:
    input: Path


class Parser(ParserContext):
    number = reg(r"[0-9]+") > int
    equation = number << ": " & rep1sep(number, " ")
    equations = rep1sep(equation, "\n") << "\n"


def concat(i1: int, i2: int) -> int:
    return int(str(i1) + str(i2))


def operator_search(
    cur_res: int,
    remaining: list[int],
    target: int,
    operators: list[Callable[[int, int], int]],
) -> bool:
    if len(remaining) == 0:
        return cur_res == target
    for operator in operators:
        new_res = operator(cur_res, remaining[0])
        if new_res <= target:
            if operator_search(new_res, remaining[1:], target, operators):
                return True
    return False


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, "args")
    args: Args = parser.parse_args().args

    with args.input.open() as input_file:
        equations: list[tuple[int, list[int]]] = Parser.equations.parse(
            input_file.read()
        ).unwrap()

    sum_test_values: int = 0

    for target, numbers in equations:
        if operator_search(numbers[0], numbers[1:], target, [add, mul]):
            sum_test_values += target

    print(f"Question 1 answer: {sum_test_values}")

    sum_test_values = 0

    for target, numbers in equations:
        if operator_search(numbers[0], numbers[1:], target, [add, mul, concat]):
            sum_test_values += target

    print(f"Question 2 answer: {sum_test_values}")


if __name__ == "__main__":
    main()
