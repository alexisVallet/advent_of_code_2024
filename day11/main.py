from pathlib import Path
from dataclasses import dataclass
from functools import cache

from simple_parsing import ArgumentParser


@dataclass
class Args:
    input: Path


def blink_stone(n: int) -> list[int]:
    if n == 0:
        return [1]
    else:
        n_str = str(n)
        if len(n_str) % 2 == 0:
            split_point = len(n_str) // 2
            return [int(n_str[:split_point]), int(n_str[split_point:])]
        return [n * 2024]


@cache
def blink_stone_count_recursive(n: int, num_blinks: int) -> int:
    if num_blinks == 0:
        return 1
    else:
        new_stones = blink_stone(n)
        return sum(blink_stone_count_recursive(m, num_blinks - 1) for m in new_stones)


def blink_stones_count_recursive(ns: list[int], num_blinks: int) -> int:
    return sum(blink_stone_count_recursive(n, num_blinks) for n in ns)


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, "args")
    args: Args = parser.parse_args().args

    with args.input.open() as input_file:
        input_text = input_file.read().strip()
    numbers = [int(s) for s in input_text.split()]

    print(f"Question 1 answer: {blink_stones_count_recursive(numbers, 25)}")
    print(f"Question 2 answer: {blink_stones_count_recursive(numbers, 75)}")


if __name__ == "__main__":
    main()
