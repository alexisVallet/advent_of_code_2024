from pathlib import Path
from dataclasses import dataclass

from simple_parsing import ArgumentParser


@dataclass
class Args:
    input: Path


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, "args")
    args: Args = parser.parse_args().args

    with args.input.open() as input_file:
        word_search = [l.strip() for l in input_file]

    x_pos: list[tuple[int, int]] = []

    for i, line in enumerate(word_search):
        for j, char in enumerate(line):
            if char == "X":
                x_pos.append((i, j))
    num_rows = len(word_search)
    num_cols = len(word_search[0])

    neighbor_patterns: list[tuple[int, int]] = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    num_xmas: int = 0

    for xi, xj in x_pos:
        for di, dj in neighbor_patterns:
            chars: list[str] = []
            for t in range(4):
                i = xi + di * t
                j = xj + dj * t
                if 0 <= i < num_rows and 0 <= j < num_cols:
                    chars.append(word_search[i][j])
            word = "".join(chars)
            if word == "XMAS":
                num_xmas += 1

    print(f"Question 1 answer: {num_xmas}")

    x_neighbor_patterns: list[tuple[int, int]] = [(1, 1), (-1, 1)]
    num_cross_mas: int = 0

    for ai in range(num_rows):
        for aj in range(num_cols):
            if word_search[ai][aj] == "A":
                diag_mas: list[bool] = []

                for di, dj in x_neighbor_patterns:
                    chars_set: set[str] = set()
                    for t in [-1, 1]:
                        i = ai + di * t
                        j = aj + dj * t
                        if 0 <= i < num_rows and 0 <= j < num_cols:
                            chars_set.add(word_search[i][j])
                    diag_mas.append(chars_set == set(["M", "S"]))
                if all(diag_mas):
                    num_cross_mas += 1

    print(f"Question 2 answer: {num_cross_mas}")


if __name__ == "__main__":
    main()
