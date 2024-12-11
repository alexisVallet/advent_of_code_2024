from pathlib import Path
from dataclasses import dataclass
from itertools import permutations, count

from simple_parsing import ArgumentParser


@dataclass
class Args:
    input: Path


def antinode_locations(
    antenna_locations: dict[str, list[tuple[int, int]]],
    map_size: tuple[int, int],
    min_t: int,
    max_t: int | None,
) -> set[tuple[int, int]]:
    out_locations: set[tuple[int, int]] = set()
    num_rows, num_cols = map_size

    for _, locations in antenna_locations.items():
        for (i1, j1), (i2, j2) in permutations(locations, r=2):
            if max_t is not None:
                t_it = range(min_t, max_t + 1)
            else:
                t_it = count(min_t)

            for t in t_it:
                i3 = i2 + t * (i2 - i1)
                j3 = j2 + t * (j2 - j1)
                if 0 <= i3 < num_rows and 0 <= j3 < num_cols:
                    out_locations.add((i3, j3))
                else:
                    break

    return out_locations


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, "args")
    args: Args = parser.parse_args().args

    with args.input.open() as input_file:
        map_lines: list[str] = [line.strip() for line in input_file if len(line) > 1]
    antenna_locations: dict[str, list[tuple[int, int]]] = {}
    num_rows = len(map_lines)
    num_cols = len(map_lines[0])

    for i in range(num_rows):
        for j in range(num_cols):
            frequency = map_lines[i][j]
            if frequency != ".":
                if frequency not in antenna_locations:
                    antenna_locations[frequency] = []
                antenna_locations[frequency].append((i, j))

    print(
        f"Question 1 answer: {len(antinode_locations(antenna_locations, (num_rows, num_cols), 1, 1))}"
    )
    print(
        f"Question 2 answer: {len(antinode_locations(antenna_locations, (num_rows, num_cols), 0, None))}"
    )


if __name__ == "__main__":
    main()
