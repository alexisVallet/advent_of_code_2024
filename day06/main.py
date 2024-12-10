from pathlib import Path
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from simple_parsing import ArgumentParser


@dataclass
class Args:
    input: Path


def visited_positions(
    map: npt.NDArray[np.bool], start_pos: tuple[int, int]
) -> set[tuple[int, int]] | None:
    cur_dir: tuple[int, int] = (-1, 0)
    cur_pos: tuple[int, int] = start_pos
    visited_pos_dir: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    while True:
        if (cur_pos, cur_dir) in visited_pos_dir:
            return None
        visited_pos_dir.add((cur_pos, cur_dir))
        i, j = cur_pos
        di, dj = cur_dir
        ti = i + di
        tj = j + dj
        if not (0 <= ti < map.shape[0]) or not (0 <= tj < map.shape[1]):
            break
        if map[ti, tj]:
            cur_dir = (dj, -di)
            continue
        cur_pos = (ti, tj)

    return set(pos for pos, _ in visited_pos_dir)


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, "args")
    args: Args = parser.parse_args().args

    with args.input.open() as input_file:
        map_list: list[str] = []

        for line in input_file:
            line = line.strip()
            if len(line) > 0:
                map_list.append(line)
    num_rows = len(map_list)
    num_cols = len(map_list[0])
    map_array: npt.NDArray[np.bool] = np.zeros(
        shape=(num_rows, num_cols), dtype=np.bool
    )
    start_pos: None | tuple[int, int] = None

    for i in range(num_rows):
        for j in range(num_cols):
            if map_list[i][j] == "#":
                map_array[i, j] = True
            elif map_list[i][j] == "^":
                start_pos = (i, j)

    assert start_pos is not None

    positions = visited_positions(map_array, start_pos)
    assert positions is not None
    print(f"Question 1 answer: {len(positions)}")
    num_loop_pos: int = 0

    for oi, oj in positions:
        new_map = map_array.copy()
        new_map[oi, oj] = True
        if visited_positions(new_map, start_pos) is None:
            num_loop_pos += 1
    print(f"Question 2 answer: {num_loop_pos}")


if __name__ == "__main__":
    main()
