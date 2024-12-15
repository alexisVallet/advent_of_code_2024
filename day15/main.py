from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from itertools import chain, product
from copy import deepcopy

import numpy as np
import numpy.typing as npt
from simple_parsing import ArgumentParser
from parsita import ParserContext, rep1, rep1sep, lit


@dataclass
class Args:
    input: Path
    interactive: bool = False
    step: bool = False


class Tile(Enum):
    BOX = 0
    WALL = 1
    Robot = 2


Vec = npt.NDArray[np.float64]


@dataclass
class Puzzle:
    tiles_map: dict[tuple[float, float], Tile]
    robot_pos: Vec
    movement: list[Vec]


def parse_tile(c: str) -> Tile | None:
    match c:
        case "#":
            return Tile.WALL
        case "O":
            return Tile.BOX
        case "@":
            return Tile.Robot
        case _:
            return None


def parse_dir(c: str) -> Vec:
    def dir_tuple():
        match c:
            case "^":
                return (-1, 0)
            case "<":
                return (0, -1)
            case "v":
                return (1, 0)
            case ">":
                return (0, 1)
            case _:
                raise ValueError("This should never happen")

    return np.array(dir_tuple(), dtype=np.float64)


def parse_puzzle(tiles_and_dirs: tuple[list[list[Tile | None]], list[Vec]]) -> Puzzle:
    tiles, dirs = tiles_and_dirs
    tiles_map: dict[tuple[int, int], Tile] = {}
    robot_pos: None | tuple[int, int] = None

    num_rows = len(tiles)
    num_cols = len(tiles[0])

    for i, j in product(range(num_rows), range(num_cols)):
        tile_pos = (i, j)
        match tiles[i][j]:
            case None:
                continue
            case Tile.Robot:
                tile = Tile.Robot
                robot_pos = tile_pos
            case _tile:
                tile = _tile
        tiles_map[tile_pos] = tile

    assert robot_pos is not None

    return Puzzle(
        tiles_map=tiles_map,
        robot_pos=np.array(robot_pos, dtype=np.float64),
        movement=dirs,
    )


class Parser(ParserContext):
    tile = lit("#") | lit(".") | lit("O") | lit("@") > parse_tile
    tile_line = rep1(tile)
    tile_array = rep1sep(tile_line, "\n") << "\n\n"
    dir = lit("^") | lit("<") | lit("v") | lit(">") > parse_dir
    dirs = rep1sep(rep1(dir), "\n") > (lambda ls: list(chain(*ls)))
    puzzle = tile_array & dirs << "\n" > parse_puzzle


class CantMove(Exception):
    pass


def simulate_q1(puzzle: Puzzle) -> Puzzle:
    tiles_map = deepcopy(puzzle.tiles_map)
    cur_pos = puzzle.robot_pos

    def move_robot_or_box(tile_pos: Vec, dir: Vec) -> Vec:
        tgt_pos = tile_pos + dir

        match tiles_map.get(tuple(tgt_pos)):
            case Tile.WALL:
                raise CantMove()
            case None:
                pass
            case _:
                move_robot_or_box(tgt_pos, dir)
        tile = tiles_map.pop(tuple(tile_pos))
        tiles_map[tuple(tgt_pos)] = tile

        return tgt_pos

    for dir in puzzle.movement:
        try:
            cur_pos = move_robot_or_box(cur_pos, dir)
        except CantMove:
            pass

    return Puzzle(
        tiles_map=tiles_map,
        robot_pos=cur_pos,
        movement=puzzle.movement,
    )


def simulate_q2(puzzle: Puzzle, interactive: bool, step: bool) -> Puzzle:
    tiles_map = deepcopy(puzzle.tiles_map)
    cur_pos = puzzle.robot_pos

    def colliding_tiles(
        tile: Tile, tile_pos: Vec, orig_tile_pos: Vec
    ) -> list[tuple[Tile, Vec]]:
        pos_to_check: list[tuple[float, float]] = [(0, 0), (0, -0.5)]
        if tile == Tile.BOX:
            pos_to_check.append((0, 0.5))
        tiles: list[tuple[Tile, Vec]] = []

        for d in pos_to_check:
            check_p = tile_pos + np.array(d, dtype=np.float64)
            if np.all(check_p == orig_tile_pos):
                continue
            match tiles_map.get(tuple(check_p)):
                case None:
                    pass
                case tile:
                    if tile in [Tile.BOX, Tile.WALL]:
                        tiles.append((tile, check_p))
        return tiles

    def move_robot_or_box(tile_pos: Vec, dir: Vec) -> set[tuple[int, int]]:
        tgt_pos = tile_pos + dir
        tile = tiles_map[tuple(tile_pos)]
        children_moves: set[tuple[int, int]] = set()

        for coll_tile, coll_tile_pos in colliding_tiles(tile, tgt_pos, tile_pos):
            match coll_tile:
                case Tile.WALL:
                    raise CantMove()
                case _:
                    children_moves = children_moves.union(
                        move_robot_or_box(coll_tile_pos, dir)
                    )
        children_moves.add(tuple(tile_pos))

        return children_moves

    def interactive_dir_gen():
        while True:
            show_puzzle_q2(Puzzle(tiles_map, cur_pos, puzzle.movement))
            dir = input()
            match dir:
                case "w":
                    yield np.array([-1, 0], dtype=np.float64)
                case "a":
                    yield np.array([0, -1], dtype=np.float64)
                case "s":
                    yield np.array([1, 0], dtype=np.float64)
                case "d":
                    yield np.array([0, 1], dtype=np.float64)
                case _:
                    continue

    if interactive:
        dir_it = interactive_dir_gen()
    else:
        dir_it = puzzle.movement

    for dir in dir_it:
        # prev_state = deepcopy(tiles_map)
        dir *= np.array([1, 0.5], dtype=np.float64)
        try:
            to_move = move_robot_or_box(cur_pos, dir)
            to_add: list[tuple[Tile, Vec]] = []
            for tile_pos in to_move:
                tile = tiles_map.pop(tile_pos)
                new_pos = np.array(tile_pos, dtype=np.float64) + dir
                to_add.append((tile, new_pos))
            for tile, new_pos in to_add:
                tiles_map[tuple(new_pos)] = tile
                if tile == Tile.Robot:
                    cur_pos = new_pos
        except CantMove:
            pass
        except KeyError as e:
            # show_puzzle_q2(Puzzle(prev_state, cur_pos, puzzle.movement))
            print(f"{dir=}")
            show_puzzle_q2(
                Puzzle(tiles_map, cur_pos, puzzle.movement), to_point=e.args[0]
            )
            raise e
        if step:
            print(f"{dir=}")
            show_puzzle_q2(Puzzle(tiles_map, cur_pos, puzzle.movement))
            _ = input()

    return Puzzle(
        tiles_map=tiles_map,
        robot_pos=cur_pos,
        movement=puzzle.movement,
    )


def sum_box_coords(puzzle: Puzzle, is_q2: bool) -> int:
    return sum(
        100 * int(i) + int(j * 2 if is_q2 else 1) if tile == Tile.BOX else 0
        for (i, j), tile in puzzle.tiles_map.items()
    )


def show_puzzle_q2(puzzle: Puzzle, to_point: None | tuple[int, int] = None) -> None:
    coords = np.array([p for p in puzzle.tiles_map])
    min_i, min_j = np.min(coords, axis=0)
    max_i, max_j = np.max(coords, axis=0)

    assert min_j == 0
    assert min_i == 0
    min_i = int(min_i)
    max_i = int(max_i)
    min_j = int(min_j)
    max_j = int(max_j)

    to_print: list[list[str]] = [["."] * (2 * max_j) for _ in range(max_i + 1)]

    for i in range(min_i, max_i + 1):
        for j_int in range(min_j, 2 * max_j + 1):
            j = j_int / 2
            if (i, j) == to_point:
                chars: str = "X"
            else:
                match puzzle.tiles_map.get((i, j)):
                    case None:
                        continue
                    case Tile.BOX:
                        chars: str = "[]"
                    case Tile.WALL:
                        chars = "##"
                    case Tile.Robot:
                        chars = "@"
            to_print[i][j_int : j_int + len(chars)] = list(chars)
    for line in to_print:
        print("".join(line))


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, "args")
    args: Args = parser.parse_args().args

    with args.input.open() as input_file:
        input_text = input_file.read()

    puzzle: Puzzle = Parser.puzzle.parse(input_text).unwrap()
    simulated = simulate_q1(puzzle)
    sum_coords_q1 = sum_box_coords(simulated, is_q2=False)
    print(f"Question 1 answer: {sum_coords_q1}")
    simulated_q2 = simulate_q2(puzzle, interactive=args.interactive, step=args.step)
    sum_coords_q2 = sum_box_coords(simulated_q2, is_q2=True)
    print(f"Question 2 answer: {sum_coords_q2}")


if __name__ == "__main__":
    main()
