from pathlib import Path
from dataclasses import dataclass
from itertools import count

import networkx as nx
import numpy as np
import numpy.typing as npt
from simple_parsing import ArgumentParser
from parsita import ParserContext, reg, rep1sep


@dataclass
class Args:
    input: Path
    size: tuple[int, int]
    start_step: int = 0
    connected_thresh: int = 200


Vec = npt.NDArray[np.int64]


@dataclass
class Robot:
    p: Vec
    v: Vec


class Parser(ParserContext):
    number = reg(r"-?[0-9]+") > int
    robot = ("p=" >> number << "," & number << " v=" & number << "," & number) > (
        lambda pv: Robot(
            p=np.array([pv[0], pv[1]], dtype=np.int64),
            v=np.array([pv[2], pv[3]], dtype=np.int64),
        )
    )
    robots = rep1sep(robot, "\n") << "\n"


def simulate(robots: list[Robot], seconds: int, size: Vec) -> list[Robot]:
    return [
        Robot(
            p=np.mod(r.p + r.v * seconds, size),
            v=r.v,
        )
        for r in robots
    ]


def safety_factor(robots: list[Robot], size: Vec) -> int:
    ul_sum: int = 0
    ur_sum: int = 0
    dl_sum: int = 0
    dr_sum: int = 0
    hx, hy = tuple(size // 2)

    for r in robots:
        rx, ry = tuple(r.p)
        if rx < hx and ry < hy:
            ul_sum += 1
        elif rx > hx and ry > hy:
            dr_sum += 1
        elif rx < hx and ry > hy:
            dl_sum += 1
        elif rx > hx and ry < hy:
            ur_sum += 1
    return ul_sum * ur_sum * dl_sum * dr_sum


def display_simulate(
    robots: list[Robot], size: Vec, start: int, connected_thresh: int
) -> None:
    w, h = tuple(size)
    robots = simulate(robots, start, size)
    neighbors: list[Vec] = [
        np.array(v)
        for v in [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    ]

    for t in count(start):
        # Display the current state
        print(f"{t=}")
        robot_p_set = frozenset([tuple(r.p) for r in robots])
        num_connected: int = 0

        for p in robot_p_set:
            if any([tuple(p + n) in robot_p_set for n in neighbors]):
                num_connected += 1

        if num_connected >= connected_thresh:
            for y in range(h):
                for x in range(w):
                    p = (x, y)
                    if p in robot_p_set:
                        print("#", end="")
                    else:
                        print(" ", end="")
                print("")
            _ = input()
        # Simulate a single step
        robots = simulate(robots, 1, size)


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, "args")
    args: Args = parser.parse_args().args
    h, w = args.size
    assert h % 2 == 1 and w % 2 == 1

    with args.input.open() as input_file:
        input_text = input_file.read()

    robots: list[Robot] = Parser.robots.parse(input_text).unwrap()
    size = np.array(args.size)
    simulated = simulate(robots, 100, size)
    factor = safety_factor(simulated, size)
    print(f"Question 1 answer: {factor}")
    display_simulate(robots, size, args.start_step, args.connected_thresh)


if __name__ == "__main__":
    main()
