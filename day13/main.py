from pathlib import Path
from dataclasses import dataclass
import math
from fractions import Fraction

import numpy as np
import numpy.typing as npt
from numpy.linalg import norm
from simple_parsing import ArgumentParser
from parsita import ParserContext, reg, rep1sep
from scipy.optimize import milp, LinearConstraint, Bounds, OptimizeResult


@dataclass
class Args:
    input: Path


@dataclass
class ClawMachine:
    # All of these have shape [2,]
    A: npt.NDArray[np.int64]
    B: npt.NDArray[np.int64]
    T: npt.NDArray[np.int64]


class Parser(ParserContext):
    number = reg(r"[0-9]+") > int
    button = reg(r"Button [A-B]: ") >> "X+" >> number & ", Y+" >> number
    claw_machine = (
        button
        & ("\n" >> button)
        & ("\nPrize: X=" >> (number & ", Y=" >> number) << "\n")
    ) > (
        lambda vecs: ClawMachine(
            A=np.array(vecs[0], dtype=np.int64),
            B=np.array(vecs[1], dtype=np.int64),
            T=np.array(vecs[2], dtype=np.int64),
        )
    )
    claw_machines = rep1sep(claw_machine, "\n")


def solve(machine: ClawMachine, upper_bound: int | None) -> int | None:
    c = np.array([3, 1])
    A = np.stack((machine.A, machine.B), axis=1)
    if upper_bound is not None:
        bounds: Bounds | None = Bounds(
            lb=np.zeros_like(c),
            ub=upper_bound * np.ones_like(c),
        )
    else:
        bounds = None
    res: OptimizeResult = milp(
        c=c,
        integrality=np.ones_like(c),
        bounds=bounds,
        constraints=LinearConstraint(
            A=A,
            lb=machine.T,
            ub=machine.T,
        ),
    )
    match res.status:
        case 0:
            return int(res.fun)
        case 2:
            return None
        case _:
            raise ValueError(f"Failed to optimize: {res=}")


def _solve_q2(
    T: npt.NDArray[np.int64], A: npt.NDArray[np.int64], B: npt.NDArray[np.int64]
) -> tuple[int, int] | None:
    l = 0
    r = int(math.ceil(np.max(T / A)))

    while l <= r:
        a = (l + r) // 2
        # find the the b value that would make it reach T
        b_fraction = Fraction(T[0] - a * A[0], B[0])
        candidate_y_fraction = Fraction(a * A[1]) + b_fraction * Fraction(B[1])
        # If we don't reach T, check whether we overshot the y coordinate or not.
        # Using Fraction rather than floating point to avoid numerical issues with large T.
        y_diff = Fraction(T[1]) - candidate_y_fraction
        if y_diff > 0:
            l = a + 1
        elif y_diff < 0:
            r = a - 1
        else:
            if b_fraction.numerator % b_fraction.denominator == 0:
                b_int = b_fraction.numerator // b_fraction.denominator
                return a, b_int
            else:
                return None

    return None


def solve_q2(machine: ClawMachine, upper_bound: int | None = None) -> int | None:
    # First check that A and B are on either side of T. If it's not the
    # case it's not solvable and we can return early.
    # for all intents and purposes T is at a 45 degree angle.
    angle_T = np.arccos(machine.T[0] / norm(machine.T))
    angle_A = np.arccos(machine.A[0] / norm(machine.A))
    angle_B = np.arccos(machine.B[0] / norm(machine.B))

    (small_name, angle_small), (_, angle_large) = sorted(
        [("A", angle_A), ("B", angle_B)], key=lambda t: t[1]
    )

    if not angle_small < angle_T < angle_large:
        return None

    if small_name == "B":
        u = machine.A
        v = machine.B
    else:
        u = machine.B
        v = machine.A

    res = _solve_q2(machine.T, u, v)
    if res is None:
        return None

    alpha, beta = res

    if small_name == "B":
        a = alpha
        b = beta
    else:
        a = beta
        b = alpha

    if upper_bound is not None and a > upper_bound and b > upper_bound:
        return None

    return 3 * a + b


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, "args")
    args: Args = parser.parse_args().args

    with args.input.open() as input_file:
        input_text = input_file.read()
    claw_machines: list[ClawMachine] = Parser.claw_machines.parse(input_text).unwrap()

    total_num_tokens_q1: int = 0

    for machine in claw_machines:
        solve_res = solve(machine, 100)
        match solve_res:
            case None:
                pass
            case cost:
                total_num_tokens_q1 += cost
        solve_q2_res = solve_q2(machine, 100)
        if solve_q2_res != solve_res:
            print(f"{machine=}, {solve_res=}, {solve_q2_res=}")

    print(f"Question 1 answer: {total_num_tokens_q1}")

    total_num_tokens_q2: int = 0

    for machine in claw_machines:
        machine.T += 10000000000000

        match solve_q2(machine):
            case None:
                pass
            case cost:
                total_num_tokens_q2 += cost

    print(f"Question 2 answer: {total_num_tokens_q2}")


if __name__ == "__main__":
    main()
