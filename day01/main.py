from pathlib import Path
from dataclasses import dataclass

from simple_parsing import ArgumentParser
import numpy as np


@dataclass
class Args:
    input: Path


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, "args")
    args: Args = parser.parse_args().args

    l1: list[int] = []
    l2: list[int] = []

    with args.input.open("r") as input_file:
        for line in input_file:
            n1, n2 = line.strip().split()
            l1.append(int(n1))
            l2.append(int(n2))
    a1 = np.array(l1)
    a2 = np.array(l2)
    a1 = np.sort(a1)
    a2 = np.sort(a2)
    total_dist = np.sum(np.abs(a1 - a2))
    print(f"Question 1 answer is: {total_dist}")

    counts = np.bincount(a2, minlength=a1[-1] + 1)
    similarity = np.sum(a1 * counts[a1])

    print(f"Question 2 answer is: {similarity}")


if __name__ == "__main__":
    main()
