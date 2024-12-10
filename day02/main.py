from pathlib import Path
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from simple_parsing import ArgumentParser


@dataclass
class Args:
    input: Path


def safe(report: npt.NDArray[np.int64]) -> bool:
    diff = report[1:] - report[:-1]
    return bool((np.all(diff < 0) or np.all(diff > 0)) and np.all(abs(diff) <= 3))


def safe_with_dampener(report: npt.NDArray[np.int64]) -> bool:
    if safe(report):
        return True
    for i in range(report.shape[0]):
        if safe(np.concatenate((report[:i], report[i + 1 :]))):
            return True
    return False


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, "args")
    args: Args = parser.parse_args().args

    reports: list[npt.NDArray[np.int64]] = []

    with args.input.open("r") as input_file:
        for line in input_file:
            reports.append(np.array([int(level) for level in line.strip().split()]))

    num_safe_reports = np.sum([safe(r) for r in reports], dtype=np.int64)
    print(f"Question 1 answer is: {num_safe_reports}")

    num_safe_reports_with_dampener = np.sum(
        [safe_with_dampener(r) for r in reports], dtype=np.int64
    )
    print(f"Question 2 answer is: {num_safe_reports_with_dampener}")


if __name__ == "__main__":
    main()
