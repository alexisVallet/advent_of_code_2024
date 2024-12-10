from pathlib import Path
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from simple_parsing import ArgumentParser
from parsita import ParserContext, lit, reg, rep1, until, eof


@dataclass
class Args:
    input: Path


class Q1Parser(ParserContext, whitespace=r"\s*"):
    number = reg(r"[0-9]+") > int
    mul = "mul(" >> number << "," & number << ")"
    muls = rep1(until(mul) >> mul) << (reg(r".*") >> eof)


class Q2Parser(ParserContext, whitespace=r"\s*"):
    number = reg(r"[0-9]+") > int
    mul = "mul(" >> number << "," & number << ")"
    do = lit("do()") > (lambda _: True)
    dont = lit("don't()") > (lambda _: False)
    token = mul | do | dont
    tokens = rep1(until(token) >> token) << (reg(r".*") >> eof)


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, "args")
    args: Args = parser.parse_args().args

    with args.input.open() as input_file:
        input_text = input_file.read()
    muls: list[tuple[int, int]] = Q1Parser.muls.parse(input_text).unwrap()
    sum_muls = sum(i1 * i2 for i1, i2 in muls)
    print(f"Question 1 answer: {sum_muls}")
    tokens: list[tuple[int, int] | bool] = Q2Parser.tokens.parse(input_text).unwrap()

    enabled = True
    sum_muls = 0

    for tok in tokens:
        match tok:
            case (i1, i2):
                if enabled:
                    sum_muls += i1 * i2
            case bool_val:
                enabled = bool_val

    print(f"Question 2 answer: {sum_muls}")


if __name__ == "__main__":
    main()
