from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Callable
from copy import deepcopy
import multiprocessing as mp
from itertools import repeat, chain, count
from functools import partial

import torch
from simple_parsing import ArgumentParser
from parsita import ParserContext, reg, rep1sep
from tqdm import tqdm


@dataclass
class Args:
    input: Path
    start_from: int = 0


A = 0
B = 1
C = 2


@dataclass
class Program:
    registers: tuple[int, int, int]
    instructions: list[int]


class ProgramParser(ParserContext):
    number = reg(r"[0-9]+") > int
    register = reg(r"Register .: ") >> number << "\n"
    instructions = "Program: " >> rep1sep(number, ",")
    program = register & register & register << "\n" & instructions << "\n" > (
        lambda l: Program(
            registers=tuple(l[:3]),
            instructions=l[3],
        )
    )


def interpret(program: Program) -> Iterator[tuple[int, int]]:
    registers = list(program.registers)

    instruction_pointer: int = 0

    while instruction_pointer < len(program.instructions):
        instruction = program.instructions[instruction_pointer]
        operand = program.instructions[instruction_pointer + 1]
        if operand <= 3:
            combo_operand: int = operand
        else:
            combo_operand = registers[operand - 4]

        match instruction:
            case 0:  # adv
                registers[A] = registers[A] // (2**combo_operand)
                instruction_pointer += 2
            case 1:  # bxl
                registers[B] = registers[B] ^ operand
                instruction_pointer += 2
            case 2:  # bsl
                registers[B] = combo_operand % 8
                instruction_pointer += 2
            case 3:  # jnz
                if registers[A] == 0:
                    instruction_pointer += 2
                else:
                    instruction_pointer = operand
            case 4:  # bxc
                registers[B] = registers[B] ^ registers[C]
                instruction_pointer += 2
            case 5:  # out
                yield combo_operand % 8, registers[A]
                instruction_pointer += 2
            case 6:  # bdv
                registers[B] = registers[A] // (2**combo_operand)
                instruction_pointer += 2
            case 7:  # cdv
                registers[C] = registers[A] // (2**combo_operand)
                instruction_pointer += 2
            case _:
                pass


def check_a_value(program: Program, a_value: int) -> bool:
    program.registers = (a_value, 0, 0)

    for out, target in zip(
        chain(interpret(program), repeat(None)),
        chain(program.instructions, repeat(None)),
    ):
        if out != target:
            return False
        if out is None and target is None:
            return True


def full_input_program(a: torch.IntTensor) -> tuple[torch.IntTensor, torch.IntTensor]:
    # Hardcoding the inner loop in pytorch to parallelize across many input A values
    b = a % 8
    b = b ^ 1
    c = a >> b
    b = b ^ 5
    b = b ^ c
    a = a >> 3
    out = b % 8
    return a, out


def test_input_program(a: torch.IntTensor) -> tuple[torch.IntTensor, torch.IntTensor]:
    a = a >> 3
    out = a % 8
    return a, out


def search_iteration(
    program: Callable[[torch.IntTensor], tuple[torch.IntTensor, torch.IntTensor]],
    device: torch.device,
    num_expected: int,
    cur_start_a: torch.IntTensor,
    cur_a_values: torch.IntTensor,
    cur_initial_a_values: torch.IntTensor,
    cur_check_index: torch.IntTensor,
    expected: torch.IntTensor,
) -> tuple[torch.IntTensor, torch.IntTensor]:
    new_a_values, out_values = program(cur_a_values)
    halted = new_a_values == 0
    match_expected = out_values == expected[cur_check_index]
    correct = halted & match_expected & (cur_check_index == num_expected - 1)
    correct_initial_values = cur_initial_a_values[correct]
    to_stop = halted | torch.logical_not(match_expected)
    num_to_stop = to_stop.sum(dtype=torch.int64)
    cur_check_index += 1
    new_init_a_values = torch.arange(
        cur_start_a, cur_start_a + num_to_stop, 1, dtype=torch.int64, device=device
    )
    new_a_values.masked_scatter_(to_stop, source=new_init_a_values)
    cur_initial_a_values.masked_scatter_(to_stop, source=new_init_a_values)
    cur_check_index.masked_scatter_(to_stop, source=torch.zeros_like(new_init_a_values))
    cur_start_a += num_to_stop

    return new_a_values, correct_initial_values


def search_gpu(
    device: torch.device,
    program: Callable[[torch.IntTensor], tuple[torch.IntTensor, torch.IntTensor]],
    expected: list[int],
    start_from: int,
) -> int:
    batch_size = 2**28

    cur_a_values = torch.arange(
        start_from, start_from + batch_size, dtype=torch.int64, device=device
    )
    cur_initial_a_values = cur_a_values.clone()
    cur_check_index = torch.zeros_like(cur_a_values)
    expected = torch.tensor(expected, dtype=torch.int64, device=device)
    num_expected = expected.shape[0]
    cur_start_a = torch.tensor(
        start_from + batch_size, dtype=torch.int64, device=device
    )

    _search_iteration = torch.compile(
        partial(
            search_iteration,
            program,
            device,
            num_expected,
        ),
    )

    with tqdm(
        bar_format="Number of checked A values: {postfix} | Elapsed: {elapsed} | {rate_fmt}",
        postfix=int(cur_start_a),
    ) as t:
        while True:
            t.postfix = int(cur_start_a)
            t.update()
            cur_a_values, correct_values = _search_iteration(
                cur_start_a,
                cur_a_values,
                cur_initial_a_values,
                cur_check_index,
                expected,
            )
            if correct_values.numel() > 0:
                return int(torch.min(correct_values))


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, "args")
    args: Args = parser.parse_args().args

    with args.input.open() as input_file:
        input_text = input_file.read()
    program: Program = ProgramParser.program.parse(input_text).unwrap()

    print(f"Question 1 answer: {','.join(str(out) for out, _ in interpret(program))}")
    # quine_value = search_quine(program)
    # print(f"Question 2 answer: {quine_value}")
    # print(f"{list(o for o, _ in interpret(Program(registers=(quine_value, 0, 0), instructions=program.instructions)))}")
    test_input_out = search_gpu(
        torch.device("cuda:0"), test_input_program, [0, 3, 5, 4, 3, 0], start_from=0
    )
    print(f"{test_input_out=}")
    full_input_out = search_gpu(
        torch.device("cuda:0"),
        full_input_program,
        [2, 4, 1, 1, 7, 5, 1, 5, 4, 0, 0, 3, 5, 5, 3, 0],
        start_from=args.start_from,
    )
    print(f"Question 2 answer: {full_input_out}")


if __name__ == "__main__":
    main()
