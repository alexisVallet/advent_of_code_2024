from pathlib import Path
from dataclasses import dataclass
from itertools import count, chain
from copy import deepcopy

from simple_parsing import ArgumentParser


@dataclass
class Args:
    input: Path


def q1_compact(input_text: str) -> list[int]:
    input: list[None | int] = list(
        chain(
            *[
                ([i // 2] if i % 2 == 0 else [None]) * int(c)
                for i, c in enumerate(input_text)
            ]
        )
    )
    compacted: list[int] = []

    for i in count():
        if i >= len(input):
            break
        in_block_id = input[i]
        match in_block_id:
            case None:
                out_block_id: None | int = None
                while out_block_id is None:
                    out_block_id = input.pop()
            case block_id:
                out_block_id = block_id
        compacted.append(out_block_id)

    return compacted


def q2_compact(input_text: str) -> list[int | None]:
    input: list[tuple[None | int, int]] = [
        ((i // 2) if i % 2 == 0 else None, int(c)) for i, c in enumerate(input_text)
    ]
    output = deepcopy(input)

    while True:
        src_block = input.pop()
        src_block_id, src_block_size = src_block
        match src_block_id:
            case None:
                continue
            case 0:
                break
            case _:
                # Remove it from the output and replace it by a free space block without
                # merging with free space blocks left/right so it's inserted at its
                # original position if no empty space on the left.
                for i in range(len(output)):
                    block_id, block_size = output[i]
                    if block_id == src_block_id:
                        output[i] = (None, block_size)
                for insertion_point in range(len(output)):
                    tgt_block_id, tgt_block_size = output[insertion_point]
                    if tgt_block_id is None and tgt_block_size >= src_block_size:
                        output[insertion_point] = (
                            None,
                            tgt_block_size - src_block_size,
                        )
                        output.insert(insertion_point, src_block)
                        break
    output_flat: list[int | None] = list(
        chain(*[[block_id] * block_size for block_id, block_size in output])
    )

    return output_flat


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, "args")
    args: Args = parser.parse_args().args

    with args.input.open() as input_file:
        input_text: str = input_file.read().strip()

    checksum_q1 = sum(i * int(c) for i, c in enumerate(q1_compact(input_text)))
    print(f"Question 1 answer: {checksum_q1}")
    q2_compacted = q2_compact(input_text)
    checksum_q2 = sum(i * int(c) for i, c in enumerate(q2_compacted) if c is not None)
    print(f"Question 2 answer: {checksum_q2}")


if __name__ == "__main__":
    main()
