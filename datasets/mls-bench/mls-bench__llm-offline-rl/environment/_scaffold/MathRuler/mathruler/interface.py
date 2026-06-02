import sys
from enum import Enum, unique

import fire

from .generate import vllm_generate
from .grader import evaluate


@unique
class Command(str, Enum):
    EVAL = "eval"
    GEN = "gen"


def main():
    command = sys.argv.pop(1) if len(sys.argv) != 1 else None
    if command == Command.EVAL:
        fire.Fire(evaluate)
    elif command == Command.GEN:
        fire.Fire(vllm_generate)
    else:
        print("Please use `mathruler eval` or `mathruler gen`.")


if __name__ == "__main__":
    main()
