#!/usr/bin/env python3
"""Run a registered scientific case sequentially or with threads."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from khepri.validation import (
    SequentialRunner,
    ThreadRunner,
    case_registry,
    write_json_run,
    write_markdown_run,
)


def main(argv=None):
    registry = case_registry()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=sorted(registry))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("validation-output"))
    args = parser.parse_args(argv)

    case = registry[args.case]
    runner = SequentialRunner() if args.threads == 1 else ThreadRunner(args.threads)
    run = runner.run(case)
    raw = write_json_run(case, run, args.output / f"{case.slug}.json")
    report = write_markdown_run(
        case,
        run,
        args.output / f"{case.slug}.md",
        raw_json=raw,
    )
    print(report)


if __name__ == "__main__":
    main()
