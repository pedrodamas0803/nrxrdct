"""
nrxrdct.slurm_integration.cli
-------------------------------
Unified CLI entry point registered as 'nrxrdct-slurm'.

Sub-commands
------------
    nrxrdct-slurm launch  [options]   →  validate, init, submit N jobs
    nrxrdct-slurm check   [options]   →  verify output completeness
"""

import argparse
import sys

from .launch_jobs  import _build_parser as _launch_parser, _cli_launch
from .check_output import _build_parser as _check_parser,  _cli_check


def main():
    p = argparse.ArgumentParser(
        prog="nrxrdct-slurm",
        description="nrxrdct SLURM integration pipeline",
    )
    sub = p.add_subparsers(dest="command", required=True)

    _launch_parser(sub)
    _check_parser(sub)

    args = p.parse_args()

    if args.command == "launch":
        _cli_launch(args)
    elif args.command == "check":
        _cli_check(args)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()