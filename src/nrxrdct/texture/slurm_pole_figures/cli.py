"""
nrxrdct.texture.slurm_pole_figures.cli
------------------------------------------
Unified CLI entry point registered as 'nrxrdct-slurm-texture'.

Sub-commands
------------
    nrxrdct-slurm-texture launch   — validate, write launch_meta, submit N jobs
    nrxrdct-slurm-texture monitor  — watch job progress
    nrxrdct-slurm-texture merge    — assemble output HDF5 from tmp files
    nrxrdct-slurm-texture check    — verify progress / completeness
"""

import argparse
import sys

from .check_output import _build_parser as _check_parser
from .check_output import _cli_check
from .launch_jobs import _build_parser as _launch_parser
from .launch_jobs import _cli_launch
from .merge import _build_parser as _merge_parser
from .merge import _cli_merge
from .monitor import _build_parser as _monitor_parser
from .monitor import _cli_monitor


def main():
    p = argparse.ArgumentParser(
        prog="nrxrdct-slurm-texture",
        description="nrxrdct SLURM pole-figure extraction pipeline",
    )
    sub = p.add_subparsers(dest="command", required=True)

    _launch_parser(sub)
    _monitor_parser(sub)
    _merge_parser(sub)
    _check_parser(sub)

    args = p.parse_args()

    if args.command == "launch":
        _cli_launch(args)
    elif args.command == "monitor":
        _cli_monitor(args)
    elif args.command == "merge":
        _cli_merge(args)
    elif args.command == "check":
        _cli_check(args)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()