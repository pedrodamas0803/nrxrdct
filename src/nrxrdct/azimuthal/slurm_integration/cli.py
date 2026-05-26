"""
nrxrdct.slurm_integration.cli
-------------------------------
Unified CLI entry point registered as 'nrxrdct-slurm'.

Sub-commands
------------
    nrxrdct-slurm launch   — validate, write launch_meta, submit N jobs
    nrxrdct-slurm monitor  — watch job progress
    nrxrdct-slurm merge    — assemble output HDF5 from tmp files
    nrxrdct-slurm check    — verify progress / completeness
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
        prog="nrxrdct-slurm",
        description="nrxrdct SLURM integration pipeline",
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

# """
# nrxrdct.slurm_integration.cli
# -------------------------------
# Unified CLI entry point registered as 'nrxrdct-slurm'.

# Sub-commands
# ------------
#     nrxrdct-slurm launch  [options]   →  validate, init, submit N jobs
#     nrxrdct-slurm check   [options]   →  verify output completeness
# """

# import argparse
# import sys

# from .check_output import _build_parser as _check_parser
# from .check_output import _cli_check
# from .launch_jobs import _build_parser as _launch_parser
# from .launch_jobs import _cli_launch
# from .monitor import _build_parser as _monitor_parser
# from .monitor import _cli_monitor


# def main():
#     """
#     Entry point for the ``nrxrdct-slurm`` command registered in ``pyproject.toml``.

#     Dispatches to one of three sub-commands:

#     - ``launch``  — validate entries, initialise the output file, and submit SLURM jobs.
#     - ``check``   — verify output completeness and optionally trigger repair jobs.
#     - ``monitor`` — poll job and scan progress until completion.
#     """
#     p = argparse.ArgumentParser(
#         prog="nrxrdct-slurm",
#         description="nrxrdct SLURM integration pipeline",
#     )
#     sub = p.add_subparsers(dest="command", required=True)

#     _launch_parser(sub)
#     _check_parser(sub)
#     _monitor_parser(sub)

#     args = p.parse_args()

#     if args.command == "launch":
#         _cli_launch(args)
#     elif args.command == "check":
#         _cli_check(args)
#     elif args.command == "monitor":
#         _cli_monitor(args)
#     else:
#         p.print_help()
#         sys.exit(1)


# if __name__ == "__main__":
#     main()
