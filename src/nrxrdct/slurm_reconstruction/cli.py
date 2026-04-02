"""
nrxrdct.slurm_reconstruction.cli
----------------------------------
Unified CLI entry point registered as 'nrxrdct-slurm-recon'.

Sub-commands
------------
    nrxrdct-slurm-recon build  [options]  →  assemble sinogram from integrated HDF5
    nrxrdct-slurm-recon launch [options]  →  submit N GPU reconstruction jobs
"""

import argparse
import sys

from .launch_recon import _build_parser_build, _build_parser_launch
from .launch_recon import _cli_build, _cli_launch


def main():
    """
    Entry point for the ``nrxrdct-slurm-recon`` command registered in
    ``pyproject.toml``.

    Dispatches to one of two sub-commands:

    - ``build``  — assemble the sinogram from an integrated HDF5 file and
      save it to a standalone HDF5 dataset for the workers to read.
    - ``launch`` — split the 2θ axis into chunks and submit one GPU SLURM
      job per chunk.
    """
    p = argparse.ArgumentParser(
        prog="nrxrdct-slurm-recon",
        description="nrxrdct SLURM tomographic reconstruction pipeline",
    )
    sub = p.add_subparsers(dest="command", required=True)

    _build_parser_build(sub)
    _build_parser_launch(sub)

    args = p.parse_args()

    if args.command == "build":
        _cli_build(args)
    elif args.command == "launch":
        _cli_launch(args)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
