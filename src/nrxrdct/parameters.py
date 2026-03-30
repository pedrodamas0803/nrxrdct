"""
Scan parameter container for XRD-CT acquisitions.

Defines the :class:`Scan` dataclass-style class that groups all motor names,
beam properties, and file paths needed to describe a single XRD-CT scan, with
helpers to persist and restore those parameters via HDF5.
"""
import os
from pathlib import Path

import numpy as np
import h5py


class Scan:
    """
    Class that holds data from each scan and their important parameters.
    Important units:
    - Energy: keV
    - Distance: mm
    """

    def __init__(
        self,
        acquisition_file: Path,
        sample_name: str,
        scan_type: str = "half-turn",
        translation_motor: str = "dty",
        rotation_motor: str = "rot",
        outer_loop_motor: str = "translation",
        beam_size: float = 100e-6,
        beam_energy: float = 44,
    ):
        """
        Parameters
        ----------
        acquisition_file : Path
            Path to the raw acquisition data file.
        sample_name : str
            Identifier for the sample being scanned.
        scan_type : str, optional
            Scan geometry, e.g. ``"half-turn"`` or ``"full-turn"`` (default ``"half-turn"``).
        translation_motor : str, optional
            Name of the fast (inner-loop) translation motor in the data file (default ``"dty"``).
        rotation_motor : str, optional
            Name of the rotation motor in the data file (default ``"rot"``).
        outer_loop_motor : str, optional
            Name of the slow (outer-loop) motor, typically a second translation
            axis (default ``"translation"``).
        beam_size : float, optional
            Beam size in metres (default 100 µm).
        beam_energy : float, optional
            Beam energy in keV (default 44 keV).  The wavelength in ångströms is
            derived automatically as ``12.398 / beam_energy``.
        """
        self.acquisition_file = acquisition_file
        self.sample_name = sample_name
        self.translation_motor = translation_motor
        self.rotation_motor = rotation_motor
        self.outer_loop_motor = outer_loop_motor
        self.beam_size = beam_size
        self.beam_energy = beam_energy
        self.scan_type = scan_type
        self.wavelength = 12.398 / self.beam_energy

    def __str__(self):
        msg = f"""
            XRDCT scan stored in {str(self.acquisition_file)}
            Translation motor name: {self.translation_motor}
            Rotation motor name: {self.rotation_motor}
            Scan outer loop: {self.outer_loop_motor}
            """
        return msg

    def save_parameter_file(self, output_file: Path = Path("xrdct_scan.h5")):
        """
        Persist all scan parameters to an HDF5 file.

        Each instance attribute is stored as a separate dataset.  String values
        are encoded automatically by h5py.

        Parameters
        ----------
        output_file : Path, optional
            Destination HDF5 file (default ``"xrdct_scan.h5"``).
        """
        with h5py.File(str(output_file), "a") as hout:
            for flag, value in self.__dict__.items():
                hout[flag] = value

    @classmethod
    def get_scan_from_parameters(cls, parameter_file: Path = Path("xrdct_scan.h5")):
        """
        Reconstruct a :class:`Scan` instance from a previously saved HDF5 parameter file.

        Parameters
        ----------
        parameter_file : Path, optional
            HDF5 file written by :meth:`save_parameter_file` (default ``"xrdct_scan.h5"``).

        Returns
        -------
        Scan
            A new :class:`Scan` instance populated from the stored parameters.
        """
        scan_dict = {}
        with h5py.File(str(parameter_file), "r") as hin:
            for key in hin.keys():
                value = hin[key][()]
                if not isinstance(value, (np.int64, np.float64)):
                    value = value.decode()
                scan_dict[key] = value

        return Scan(**scan_dict)
