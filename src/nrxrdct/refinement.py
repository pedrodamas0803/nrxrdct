"""
GSAS-II Rietveld refinement wrappers for XRD-CT data.

Provides :class:`BaseRefinement`, a :class:`~nrxrdct.parameters.Scan` subclass
that drives sequential GSAS-II refinement steps (background, scale, zero shift,
peak broadening, cell, preferred orientation, crystallite size, microstrain,
extinction), and :class:`InstrumentCalibration`, a specialised subclass for
calibrant-based instrument parameter calibration with dedicated plotting.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from GSASII import GSASIIscriptable as G2sc
from matplotlib import gridspec

from .io import read_xy_file, write_starting_instrument_pars
from .parameters import Scan
from .refine_dict import *

COLORS = ["magenta", "darkgreen", "blue", "red"]


class BaseRefinement(Scan):
    """
    Base class for GSAS-II Rietveld refinement of a single powder pattern.

    Inherits scan geometry from :class:`~nrxrdct.parameters.Scan` and layers on
    top a GSAS-II project (``self.gpx``) with methods for each common refinement
    step.  Subclass or use directly for calibrant and sample refinements.
    """

    def __init__(
        self,
        acquisition_file,
        sample_name,
        scan_type="half-turn",
        translation_motor="dty",
        rotation_motor="rot",
        outer_loop_motor="translation",
        beam_size=0.0001,
        beam_energy=44,
        tth_lims: tuple = (None, None),
        xy_file: Path = Path("integrated_data.xy"),
        param_file: Path = Path("calibrated_instrument.instprm"),
        polarization: float = 0.99,
    ) -> None:
        """
        Args:
            acquisition_file (Path): Raw acquisition data file (passed to :class:`~nrxrdct.parameters.Scan`).
            sample_name (str): Sample identifier used in output file names.
            scan_type (str, optional): Scan geometry (default ``"half-turn"``).
            translation_motor (str, optional): Inner-loop translation motor name (default ``"dty"``).
            rotation_motor (str, optional): Rotation motor name (default ``"rot"``).
            outer_loop_motor (str, optional): Outer-loop motor name (default ``"translation"``).
            beam_size (float, optional): Beam size in metres (default 100 µm).
            beam_energy (float, optional): Beam energy in keV (default 44).
            tth_lims (tuple of (float or None, float or None), optional): ``(low, high)`` 2θ limits
                in degrees.  ``None`` means use the pattern extent (default ``(None, None)``).
            xy_file (Path, optional): Integrated pattern to fit (default ``"integrated_data.xy"``).
            param_file (Path, optional): Calibrated ``.instprm`` file (default ``"calibrated_instrument.instprm"``).
            polarization (float, optional): Beam polarization fraction used when writing the starting
                instrument parameter file (default 0.99).
        """
        super().__init__(
            acquisition_file,
            sample_name,
            scan_type,
            translation_motor,
            rotation_motor,
            outer_loop_motor,
            beam_size,
            beam_energy,
        )

        self.xy_file = xy_file
        self.param_file = param_file
        self.low_lim, self.high_lim = tth_lims
        self.tth, self.intensity = read_xy_file(str(self.xy_file))
        self.phases = []

        if self.low_lim == None:
            self.low_lim = self.tth.min()
        if self.high_lim == None:
            self.hih_lim = self.tth.max()

        self.param_file_init = write_starting_instrument_pars(
            polarization=polarization, wavelength=self.wavelength
        )

        print(60 * "=")
        print("=== If you need more information on the parameters, see the link:")
        print(
            "https://gsas-ii.readthedocs.io/en/latest/objvarorg.html#parameter-names-in-gsas-ii"
        )
        print(60 * "=")

    def load_model(self, gpx_file: Path) -> tuple:
        """
        Load an existing GSAS-II project from a ``.gpx`` file.

        Use this to resume a refinement that was previously saved, or to
        reload a project after restarting Python.  The first histogram and
        all phases found in the project are attached to ``self.hist`` and
        ``self.phases`` respectively so that all other methods work
        immediately after loading.

        Args:
            gpx_file (Path): Path to an existing ``.gpx`` file.

        Returns:
            tuple: ``(gpx, hist)`` — the :class:`G2Project` and the first histogram.
        """
        self.gpx = G2sc.G2Project(gpxfile=str(gpx_file))
        self.hist = self.gpx.histograms()[0]
        self.phases = self.gpx.phases()
        if self.phases:
            self.phase = self.phases[-1]
            self.calibrant_composition = self.phase.name
        print(f"Loaded project: {gpx_file}")
        print(f"  Histogram : {self.hist.name}")
        print(f"  Phases    : {[ph.name for ph in self.phases]}")
        return self.gpx, self.hist

    def backup_model(self, label: str | None = None) -> Path:
        """
        Save a timestamped copy of the current ``.gpx`` project to a
        ``bkp_model`` folder located next to the project file.

        The project is saved before copying so the backup reflects the latest
        state.  Each backup lives in its own subfolder named
        ``YYYYMMDD_HHMMSS`` (optionally suffixed with a user-supplied label),
        making it easy to identify and sort chronologically.

        Args:
            label (str or None, optional): Short descriptive tag appended to
                the subfolder name, e.g. ``"after_cell"`` produces a folder
                named ``"20240115_143022_after_cell"``.  ``None`` (default)
                uses the timestamp alone.

        Returns:
            Path: Path to the backup ``.gpx`` file that was written.
        """
        self.gpx.save()

        gpx_path = Path(self.gpx.filename)
        bkp_root = gpx_path.parent / "bkp_model"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{label}" if label else timestamp
        bkp_dir = bkp_root / folder_name
        bkp_dir.mkdir(parents=True, exist_ok=True)

        dest = bkp_dir / gpx_path.name
        shutil.copy2(gpx_path, dest)

        print(f"Backup saved to:\n  {dest}")
        return dest

    def restore_backup(self, backup: int | str) -> None:
        """
        Restore the ``.gpx`` project from a previously saved backup.

        Backups are looked up from the ``bkp_model`` folder next to the
        current project file.  The chosen backup is copied over the live
        project file and the project is reloaded so all subsequent method
        calls reflect the restored state.

        Args:
            backup (int or str): Identifies which backup to restore.

                * **int** — zero-based index into the list of backups sorted
                  chronologically (``0`` is the oldest, ``-1`` is the most
                  recent).  Call :meth:`list_backups` to see the index table.
                * **str** — exact subfolder name (e.g.
                  ``"20240115_143022_after_cell"``).

        Raises:
            FileNotFoundError: If ``bkp_model`` does not exist or is empty.
            IndexError: If an integer index is out of range.
            ValueError: If a string name does not match any backup.
        """
        gpx_path = Path(self.gpx.filename)
        bkp_root = gpx_path.parent / "bkp_model"

        if not bkp_root.exists():
            raise FileNotFoundError(f"No backup folder found at '{bkp_root}'")

        backups = sorted(bkp_root.iterdir())
        if not backups:
            raise FileNotFoundError(f"Backup folder '{bkp_root}' is empty")

        if isinstance(backup, int):
            try:
                bkp_dir = backups[backup]
            except IndexError:
                raise IndexError(
                    f"Backup index {backup} out of range " f"(0 – {len(backups) - 1})"
                )
        else:
            matches = [b for b in backups if b.name == backup]
            if not matches:
                raise ValueError(
                    f"No backup named '{backup}'. "
                    f"Available: {[b.name for b in backups]}"
                )
            bkp_dir = matches[0]

        src = bkp_dir / gpx_path.name
        if not src.exists():
            raise FileNotFoundError(f"Expected .gpx file not found in backup: '{src}'")

        shutil.copy2(src, gpx_path)
        print(f"Restored backup '{bkp_dir.name}' → '{gpx_path}'")
        self.load_model(gpx_path)

    def list_backups(self) -> list[str]:
        """
        Print and return the available backups in chronological order.

        Returns:
            list[str]: Subfolder names, oldest first.
        """
        gpx_path = Path(self.gpx.filename)
        bkp_root = gpx_path.parent / "bkp_model"

        if not bkp_root.exists() or not any(bkp_root.iterdir()):
            print("No backups found.")
            return []

        backups = sorted(bkp_root.iterdir())
        print(f"{'Index':>6}  Backup")
        print("-" * 40)
        for i, b in enumerate(backups):
            print(f"{i:>6}  {b.name}")
        return [b.name for b in backups]

    def create_model(self, gpx_file: Path = Path("model.gpx")) -> tuple:
        """
        Create a new GSAS-II project and add the powder histogram.

        Args:
            gpx_file (Path, optional): Output ``.gpx`` file path (default ``"model.gpx"``).

        Returns:
            tuple: ``(gpx, hist)`` — the :class:`G2Project` and the added histogram object.
        """
        self.gpx = G2sc.G2Project(newgpx=str(gpx_file))

        self.hist = self.gpx.add_powder_histogram(
            datafile=self.xy_file, iparams=self.param_file, phases="all"
        )
        self.hist["data"][0]["Limits"] = [self.low_lim, self.high_lim]
        self.gpx.save()
        return self.gpx, self.hist

    def add_phase(
        self,
        cif_file: Path = Path("cif_file"),
        phase_name: str = "LaB6",
        block_cell: bool = True,
    ) -> G2sc.G2Phase:
        """
        Add a crystallographic phase from a CIF file to the GSAS-II project.

        Args:
            cif_file (Path, optional): Path to the CIF file (default ``"cif_file"``).
            phase_name (str, optional): Name to assign the phase in GSAS-II (default ``"LaB6"``).
            block_cell (bool, optional): If ``True``, fixes all atom positions and the unit cell so only
                instrumental parameters are refined in subsequent steps (default ``True``).

        Returns:
            G2Phase: The newly added GSAS-II phase object.
        """
        self.calibrant_composition = phase_name
        self.phase = self.gpx.add_phase(
            str(cif_file), phasename=phase_name, histograms=[self.hist]
        )
        if block_cell:
            for atom in self.phase.atoms():
                atom.refinement_flags = ""
            self.phase.set_refinements({"Cell": False})
        self.phases.append(self.phase)
        self.gpx.save()
        return self.phase

    def set_limits(self, low: float, high: float) -> None:
        """
        Set the active 2θ refinement range.

        Only data points within ``[low, high]`` are included in the
        least-squares fit.  Can be called at any time to narrow or widen
        the range without rebuilding the project.

        Args:
            low (float): Lower 2θ limit in degrees.
            high (float): Upper 2θ limit in degrees.
        """
        self.hist.set_refinements({"Limits": [low, high]})
        self.low_lim = low
        self.high_lim = high
        self.gpx.save()
        print(f"2θ limits set to [{low:.4f}, {high:.4f}] °")

    def add_excluded_region(self, low: float, high: float) -> None:
        """
        Exclude a 2θ interval from the refinement.

        Data points in ``[low, high]`` are masked out of the residual
        calculation.  Multiple calls add independent excluded regions.
        Typical uses: ice rings, substrate peaks, beam-stop shadow,
        or detector artefacts.

        Args:
            low (float): Lower bound of the excluded region in degrees.
            high (float): Upper bound of the excluded region in degrees.
        """
        current = self.hist.Excluded()
        current.append([low, high])
        self.hist.Excluded(current)
        self.gpx.save()
        print(
            f"Excluded region added: [{low:.4f}, {high:.4f}] °"
            f"  (total excluded regions: {len(current)})"
        )

    def set_LeBail(
        self,
        phase: str | list[str] | None = None,
        enable: bool = True,
    ) -> None:
        """
        Activate or deactivate LeBail extraction for one or more phases.

        In LeBail mode the integrated intensities of all reflections are
        treated as free parameters and refined to best match the observed
        pattern, without any structural model.  This is useful for:

        * Checking the unit cell and space group before a full Rietveld
          refinement.
        * Extracting intensities for structure solution.
        * Fitting patterns where the structure of one phase is unknown
          while the others are refined by Rietveld.

        When LeBail is active for a phase, the HAP scale factor for that
        phase is refined instead of the structural scale, and atomic
        structure factors are not used.

        Args:
            phase (str, list of str, or None, optional): Phase name(s) to change.  ``None``
                (default) applies to all phases.
            enable (bool, optional): ``True`` (default) activates LeBail; ``False`` switches back
                to Rietveld mode.
        """
        available = {ph.name: ph for ph in self.gpx.phases()}
        if phase is None:
            targets = list(available.values())
        else:
            names = [phase] if isinstance(phase, str) else list(phase)
            for name in names:
                if name not in available:
                    raise ValueError(
                        f"Phase '{name}' not found. "
                        f"Available phases: {list(available)}"
                    )
            targets = [available[n] for n in names]

        for ph in targets:
            ph.set_refinements({"LeBail": enable})
            mode = "LeBail" if enable else "Rietveld"
            print(f"Phase '{ph.name}' set to {mode} mode")
        self.gpx.save()

    def refine_background(
        self,
        number_coeff: int = 12,
        do_refine: bool = True,
        function: str = "chebyschev",
        debye_terms: list | None = None,
        freeze: bool = False,
    ) -> None:
        """
        Refine the powder pattern background.

        Args:
            number_coeff (int, optional): Number of background function coefficients (default 12).
            do_refine (bool, optional): Whether to activate the background refinement flag (default ``True``).
            function (str, optional): Background function type.  Must be one of:

                * ``"chebyschev"``        — Chebyshev polynomial (default)
                * ``"chebyschev-1"``      — Chebyshev polynomial (1st kind)
                * ``"cosine"``            — Cosine series
                * ``"Q^2 power series"``  — Q² power series
                * ``"Q^-2 power series"`` — Q⁻² power series
                * ``"lin interpolate"``   — Linear interpolation
                * ``"inv interpolate"``   — Inverse interpolation
                * ``"log interpolate"``   — Logarithmic interpolation
            debye_terms (list of dict, optional): Debye–scattering components for amorphous content.
                Each entry is a dict with the following keys (all optional; unset keys fall back to
                defaults):

                * ``"A"``        — amplitude (default 1.0)
                * ``"R"``        — characteristic radius in Å (default 1.0)
                * ``"U"``        — damping / thermal factor (default 0.01)
                * ``"refine_A"`` — refine A (default ``True``)
                * ``"refine_R"`` — refine R (default ``True``)
                * ``"refine_U"`` — refine U (default ``True``)

                Example::

                    debye_terms=[
                        {"A": 1000.0, "R": 4.5, "U": 0.01,
                         "refine_A": True, "refine_R": True, "refine_U": False},
                    ]
        """
        valid_functions = {
            "chebyschev",
            "chebyschev-1",
            "cosine",
            "Q^2 power series",
            "Q^-2 power series",
            "lin interpolate",
            "inv interpolate",
            "log interpolate",
        }
        if function not in valid_functions:
            raise ValueError(
                f"Unknown background function '{function}'. "
                f"Valid options are: {sorted(valid_functions)}"
            )

        self.hist.set_refinements(
            {
                "Background": {
                    "type": function,
                    "no. coeffs": number_coeff,
                    "refine": do_refine,
                }
            }
        )

        if debye_terms:
            bkg_extra = self.hist["Background"][1]
            terms = []
            for term in debye_terms:
                A = term.get("A", 1.0)
                R = term.get("R", 1.0)
                U = term.get("U", 0.01)
                refA = term.get("refine_A", True)
                refR = term.get("refine_R", True)
                refU = term.get("refine_U", True)
                terms.append([A, refA, R, refR, U, refU])
            bkg_extra["nDebye"] = len(terms)
            bkg_extra["debyeTerms"] = terms

        self.gpx.save()
        self.gpx.do_refinements([{}])

        if freeze:
            bkg = self.hist["Background"]
            bkg[0][1] = False
            for term in bkg[1].get("debyeTerms", []):
                term[1] = term[3] = term[5] = False  # refA, refR, refU
            self.gpx.save()

        frozen_info = " (parameters frozen)" if freeze else ""
        print(
            f"Background refinement performed (function='{function}'"
            + (f", {len(debye_terms)} Debye term(s)" if debye_terms else "")
            + frozen_info
            + ")"
        )

    def refine_histogram_scale(self, freeze: bool = False) -> None:
        """Refine the overall histogram scale factor."""
        self.hist.SampleParameters["Scale"][1] = True
        self.gpx.save()
        self.gpx.do_refinements([{}])
        if freeze:
            self.hist.SampleParameters["Scale"][1] = False
            self.gpx.save()
        frozen_info = " (parameter frozen)" if freeze else ""
        print(f"Histogram scale refinement done{frozen_info}")

    def refine_phase_scale(self, phase: str | list[str], freeze: bool = False) -> None:
        """
        Refine the HAP scale factor for one or more phases.

        Args:
            phase (str or list of str): Phase name, or list of phase names, whose HAP scale factor
                should be refined.  Names must match those used in :meth:`add_phase`.
            freeze (bool, optional): If ``True``, freeze the scale flag after the refinement cycle
                (default ``False``).
        """
        names = [phase] if isinstance(phase, str) else list(phase)
        available = {ph.name: ph for ph in self.gpx.phases()}
        for name in names:
            if name not in available:
                raise ValueError(
                    f"Phase '{name}' not found. " f"Available phases: {list(available)}"
                )
            available[name].set_HAP_refinements({"Scale": True}, histograms=[self.hist])
            self.gpx.save()
            self.gpx.do_refinements([{}])
            if freeze:
                available[name].set_HAP_refinements(
                    {"Scale": False}, histograms=[self.hist]
                )
                self.gpx.save()
            frozen_info = " (parameter frozen)" if freeze else ""
            print(f"Phase scale refinement done for '{name}'{frozen_info}")

    def refine_zero_shift(self, freeze: bool = False) -> None:
        """Refine the 2θ zero-shift instrument parameter."""
        self.hist.set_refinements({"Instrument Parameters": ["Zero"]})
        self.gpx.save()
        self.gpx.do_refinements([{}])
        if freeze:
            self.hist["Instrument Parameters"][0]["Zero"][2] = False
            self.gpx.save()
        frozen_info = " (parameter frozen)" if freeze else ""
        print(f"Zero shift refinement done{frozen_info}")

    def refine_wavelength(self, freeze: bool = False) -> None:
        """
        Refine the incident-beam wavelength (``Lam``).

        The wavelength enters every d-spacing calculation via Bragg's law and
        is therefore strongly correlated with the unit-cell parameters.  It
        should only be refined when the nominal wavelength from the beamline
        monochromator calibration is uncertain, and only after the cell
        parameters are well converged.  Refining wavelength and cell
        simultaneously is generally not recommended.

        Args:
            freeze (bool, optional): If ``True``, clear the ``Lam`` refinement flag after the cycle
                so the wavelength stays fixed in subsequent steps (default ``False``).
        """
        self.hist.set_refinements({"Instrument Parameters": ["Lam"]})
        self.gpx.save()
        self.gpx.do_refinements([{}])
        if freeze:
            self.hist["Instrument Parameters"][0]["Lam"][2] = False
            self.gpx.save()
        lam_val = self.hist["Instrument Parameters"][0]["Lam"][1]
        energy_kev = 12.398 / lam_val
        frozen_info = " (parameter frozen)" if freeze else ""
        print(
            f"Wavelength refinement done: Lam = {lam_val:.8f} Å  ({energy_kev:.4f} keV){frozen_info}"
        )

    def print_instrument_parameters(self) -> None:
        """
        Print all instrument parameters currently present in the histogram.

        Lists every entry found in ``self.hist["Instrument Parameters"][0]``
        together with its current value and whether it is free to refine or
        frozen.  Use this to discover which parameter strings are valid
        arguments to :meth:`set_instrument_parameter`.
        """
        ip = self.hist["Instrument Parameters"][0]
        profile_type = ip.get("Type", ["?"])[0]
        print(f"Instrument parameters  (profile: {profile_type}):")
        print(f"  {'Parameter':<14}  {'Value':>14}  Status")
        print(f"  {'-'*14}  {'-'*14}  ------")
        for key, entry in ip.items():
            if key == "Type":
                continue
            if isinstance(entry, list) and len(entry) >= 2:
                val = entry[1]
                if not isinstance(val, (int, float)):
                    continue
                refine = entry[2] if len(entry) >= 3 else False
                flag = "refine" if refine else "fixed"
            elif isinstance(entry, (int, float)):
                val = entry
                flag = "fixed"
            else:
                continue
            print(f"  {key:<14}  {val:>14.6g}  ({flag})")

    def set_instrument_parameter(self, parameter: str, value: float) -> None:
        """
        Set an instrument parameter to a fixed value and freeze it.

        The parameter is set to *value* in both the initial-value slot
        (``entry[0]``) and the current-value slot (``entry[1]``) of the
        GSAS-II instrument-parameter list, and its refinement flag
        (``entry[2]``) is set to ``False`` so it will not move during
        subsequent refinement cycles.

        Args:
            parameter (str): Key in ``self.hist["Instrument Parameters"][0]`` to modify.
                Call :meth:`print_instrument_parameters` to see all available keys for the
                current project (e.g. ``"Zero"``, ``"Lam"``, ``"U"``, ``"V"``, ``"W"``).
            value (float): Value to assign to the parameter.

        Raises:
            KeyError: If *parameter* is not found in the instrument-parameter dictionary for
                this histogram.
            TypeError: If the entry for *parameter* is not a list (i.e. it is a read-only
                metadata field such as ``"Type"``).
        """
        ip = self.hist["Instrument Parameters"][0]
        if parameter not in ip:
            available = [
                k
                for k, v in ip.items()
                if isinstance(v, list)
                and len(v) >= 3
                and isinstance(v[1], (int, float))
            ]
            raise KeyError(
                f"Parameter '{parameter}' not found in instrument parameters. "
                f"Available parameters: {sorted(available)}"
            )
        entry = ip[parameter]
        if not isinstance(entry, list):
            raise TypeError(
                f"'{parameter}' is a read-only metadata field and cannot be set."
            )
        entry[0] = value  # initial / reset value
        entry[1] = value  # current refined value
        entry[2] = False  # freeze
        self.gpx.save()
        print(f"Instrument parameter '{parameter}' set to {value} and frozen.")

    def refine_sample_displacement(
        self, parameter: str = "Shift", freeze: bool = False
    ) -> None:
        """
        Refine a sample displacement parameter.

        Args:
            parameter (str, optional): Sample parameter to refine.  Choose according to the geometry:

                * ``"Shift"`` (default) — Bragg-Brentano geometry.  Displacement of
                  the sample surface along the diffractometer axis (perpendicular to
                  the sample surface, i.e. along the bisector of the incident and
                  diffracted beams).  Shifts peak positions as:

                      Δ(2θ) = −2 · (0.18 / π·R) · Shift · cos(θ)

                * ``"DisplaceX"`` — Debye-Scherrer (capillary) geometry.
                  Displacement **perpendicular to the incident beam** in the
                  horizontal plane (the plane containing the beam and the detector
                  arc).  Shifts peak positions as:

                      Δ(2θ) = −(0.18 / π·R) · DisplaceX · cos(2θ)

                * ``"DisplaceY"`` — Debye-Scherrer (capillary) geometry.
                  Displacement **parallel to the incident beam** (along the beam
                  propagation direction).  Shifts peak positions as:

                      Δ(2θ) = −(0.18 / π·R) · DisplaceY · sin(2θ)

                In all formulae R is the goniometer radius in mm and displacements
                are in μm.  ``DisplaceX`` and ``DisplaceY`` are mutually orthogonal
                components whose combined effect rotates with 2θ through the cos/sin
                weighting.
        """
        valid = {"Shift", "DisplaceX", "DisplaceY"}
        if parameter not in valid:
            raise ValueError(
                f"Unknown displacement parameter '{parameter}'. "
                f"Valid options are: {sorted(valid)}"
            )
        self.hist.set_refinements({"Sample Parameters": [parameter]})
        self.gpx.save()
        self.gpx.do_refinements([{}])
        if freeze:
            self.hist.clear_refinements({"Sample Parameters": [parameter]})
            self.gpx.save()
        frozen_info = " (parameter frozen)" if freeze else ""
        print(f"Sample displacement refinement done for '{parameter}'{frozen_info}")

    def set_absorption(
        self,
        absorption: float,
        refine: bool = False,
    ) -> None:
        """
        Set the GSAS-II ``Absorption`` sample parameter and optionally refine it.

        The absorption correction accounts for the attenuation of both the
        incident and diffracted beams as they travel through the sample.  Its
        effect is strongest at low 2θ angles (long path length through the
        sample) and weakens at higher angles.  Neglecting absorption when μr
        (or μt) is significant causes systematic under-estimation of
        low-angle reflection intensities relative to high-angle ones.

        Use :func:`~nrxrdct.utils.calculate_absorption_coefficient` to
        estimate the parameter value from the chemical composition, density,
        beam energy, and sample dimensions before calling this method.

        Args:
            absorption (float): GSAS-II ``Absorption`` parameter value.  The physical meaning
                depends on the sample geometry set in the GSAS-II project:

                * **Debye-Scherrer (capillary, transmission)** — dimensionless
                  product μr, where μ is the linear attenuation coefficient
                  (cm⁻¹) and r is the capillary radius (cm).  Typical values
                  for 0.3–0.7 mm capillaries at synchrotron energies
                  (30–100 keV) are in the range 0.001–0.5.

                * **Flat-plate (reflection or transmission)** — dimensionless
                  product μt, where t is the sample thickness (cm).

            refine (bool, optional): If ``True``, activate the refinement flag so the absorption
                parameter is included in the next least-squares cycle.
                Default ``False`` (parameter fixed at the supplied value).

                Note:
                    Absorption and the overall scale factor are strongly
                    correlated — both scale the pattern intensity.  Only free
                    absorption for refinement after the scale factor and
                    background are well converged, and only when the pattern
                    exhibits a clear low-angle intensity deficit consistent with
                    absorption.  For thin capillaries (μr < 0.1) the correction
                    is small enough to fix at the calculated value.
        """
        self.hist.SampleParameters["Absorption"][0] = absorption
        self.hist.SampleParameters["Absorption"][1] = refine
        self.gpx.save()
        status = "free for refinement" if refine else "fixed"
        print(f"Absorption set to {absorption:.6f} ({status})")

    def set_sample_parameter(
        self, parameter: str, value: float, freeze: bool = False
    ) -> None:
        """
        Set a sample parameter to a given value and optionally freeze it.

        The parameter value slot (``entry[0]``) is updated.  If
        ``freeze=True`` the refinement flag (``entry[1]``) is also cleared
        so the parameter will not move in subsequent refinement cycles.

        Args:
            parameter (str): Key in ``self.hist.SampleParameters`` to modify.  Common
                choices: ``"Scale"``, ``"Absorption"``, ``"Shift"``,
                ``"DisplaceX"``, ``"DisplaceY"``, ``"Transparency"``,
                ``"SurfRoughA"``, ``"SurfRoughB"``.
            value (float): Value to assign to the parameter.
            freeze (bool, optional): If ``True``, clear the refinement flag after setting the value
                so the parameter stays fixed in subsequent cycles (default ``False``).

        Raises:
            KeyError: If *parameter* is not found in the sample-parameter dictionary.
            TypeError: If the entry for *parameter* is not a list (read-only field).
        """
        sp = self.hist.SampleParameters
        if parameter not in sp:
            available = [k for k, v in sp.items() if isinstance(v, list)]
            raise KeyError(
                f"Sample parameter '{parameter}' not found. "
                f"Available parameters: {sorted(available)}"
            )
        entry = sp[parameter]
        if not isinstance(entry, list):
            raise TypeError(
                f"'{parameter}' is a read-only metadata field and cannot be set."
            )
        entry[0] = value
        if freeze:
            entry[1] = False
        self.gpx.save()
        frozen_info = " and frozen" if freeze else ""
        print(f"Sample parameter '{parameter}' set to {value}{frozen_info}.")

    def refine_gaussian_broadening(
        self, refine: list = ["U", "V", "W", "SH/L"], freeze: bool = False
    ) -> None:
        """
        Refine Gaussian peak-width parameters one at a time.

        Each parameter is refined in a separate cycle.  After the cycle its
        flag is cleared so that the value is not changed by later refinement
        steps unless explicitly freed again.

        Args:
            refine (list of str, optional): Ordered list of Gaussian instrument parameters to refine.
                Valid choices (all available in the FCJVoigt / ExpFCJVoigt profile):

                ``"U"``
                    Caglioti quadratic term: FWHM²_G += U·tan²θ.
                    Dominated by sample microstrain and wavelength dispersion.
                    Units: centideg².

                ``"V"``
                    Caglioti linear term: FWHM²_G += V·tanθ.
                    Usually small; cross-term between source and detector
                    contributions.  Units: centideg².

                ``"W"``
                    Caglioti constant term: FWHM²_G += W.
                    For a well-collimated synchrotron beam with a 2-D detector
                    this is often the *only* non-negligible Gaussian contribution.
                    Units: centideg².

                ``"Z"``
                    Size-broadening Gaussian term: FWHM²_G += Z/cos²θ.
                    Encodes crystallite-size broadening in the Gaussian channel.
                    Units: centideg².

                ``"SH/L"``
                    Finger–Cox–Jepcoat axial-divergence parameter (ratio of
                    sample-to-detector half-height over sample-to-detector
                    distance).  Affects low-angle peak asymmetry.  Dimensionless.
                    Not available in the EpsVoigt (PXB) profile.

                For a synchrotron experiment with a 2-D integrating detector a
                typical starting point is ``["W"]``.  Laboratory sources usually
                benefit from refining ``["U", "V", "W"]`` together (sequentially).

        Note:
            Parameters are refined *sequentially* (one per cycle).  To refine
            them simultaneously use :meth:`refine_peak_profile` and pass the
            desired list via the ``parameters`` argument.
        """
        ip = self.hist["Instrument Parameters"][0]
        for param in refine:
            self.hist.set_refinements({"Instrument Parameters": [param]})
            self.gpx.save()
            self.gpx.do_refinements([{}])
            if (
                freeze
                and param in ip
                and isinstance(ip[param], list)
                and len(ip[param]) >= 3
            ):
                ip[param][2] = False
            frozen_info = " (parameter frozen)" if freeze else ""
            print(f"Refined {param} (Gaussian broadening{frozen_info})")
        if freeze:
            self.gpx.save()

    def refine_lorentzian_broadening(
        self, refine: list = ["X", "Y"], freeze: bool = False
    ) -> None:
        """
        Refine Lorentzian peak-width parameters one at a time.

        Each parameter is refined in a separate cycle.  After the cycle its
        flag is cleared so that the value is not changed by later refinement
        steps unless explicitly freed again.

        Args:
            refine (list of str, optional): Ordered list of Lorentzian instrument parameters to refine.
                Valid choices:

                ``"X"``
                    Lorentzian size-broadening term: FWHM_L += X/cosθ.
                    Encodes crystallite-size-induced Lorentzian broadening.
                    Units: centideg.

                ``"Y"``
                    Lorentzian strain-broadening term: FWHM_L += Y·tanθ.
                    Encodes microstrain-induced Lorentzian broadening.
                    Units: centideg.

                The total Lorentzian FWHM combines with the Gaussian via the
                Thompson–Cox–Hastings pseudo-Voigt mixing rule:

                    FWHM_total⁵ = FWHM_G⁵ + FWHM_L⁵  (approximate)

        Note:
            Parameters are refined *sequentially* (one per cycle).  To refine
            them simultaneously use :meth:`refine_peak_profile` and pass the
            desired list via the ``parameters`` argument.
        """
        ip = self.hist["Instrument Parameters"][0]
        for param in refine:
            self.hist.set_refinements({"Instrument Parameters": [param]})
            self.gpx.save()
            self.gpx.do_refinements([{}])
            if (
                freeze
                and param in ip
                and isinstance(ip[param], list)
                and len(ip[param]) >= 3
            ):
                ip[param][2] = False
            frozen_info = " (parameter frozen)" if freeze else ""
            print(f"Refined {param} (Lorentzian broadening{frozen_info})")
        if freeze:
            self.gpx.save()

    def refine_peak_profile(
        self,
        profile: str = "FCJVoigt",
        parameters: list[str] | None = None,
    ) -> None:
        """
        Refine instrument peak-profile parameters for the chosen profile model.

        **Profile models**

        GSAS-II encodes the active profile in the third character of the
        ``Type`` instrument-parameter string (e.g. ``'PXC'`` → FCJVoigt).
        Switching ``profile`` here updates that character automatically and
        initialises any parameters that the new model needs but the project
        does not yet contain.

        **FCJVoigt** (``'PXC'``, default)
            Finger-Cox-Jepcoat convolution of a pseudo-Voigt with an axial-
            divergence correction.  The standard choice for laboratory and
            synchrotron CW data.  Peak width is parametrised via the
            Thompson-Cox-Hastings (TCH) approach:

            - Gaussian FWHM²  = U·tan²θ + V·tanθ + W + Z/cos²θ
            - Lorentzian FWHM = X/cosθ  + Y·tanθ

            Parameters:

            ``U`` — Gaussian width, quadratic-in-tanθ term.  Dominated by
            sample microstrain and wavelength dispersion.  Units: centideg².

            ``V`` — Gaussian width, linear-in-tanθ term.  Usually small;
            cross-term between source and detector contributions.
            Units: centideg².

            ``W`` — Gaussian width, angle-independent (constant) term.
            For a well-collimated synchrotron beam with a 2-D detector this
            is often the *only* non-negligible Gaussian contribution.
            Units: centideg².

            ``X`` — Lorentzian width, 1/cosθ term.  Proportional to
            crystallite-size broadening in the Scherrer equation; also
            picks up instrumental contributions along the beam direction.
            Units: centideg.

            ``Y`` — Lorentzian width, tanθ term.  Proportional to
            microstrain broadening (Williamson-Hall).  Units: centideg.

            ``Z`` — Gaussian width added in quadrature, independent of
            angle.  Rarely refined; used to model detector point-spread.
            Units: centideg².

            ``SH/L`` — Axial divergence ratio (S+H)/L, where S is the
            receiving-slit height, H the sample height, and L the
            sample-to-detector distance.  Introduces the characteristic
            low-angle asymmetry (low-2θ tail) in CW patterns.  Unitless.

        **ExpFCJVoigt** (``'PXA'``)
            FCJVoigt convolved with exponential rise and decay functions.
            Designed for *pink-beam* (moderately polychromatic) sources
            where the bandpass introduces asymmetric tails on every
            reflection.

            Shares U, V, W, X, Y, Z, SH/L with FCJVoigt (same meaning),
            plus:

            ``alpha-0`` — constant term of the rise (low-2θ tail)
            exponential coefficient:  α = alpha-0 + alpha-1·sinθ.
            Controls how steeply the leading edge of each peak rises.
            Unitless.

            ``alpha-1`` — sinθ-dependent term of the rise coefficient.
            Captures the angular dependence of the bandpass asymmetry on
            the low-angle side.  Unitless.

            ``beta-0`` — constant term of the decay (high-2θ tail)
            exponential coefficient:  β = beta-0 + beta-1·sinθ.
            Controls the trailing edge of each peak.  Unitless.

            ``beta-1`` — sinθ-dependent term of the decay coefficient.
            Unitless.

        **EpsVoigt** (``'PXB'``)
            Double-exponential pseudo-Voigt *without* axial-divergence
            correction.  Use when axial divergence is negligible (well-
            collimated beam, no SH/L effect needed) but exponential tails
            are still required (e.g. some pink-beam setups).

            Shares U, V, W, X, Y, Z with FCJVoigt and alpha-0, alpha-1,
            beta-0, beta-1 with ExpFCJVoigt.  ``SH/L`` is **not**
            available for this model.

        Args:
            profile (``"FCJVoigt"`` | ``"ExpFCJVoigt"`` | ``"EpsVoigt"``, optional):
                Peak profile model to activate and refine (default ``"FCJVoigt"``).
            parameters (list of str, optional): Instrument parameters to refine, refined sequentially.
                Must be valid for the chosen ``profile`` — see table above.
                If ``None`` (default), a sensible starting set is used:

                * FCJVoigt   → ``["W", "X", "Y"]``
                * ExpFCJVoigt → ``["W", "alpha-0", "alpha-1", "beta-0", "beta-1"]``
                * EpsVoigt   → ``["W", "alpha-0", "alpha-1", "beta-0", "beta-1"]``
        """
        _TYPE_CHAR = {"FCJVoigt": "C", "ExpFCJVoigt": "A", "EpsVoigt": "B"}
        _VALID_PARAMS = {
            "FCJVoigt": {"U", "V", "W", "X", "Y", "Z", "SH/L"},
            "ExpFCJVoigt": {
                "U",
                "V",
                "W",
                "X",
                "Y",
                "Z",
                "SH/L",
                "alpha-0",
                "alpha-1",
                "beta-0",
                "beta-1",
            },
            "EpsVoigt": {
                "U",
                "V",
                "W",
                "X",
                "Y",
                "Z",
                "alpha-0",
                "alpha-1",
                "beta-0",
                "beta-1",
            },
        }
        _DEFAULT_PARAMS = {
            "FCJVoigt": ["W", "X", "Y"],
            "ExpFCJVoigt": ["W", "alpha-0", "alpha-1", "beta-0", "beta-1"],
            "EpsVoigt": ["W", "alpha-0", "alpha-1", "beta-0", "beta-1"],
        }
        _EXTRA_DEFAULTS = {
            "alpha-0": [0.1, 0.1, False],
            "alpha-1": [0.0, 0.0, False],
            "beta-0": [0.1, 0.1, False],
            "beta-1": [0.0, 0.0, False],
        }

        if profile not in _TYPE_CHAR:
            raise ValueError(
                f"Unknown profile '{profile}'. " f"Valid options: {list(_TYPE_CHAR)}"
            )

        if parameters is None:
            parameters = _DEFAULT_PARAMS[profile]

        invalid = set(parameters) - _VALID_PARAMS[profile]
        if invalid:
            raise ValueError(
                f"Parameter(s) {sorted(invalid)} are not valid for profile "
                f"'{profile}'. Valid parameters: {sorted(_VALID_PARAMS[profile])}"
            )

        # Update profile type in the instrument parameters
        ip = self.hist["Instrument Parameters"][0]
        current = ip["Type"][0]  # e.g. 'PXC'
        new_type = current[:2] + _TYPE_CHAR[profile]
        if current != new_type:
            ip["Type"][0] = new_type
            ip["Type"][1] = new_type
            for param, default in _EXTRA_DEFAULTS.items():
                if param in _VALID_PARAMS[profile] and param not in ip:
                    ip[param] = default[:]

        for param in parameters:
            self.hist.set_refinements({"Instrument Parameters": [param]})
            self.gpx.save()
            self.gpx.do_refinements([{}])
            print(f"Refined {param} ({profile} profile)")

    def free_and_refine_cell(
        self,
        phase: str | list[str] | None = None,
        freeze_after: bool = False,
    ) -> None:
        """
        Frees and performs cell refinement for one or more phases.

        Tends to converge fast for higher symmetries provided that the
        spacegroup is correctly given.

        Args:
            phase (str, list of str, or None, optional): Phase name(s) to
                refine the unit cell for.  ``None`` (default) applies to all
                phases in the project.  Names must match those passed to
                :meth:`add_phase`.
            freeze_after (bool, optional): If ``True``, fix the cell parameters
                again after the refinement cycle so they are not varied in
                subsequent steps (default ``False``).
        """
        available = {ph.name: ph for ph in self.gpx.phases()}
        if phase is None:
            targets = list(available.values())
        else:
            names = [phase] if isinstance(phase, str) else list(phase)
            for name in names:
                if name not in available:
                    raise ValueError(
                        f"Phase '{name}' not found. "
                        f"Available phases: {list(available)}"
                    )
            targets = [available[n] for n in names]

        for ph in targets:
            ph.set_refinements({"Cell": True})
        self.gpx.save()
        self.gpx.do_refinements([{}])
        for ph in targets:
            print(f"Cell refined for phase '{ph.name}'")

        if freeze_after:
            for ph in targets:
                ph.set_refinements({"Cell": False})
            self.gpx.save()
            print("Cell parameters frozen after refinement.")

    def freeze_cell(self, phase: str | list[str] | None = None) -> None:
        """
        Fixes the unit-cell parameters for one or more phases so they are not
        varied in subsequent refinement cycles.

        Args:
            phase (str, list of str, or None, optional): Phase name(s) to
                freeze.  ``None`` (default) applies to all phases in the
                project.  Names must match those passed to :meth:`add_phase`.
        """
        available = {ph.name: ph for ph in self.gpx.phases()}
        if phase is None:
            targets = list(available.values())
        else:
            names = [phase] if isinstance(phase, str) else list(phase)
            for name in names:
                if name not in available:
                    raise ValueError(
                        f"Phase '{name}' not found. "
                        f"Available phases: {list(available)}"
                    )
            targets = [available[n] for n in names]

        for ph in targets:
            ph.set_refinements({"Cell": False})
            print(f"Cell parameters frozen for phase '{ph.name}'")
        self.gpx.save()

    def refine_preferential_orientation(
        self,
        model: str = "MD",
        phase: str | list[str] | None = None,
        parsMD: dict = MD_DICT,
        parsSH: dict = SH_DICT,
    ) -> None:
        """
        Refine preferred orientation (texture) for one or more phases.

        Args:
            model (``"MD"`` | ``"SH"``, optional): Texture model (default ``"MD"``).
                See model descriptions below.
            phase (str, list of str, or None, optional): Phase name(s) to apply the refinement to.
                ``None`` (default) applies to all phases in the project.
            parsMD (dict, optional): Full parameter dictionary for the March-Dollase model.
                Default is ``MD_DICT`` from ``refine_dict``.  You must supply the complete dict
                when overriding any value — partial updates are not merged automatically.
            parsSH (dict, optional): Full parameter dictionary for the spherical-harmonics model.
                Default is ``SH_DICT`` from ``refine_dict``.  Same caveat as ``parsMD``.

        **Models:**

        **March-Dollase** (``"MD"``)
            Single-parameter phenomenological model.  It assumes a
            cylindrically symmetric texture (fibre or rolling texture) about
            one preferred direction and models the orientation distribution
            function as a March-Dollase ellipsoid.

            - Simple, fast to converge, robust for weak-to-moderate texture.
            - Poor choice when the texture is strong or genuinely multi-axial.

            ``parsMD`` keys (wrap inside ``{"Pref.Ori.": {...}}``):

            ``"Model"`` — must be ``"MD"``.

            ``"Axis"`` — preferred-orientation axis in *direct-space* Miller
            indices ``[h, k, l]``.  Physically this is the crystallographic
            direction that aligns preferentially with the sample's fibre or
            compression axis.  Example: ``[0, 0, 1]`` for basal-plane texture
            in a hexagonal phase.

            ``"Ratio"`` — March-Dollase parameter *r* (dimensionless).
            ``r = 1`` means no texture; ``r < 1`` means crystallites with
            ``Axis`` perpendicular to the beam are depleted (needle texture);
            ``r > 1`` means they are enriched (plate texture).

            ``"Ref"`` — boolean, whether to refine ``Ratio``.

            Default::

                MD_DICT = {
                    "Pref.Ori.": {
                        "Model": "MD",
                        "Axis": [1, 1, 1],
                        "Ratio": 1.0,
                        "Ref": True,
                    }
                }

        **Spherical harmonics** (``"SH"``)
            Expands the orientation distribution function (ODF) in a
            symmetry-adapted spherical-harmonic series.  More flexible than
            March-Dollase: captures complex multi-component textures but
            introduces many parameters that require good counting statistics.

            ``parsSH`` keys (wrap inside ``{"Pref.Ori.": {...}}``):

            ``"Model"`` — must be ``"SH"``.

            ``"SHord"`` — maximum spherical-harmonic order *L* (even integer,
            typically 2–16).  Higher orders capture sharper texture features
            but add more parameters.  Number of terms grows as
            (L/2 + 1)² for cylindrical symmetry.

            ``"SHsym"`` — assumed *sample* symmetry.  Choose the highest
            symmetry consistent with the sample fabrication history:

            * ``"cylindrical"``  — rotation symmetry about one axis
              (fibre texture, wire drawing, uniaxial pressing).  Fewest
              parameters.
            * ``"orthorhombic"`` — three mutually perpendicular symmetry
              axes (cold-rolled sheet, extruded bar).
            * ``"monoclinic"``   — one symmetry axis only.
            * ``"triclinic"``    — no sample symmetry assumed.  Maximum
              flexibility, most parameters.  Use only with high-quality,
              well-statistics data.

            ``"Axis"`` — fibre/symmetry axis in *direct-space* Miller
            indices ``[h, k, l]``.  For cylindrical symmetry this is the
            unique axis of the ODF.

            ``"SHcoef"`` — dict of spherical-harmonic coefficients, keyed
            by GSAS-II internal labels.  Leave as ``{}``; GSAS-II populates
            and refines these automatically based on ``SHord`` and ``SHsym``.

            ``"Ref"`` — boolean, whether to refine the SH coefficients.

            Default::

                SH_DICT = {
                    "Pref.Ori.": {
                        "Model": "SH",
                        "SHord": 4,
                        "SHsym": "cylindrical",
                        "Axis": [0, 0, 1],
                        "SHcoef": {},
                        "Ref": True,
                    }
                }
        """
        if model not in ("MD", "SH"):
            raise ValueError(
                f"Unknown model '{model}'. Valid options are 'MD' and 'SH'."
            )

        available = {ph.name: ph for ph in self.gpx.phases()}
        if phase is None:
            targets = list(available.values())
        else:
            names = [phase] if isinstance(phase, str) else list(phase)
            for name in names:
                if name not in available:
                    raise ValueError(
                        f"Phase '{name}' not found. "
                        f"Available phases: {list(available)}"
                    )
            targets = [available[n] for n in names]

        pars = parsMD if model == "MD" else parsSH
        for ph in targets:
            ph.set_HAP_refinements(pars, histograms=[self.hist])
            self.gpx.save()
            self.gpx.do_refinements([{}])
            print(f"Preferred orientation ({model}) refined for phase '{ph.name}'")

    def refine_atomic_positions(
        self,
        flags: list[str] = ["U", "XU"],
        phase: str | list[str] | None = None,
        atoms: list[str] | None = None,
    ) -> None:
        """
        Refine atomic parameters for selected phases and atoms.

        Each flag in ``flags`` is applied to the target atoms and a full
        refinement cycle is run before moving to the next flag.  This staged
        approach lets less-correlated parameters (e.g. U\ :sub:`iso`)
        converge before adding more (e.g. fractional coordinates).

        Args:
            flags (list of str, optional): Sequence of GSAS-II atom-refinement flag strings applied
                in order.  Each string is any combination of the tokens below;
                the valid characters are ``X``, ``U``, ``F``, and space.

                ``""`` or ``" "``
                    All atomic parameters fixed.  Safe starting point before
                    any structural refinement begins.

                ``"U"``
                    Isotropic displacement parameter U\ :sub:`iso` only.
                    Good first step once scale and background are stable:
                    U\ :sub:`iso` is weakly correlated with most structural
                    parameters and converges quickly.

                ``"X"``
                    Fractional coordinates (x, y, z) only.  Use when the
                    displacement model is already well determined and you want
                    to let the atoms relax to their equilibrium positions.

                ``"XU"``
                    Fractional coordinates + U\ :sub:`iso`.  Standard choice
                    for a well-conditioned Rietveld refinement.  Default second
                    stage after ``"U"``.

                ``"F"``
                    Site occupancy only.  Isolate occupancy refinement when
                    the structure is well determined but partial occupancy is
                    suspected (defects, solid solutions).

                ``"XF"``
                    Fractional coordinates + site occupancy.  Use when
                    U\ :sub:`iso` is already stable and occupancy needs to
                    be freed simultaneously with positional parameters.

                ``"XUF"``
                    Fractional coordinates + U\ :sub:`iso` + occupancy.  Most
                    complete isotropic refinement.  Occupancy and U\ :sub:`iso`
                    are often correlated — converge them separately first if
                    the refinement is unstable.

                Note:
                    Anisotropic displacement parameters (ADP / U\ :sub:`ij`
                    tensor) are **not** controlled through this flag string.
                    They require changing the atom's displacement model from
                    isotropic (``'I'``) to anisotropic (``'A'``) via the
                    ``cia`` field directly in the GSAS-II project, and are
                    only justified for single-crystal-quality powder data.

            phase (str, list of str, or None, optional): Phase name(s) to apply the refinement to.
                ``None`` (default) applies to all phases in the project.  Names must match those
                passed to :meth:`add_phase`.

            atoms (list of str, or None, optional): Restrict refinement to a subset of atoms.
                Each entry is matched against atom **labels** (e.g. ``"Fe1"``, ``"O2"``) first,
                then against **element symbols** (e.g. ``"Fe"``, ``"O"``).  If an entry matches
                neither in a given phase it is silently skipped for that phase.  ``None`` (default)
                refines all atoms.
        """
        available = {ph.name: ph for ph in self.gpx.phases()}
        if phase is None:
            targets = list(available.values())
        else:
            names = [phase] if isinstance(phase, str) else list(phase)
            for name in names:
                if name not in available:
                    raise ValueError(
                        f"Phase '{name}' not found. "
                        f"Available phases: {list(available)}"
                    )
            targets = [available[n] for n in names]

        for flag in flags:
            for ph in targets:
                for atom in ph.atoms():
                    if atoms is None or atom.label in atoms or atom.element in atoms:
                        atom.refinement_flags = flag
            self.gpx.save()
            self.gpx.do_refinements([{}])
            phase_names = [ph.name for ph in targets]
            atom_info = f", atoms={atoms}" if atoms is not None else ""
            print(
                f"Atomic refinement flag '{flag}' applied to {phase_names}{atom_info}"
            )

    def refine_occupancy(
        self,
        phase: str | list[str] | None = None,
        atoms: list[str] | None = None,
        freeze: bool = False,
    ) -> None:
        """
        Refine site occupancy for selected atoms.

        The ``"F"`` refinement flag is added to the current flags of every
        target atom before the refinement cycle and removed afterwards, so
        that positional and displacement flags that were already active are
        left untouched.

        Args:
            phase (str, list of str, or None, optional): Phase name(s) to apply the refinement to.
                ``None`` (default) applies to all phases in the project.  Names must match those
                used in :meth:`add_phase`.
            atoms (list of str, or None, optional): Restrict refinement to a subset of atoms.  Each
                entry is matched against atom **labels** (e.g. ``"Fe1"``) first, then against
                **element symbols** (e.g. ``"Fe"``).  ``None`` (default) refines the occupancy of
                all atoms in the target phases.

        Note:
            Occupancy and U\ :sub:`iso` are often correlated, particularly for
            light atoms or when the site is close to fully occupied.  Converge
            the displacement parameters first (via :meth:`refine_atomic_positions`
            with flag ``"U"`` or ``"XU"``) before freeing occupancy.

            Occupancy refinement is only physically meaningful when partial
            occupancy is expected — for example in solid solutions, vacancy-
            disordered structures, or mixed-occupancy Wyckoff sites.  Refining
            occupancy on a fully occupied structure will correlate strongly with
            the scale factor and may cause divergence.
        """
        available = {ph.name: ph for ph in self.gpx.phases()}
        if phase is None:
            targets = list(available.values())
        else:
            names = [phase] if isinstance(phase, str) else list(phase)
            for name in names:
                if name not in available:
                    raise ValueError(
                        f"Phase '{name}' not found. "
                        f"Available phases: {list(available)}"
                    )
            targets = [available[n] for n in names]

        # Collect target atoms and store original flags
        target_atoms = []
        for ph in targets:
            for atom in ph.atoms():
                if atoms is None or atom.label in atoms or atom.element in atoms:
                    target_atoms.append(atom)

        original_flags = {id(a): a.refinement_flags for a in target_atoms}
        for atom in target_atoms:
            if "F" not in atom.refinement_flags:
                atom.refinement_flags = atom.refinement_flags + "F"

        self.gpx.save()
        self.gpx.do_refinements([{}])

        if freeze:
            for atom in target_atoms:
                atom.refinement_flags = original_flags[id(atom)]
            self.gpx.save()

        phase_names = [ph.name for ph in targets]
        atom_info = f", atoms={atoms}" if atoms is not None else ""
        frozen_info = " (flag frozen)" if freeze else ""
        print(f"Occupancy refinement done for {phase_names}{atom_info}{frozen_info}")

    def refine_Uiso(
        self,
        phase: str | list[str] | None = None,
        atoms: list[str] | None = None,
        freeze: bool = False,
    ) -> None:
        """
        Refine the isotropic atomic displacement parameter U\ :sub:`iso` for
        selected atoms, then optionally freeze the flag.

        U\ :sub:`iso` (also written B\ :sub:`iso` / 8π²U\ :sub:`iso` in older
        notation) describes the mean-square displacement of an atom from its
        equilibrium site, averaged over all directions.  It combines genuine
        thermal vibrations with static disorder (positional spread across the
        unit cells of the crystal).  Physically, U\ :sub:`iso` enters the
        structure factor as a Debye-Waller factor:

            T(sinθ/λ) = exp(−8π² U\ :sub:`iso` sin²θ / λ²)

        which attenuates the calculated intensity increasingly at high 2θ
        angles.  Refining U\ :sub:`iso` therefore corrects the high-angle
        intensity fall-off and is one of the earliest structural parameters
        to stabilise in a Rietveld refinement sequence.

        Only atoms whose ``adp_flag`` is ``'I'`` (isotropic displacement model)
        are modified.  Atoms already set to anisotropic (``'A'``) are skipped
        silently — use the GSAS-II GUI or direct data manipulation to refine
        anisotropic ADPs.

        The ``'U'`` refinement flag is added to each target atom's existing
        flag string before the cycle and, if ``freeze=True``, removed
        afterwards, leaving any ``'X'`` or ``'F'`` flags that were already
        active untouched.

        Args:
            phase (str, list of str, or None, optional): Phase name(s) to apply the refinement to.
                ``None`` (default) applies to all phases in the project.  Names must match those
                used in :meth:`add_phase`.
            atoms (list of str, or None, optional): Restrict refinement to a subset of atoms matched
                against atom **labels** (e.g. ``"Fe1"``) first, then **element symbols**
                (e.g. ``"Fe"``).  ``None`` (default) refines all isotropic atoms in the target phases.
            freeze (bool, optional): If ``True`` (default), remove the ``'U'`` flag after the
                refinement cycle, restoring the original flag state.  Set to ``False`` to keep
                U\ :sub:`iso` free for subsequent cycles (e.g. when continuing with
                :meth:`refine_atomic_positions`).

        Note:
        **Typical refinement sequence:**

        1. :meth:`refine_background` — fit the baseline first.
        2. :meth:`refine_histogram_scale` — establish the overall intensity scale.
        3. :meth:`refine_Uiso` (this method, ``freeze=False``) — let
           displacement parameters absorb residual scale errors before
           freeing atomic positions.
        4. :meth:`refine_atomic_positions` with flag ``"XU"`` — refine
           positions and U\ :sub:`iso` simultaneously once the model is stable.

        **Correlations to watch:**

        * U\ :sub:`iso` is positively correlated with the overall scale factor
          (both scale the calculated pattern) and with site occupancy.  Ensure
          the scale is well determined before freeing U\ :sub:`iso`.
        * For light atoms (Z ≲ 10) the Debye-Waller attenuation is weak and
          U\ :sub:`iso` may refine to unphysical values; consider fixing it.
        * Very large U\ :sub:`iso` (> 0.05 Å²) often indicates a wrong atom
          type, split site, or structural disorder rather than genuine thermal
          motion.
        * Very small or negative U\ :sub:`iso` usually means the calculated
          intensities are too low at high angles — check for preferred
          orientation, absorption, or an incorrect structure model.

        **Typical values:**

        At room temperature, U\ :sub:`iso` for most inorganic phases falls in
        the range 0.003–0.020 Å².  Lighter atoms and softer bonding
        environments tend toward the upper end.
        """
        available = {ph.name: ph for ph in self.gpx.phases()}
        if phase is None:
            targets = list(available.values())
        else:
            names = [phase] if isinstance(phase, str) else list(phase)
            for name in names:
                if name not in available:
                    raise ValueError(
                        f"Phase '{name}' not found. "
                        f"Available phases: {list(available)}"
                    )
            targets = [available[n] for n in names]

        target_atoms = []
        for ph in targets:
            for atom in ph.atoms():
                if atom.adp_flag != "I":
                    continue
                if atoms is None or atom.label in atoms or atom.element in atoms:
                    target_atoms.append(atom)

        original_flags = {id(a): a.refinement_flags for a in target_atoms}
        for atom in target_atoms:
            if "U" not in atom.refinement_flags:
                atom.refinement_flags = atom.refinement_flags + "U"

        self.gpx.save()
        self.gpx.do_refinements([{}])

        if freeze:
            for atom in target_atoms:
                atom.refinement_flags = original_flags[id(atom)]
            self.gpx.save()

        phase_names = [ph.name for ph in targets]
        atom_info = f", atoms={atoms}" if atoms is not None else ""
        frozen_info = " (flag frozen)" if freeze else ""
        print(f"Uiso refinement done for {phase_names}{atom_info}{frozen_info}")

    def refine_phase_content(self) -> None:
        """
        Sets the scale factor of the histogram to zero and activate the refinement per phase to extract phase fractions.

        Excludes phases set to LeBail refinement.
        """
        self.hist.SampleParameters["Scale"][1] = False
        self.hist.SampleParameters["Scale"][0] = 1.0
        self.gpx.save()
        for ph in self.gpx.phases():
            hap = ph.data["Histograms"].get(self.hist.name, {})
            if hap.get("LeBail", False):
                ph.set_HAP_refinements({"Scale": True}, histograms=[self.hist])
            self.gpx.save()
            self.gpx.do_refinements([{}])

    def refine_crystallite_size(
        self,
        refine_type: str = "isotropic",
        refine_dict: dict | None = None,
        phase: str | list[str] | None = None,
    ) -> None:
        """
        Refine the crystallite (domain) size broadening for one or more phases.

        Crystallite size broadening arises when coherently diffracting domains
        are small enough that diffraction peaks are broadened beyond the
        instrumental resolution.  It contributes to the Lorentzian width of
        the peak profile as 1/cosθ (Scherrer broadening).  The three models
        below correspond to different assumptions about the shape of the
        size distribution.

        Args:
            refine_type (``"isotropic"`` | ``"uniaxial"`` | ``"generalized"``, optional):
                Size broadening model (default ``"isotropic"``).  Ignored when
                ``refine_dict`` is supplied.
            refine_dict (dict or None, optional): Custom parameter dictionary following the same
                structure as the predefined ``SIZE_*_DICT`` constants (i.e. the top-level key must
                be ``"Size"``).  When provided, ``refine_type`` is ignored.
            phase (str, list of str, or None, optional): Phase name(s) to refine.
                ``None`` (default) refines all phases.

        **Models:**

        **isotropic** (default)
            Single size parameter *p* (µm) that scales the Lorentzian
            contribution uniformly in all directions.  The Scherrer formula
            gives an apparent crystallite size L = Kλ / (β cosθ), where β is
            the Lorentzian FWHM and K ≈ 0.9 is the Scherrer constant.  Use
            when there is no reason to expect shape anisotropy::

                SIZE_ISO_DICT = {
                    "Size": {"type": "isotropic", "refine": True, "value": 1.0}
                }

        **uniaxial**
            Two size parameters — ``equatorial`` (perpendicular to the unique
            axis) and ``axial`` (along the unique axis) — plus the unique
            ``axis`` direction in direct-space Miller indices.  Use for
            needle-shaped or plate-shaped crystallites whose long or short
            dimension is known::

                SIZE_UNI_DICT = {
                    "Size": {
                        "type": "uniaxial",
                        "refine": True,
                        "equatorial": 1.0,
                        "axial": 1.0,
                        "axis": [0, 0, 1],
                    }
                }

        **generalized**
            Full symmetry-constrained size tensor.  GSAS-II generates the
            symmetry-allowed terms automatically from the space group and
            refines all of them simultaneously.  Provides the most complete
            description of anisotropic size broadening but requires high-
            quality data with sufficient angular range::

                SIZE_GEN_DICT = {"Size": {"type": "generalized", "refine": True}}

        Note:
            Crystallite size broadening and microstrain broadening both contribute
            to Lorentzian peak widths and are strongly correlated.  Refine them
            sequentially — size first if the phase is nanocrystalline, mustrain
            first if the sample has significant lattice distortions — or use
            Williamson-Hall analysis to decide which dominates before starting the
            Rietveld refinement.
        """
        if isinstance(refine_dict, dict):
            refine_type = "personalized"

        if refine_type.lower() not in (
            "isotropic",
            "uniaxial",
            "generalized",
            "personalized",
        ):
            raise ValueError(
                f"Unknown size model '{refine_type}'. "
                "Valid options: 'isotropic', 'uniaxial', 'generalized'."
            )

        match refine_type:
            case "isotropic":
                ref_dict = SIZE_ISO_DICT
            case "uniaxial":
                ref_dict = SIZE_UNI_DICT
            case "generalized":
                ref_dict = SIZE_GEN_DICT
            case "personalized":
                ref_dict = refine_dict

        available = {ph.name: ph for ph in self.gpx.phases()}
        if phase is None:
            targets = list(available.values())
        else:
            names = [phase] if isinstance(phase, str) else list(phase)
            for name in names:
                if name not in available:
                    raise ValueError(
                        f"Phase '{name}' not found. "
                        f"Available phases: {list(available)}"
                    )
            targets = [available[n] for n in names]

        for ph in targets:
            ph.set_HAP_refinements(ref_dict, histograms=[self.hist])
            self.gpx.save()
            self.gpx.do_refinements([{}])
            print(f"Crystallite size ({refine_type}) refined for phase '{ph.name}'")

    def refine_mustrain(
        self,
        refine_type: str = "isotropic",
        refine_dict: dict | None = None,
        phase: str | list[str] | None = None,
    ) -> None:
        """
        Refine microstrain broadening for one or more phases.

        Microstrain broadening arises from heterogeneous lattice distortions
        within crystallites — local variations in d-spacing caused by defects,
        dislocations, composition gradients, or residual stress.  It
        contributes to the Lorentzian width of the peak profile as tanθ
        (Williamson-Hall slope), in contrast to size broadening which goes as
        1/cosθ.

        Args:
            refine_type (``"isotropic"`` | ``"uniaxial"`` | ``"generalized"``, optional):
                Microstrain model (default ``"isotropic"``).  Ignored when ``refine_dict``
                is supplied.
            refine_dict (dict or None, optional): Custom parameter dictionary following the same
                structure as the predefined ``MUSTRAIN_*_DICT`` constants (i.e. no top-level
                ``"Mustrain"`` key — the inner dict is passed directly).  When provided,
                ``refine_type`` is ignored.
            phase (str, list of str, or None, optional): Phase name(s) to refine.
                ``None`` (default) refines all phases.

        **Models:**

        **isotropic** (default)
            Single microstrain parameter *e* (µstrain) applied uniformly
            in all directions.  The Williamson-Hall relationship gives
            βₗ cosθ = Kλ/L + 4e sinθ, where βₗ is the Lorentzian FWHM.
            Use when there is no reason to expect directional strain
            anisotropy::

                MUSTRAIN_ISO_DICT = {
                    "type": "isotropic", "refine": True, "value": 1000.0
                }

        **uniaxial**
            Two strain parameters — ``equatorial`` (perpendicular to the
            unique axis) and ``axial`` (along the unique axis) — plus the
            ``axis`` direction in direct-space Miller indices.  Use when
            the dominant source of strain has a well-defined axis (e.g.
            uniaxial stress, rolled sheet, wire-drawn material)::

                MUSTRAIN_UNI_DICT = {
                    "type": "uniaxial",
                    "refine": True,
                    "equatorial": 1000.0,
                    "axial": 1000.0,
                    "axis": [0, 0, 1],
                }

        **generalized** (Stephens model)
            Full symmetry-constrained anisotropic strain tensor, expanded in
            a basis of symmetry-allowed S_hkl coefficients.  Captures
            complex hkl-dependent line broadening (e.g. due to stacking
            faults, anisotropic microstress, or plastic deformation).  GSAS-II
            generates the allowed terms from the space group automatically.
            Requires high-quality data and good angular coverage::

                MUSTRAIN_GEN_DICT = {"type": "generalized", "refine": True}

        Note:
            Microstrain and crystallite-size broadening are highly correlated
            because both affect the Lorentzian component of the peak profile.  A
            Williamson-Hall plot (βcosθ vs sinθ) can help determine which
            contribution dominates before committing to a model.  Avoid refining
            both simultaneously unless the data quality and angular range clearly
            support it.
        """
        if isinstance(refine_dict, dict):
            refine_type = "personalized"

        if refine_type.lower() not in (
            "isotropic",
            "uniaxial",
            "generalized",
            "personalized",
        ):
            raise ValueError(
                f"Unknown mustrain model '{refine_type}'. "
                "Valid options: 'isotropic', 'uniaxial', 'generalized'."
            )

        match refine_type:
            case "isotropic":
                ref_dict = MUSTRAIN_ISO_DICT
            case "uniaxial":
                ref_dict = MUSTRAIN_UNI_DICT
            case "generalized":
                ref_dict = MUSTRAIN_GEN_DICT
            case "personalized":
                ref_dict = refine_dict

        available = {ph.name: ph for ph in self.gpx.phases()}
        if phase is None:
            targets = list(available.values())
        else:
            names = [phase] if isinstance(phase, str) else list(phase)
            for name in names:
                if name not in available:
                    raise ValueError(
                        f"Phase '{name}' not found. "
                        f"Available phases: {list(available)}"
                    )
            targets = [available[n] for n in names]

        for ph in targets:
            ph.set_HAP_refinements(
                {"Mustrain": ref_dict},
                histograms=[self.hist],
            )
            self.gpx.save()
            self.gpx.do_refinements([{}])
            print(f"Microstrain ({refine_type}) refined for phase '{ph.name}'")

    def refine_hstrain(self, phase: str | list[str] | None = None) -> None:
        """
        Refine the hydrostatic/deviatoric strain tensor for one or more phases.

        HStrain models uniform macroscopic residual stress by allowing the
        average d-spacing of each reflection to shift in an hkl-dependent way.
        Unlike the global ``Zero`` offset or sample displacement (which shift
        all peaks by the same amount), HStrain produces shifts that vary with
        the reflection direction, reproducing the effect of a bulk elastic
        strain state on the diffraction pattern.

        The strain tensor is parametrised by symmetry-independent D_ij
        components whose number depends on the crystal system:

        ============== ========= ================================
        Crystal system  # params  Independent components
        ============== ========= ================================
        Cubic                  1  D11 (= D22 = D33)
        Hexagonal / Trigonal   2  D11 (= D22), D33
        Tetragonal             2  D11 (= D22), D33
        Orthorhombic           3  D11, D22, D33
        Monoclinic             4  D11, D22, D33, D13
        Triclinic              6  D11, D22, D33, D12, D13, D23
        ============== ========= ================================

        Args:
            phase (str, list of str, or None, optional): Phase name(s) to refine.
                ``None`` (default) refines all phases.

        Note:
            HStrain shifts peak positions without broadening them, so it is
            distinct from microstrain (which broadens peaks).  It is strongly
            correlated with the unit cell parameters — both affect d-spacings in
            an hkl-dependent manner.  Always converge the cell refinement before
            introducing HStrain, and be cautious in low-symmetry systems where
            the number of D_ij parameters approaches the number of independent
            d-spacing observations.
        """
        available = {ph.name: ph for ph in self.gpx.phases()}
        if phase is None:
            targets = list(available.values())
        else:
            names = [phase] if isinstance(phase, str) else list(phase)
            for name in names:
                if name not in available:
                    raise ValueError(
                        f"Phase '{name}' not found. "
                        f"Available phases: {list(available)}"
                    )
            targets = [available[n] for n in names]

        for ph in targets:
            ph.set_HAP_refinements({"HStrain": True}, histograms=[self.hist])
            self.gpx.save()
            self.gpx.do_refinements([{}])
            print(f"HStrain refined for phase '{ph.name}'")

    def refine_extinction(self, phase: str | list[str] | None = None) -> None:
        """
        Refine the primary extinction parameter for one or more phases.

        Primary extinction occurs when a strong Bragg reflection partially
        depletes the incident beam before it reaches deeper layers of the
        crystal, reducing the measured intensity below the kinematic
        (structure-factor-squared) prediction.  It is most pronounced for
        large, nearly perfect crystallites and for low-angle, high-intensity
        reflections where the structure factor is large.

        GSAS-II models primary extinction with a single scalar parameter *x*
        (the extinction coefficient).  The corrected intensity is:

            I_corr = I_kin / (1 + x · F²)

        where F² is the squared structure factor for that reflection.  When
        *x* = 0 there is no extinction correction.

        Args:
            phase (str, list of str, or None, optional): Phase name(s) to refine.
                ``None`` (default) refines all phases.

        Note:
            Primary extinction is only significant for well-crystallised,
            large-grained phases (grain size ≳ a few micrometres) or for samples
            with very low mosaic spread.  For typical powder diffraction specimens,
            where the crystallites are small and randomly oriented, extinction
            effects are negligible and this parameter should be kept fixed.

            Do not confuse primary extinction with secondary extinction (multiple
            scattering between grains), which is not modelled by this parameter,
            or with absorption, which is handled separately through the histogram
            sample parameters.  Refine extinction only after the structure is
            well determined; it is correlated with the overall scale factor and
            with U\ :sub:`iso` for the heaviest scatterers.
        """
        available = {ph.name: ph for ph in self.gpx.phases()}
        if phase is None:
            targets = list(available.values())
        else:
            names = [phase] if isinstance(phase, str) else list(phase)
            for name in names:
                if name not in available:
                    raise ValueError(
                        f"Phase '{name}' not found. "
                        f"Available phases: {list(available)}"
                    )
            targets = [available[n] for n in names]

        for ph in targets:
            ph.set_HAP_refinements({"Extinction": True}, histograms=[self.hist])
            self.gpx.save()
            self.gpx.do_refinements([{}])
            print(f"Extinction refined for phase '{ph.name}'")

    def refine_babinet(
        self,
        refine: str | list[str] = "BabA",
        phase: str | list[str] | None = None,
    ) -> None:
        """
        Refine Babinet complementary scattering parameters for one or more phases.

        The Babinet principle models the contribution of a diffuse, disordered
        component (e.g. amorphous matrix, disordered solvent, void space) that
        is complementary to the crystalline phase.  It modifies the calculated
        structure factors as:

            F²_corr = F²_cryst · exp(−BabU · Q²) + BabA · exp(−BabU · Q²)

        where Q = 4π sinθ / λ.  The two parameters are:

        ``BabA`` — Babinet amplitude.  Scales the complementary scattering
        contribution relative to the crystalline phase.  Physically represents
        the amount of disordered material surrounding or interpenetrating the
        crystalline domains.  Start with ``BabA`` alone; ``BabU`` is strongly
        correlated with it.

        ``BabU`` — Babinet U parameter (Å²).  Controls the Q-dependence
        (angular fall-off) of the complementary scattering, analogous to an
        isotropic displacement parameter for the disordered component.

        Args:
            refine (str or list of str, optional): Parameter(s) to refine: ``"BabA"``, ``"BabU"``,
                or both (default ``"BabA"``).  Refine ``BabA`` first; add ``BabU`` only once
                ``BabA`` is stable.
            phase (str, list of str, or None, optional): Phase name(s) to apply the refinement to.
                ``None`` (default) applies to all phases.

        Note:
            Babinet parameters are most useful when there is clear diffuse
            scattering beneath the Bragg peaks that cannot be accounted for by
            the background function alone.  They are strongly correlated with the
            overall scale factor and with the background coefficients — converge
            both before introducing Babinet terms.
        """
        params = [refine] if isinstance(refine, str) else list(refine)
        valid = {"BabA", "BabU"}
        invalid = set(params) - valid
        if invalid:
            raise ValueError(
                f"Unknown Babinet parameter(s) {sorted(invalid)}. "
                f"Valid options: {sorted(valid)}"
            )

        available = {ph.name: ph for ph in self.gpx.phases()}
        if phase is None:
            targets = list(available.values())
        else:
            names = [phase] if isinstance(phase, str) else list(phase)
            for name in names:
                if name not in available:
                    raise ValueError(
                        f"Phase '{name}' not found. "
                        f"Available phases: {list(available)}"
                    )
            targets = [available[n] for n in names]

        bab_dict = {"Babinet": {p: {"refine": True} for p in params}}
        for ph in targets:
            ph.set_HAP_refinements(bab_dict, histograms=[self.hist])
            self.gpx.save()
            self.gpx.do_refinements([{}])
            print(f"Babinet {params} refined for phase '{ph.name}'")

    def print_refinement_results(self) -> None:
        """Print a full summary of all refinable parameters and their current state."""
        print("\n" + "=" * 60)
        print("REFINEMENT RESULTS")
        print("=" * 60)

        # ── R-factors ──────────────────────────────────────────────────────────
        wR = self.hist.get_wR()
        residuals = self.hist.residuals
        print("\nR-factors:")
        print(f"  Rwp  = {wR:.4f} %" if wR is not None else "  Rwp  = n/a")
        for key in ("R", "Rb", "wRb", "wRmin"):
            v = residuals.get(key)
            if v is not None:
                print(f"  {key:<5}= {v:.4f} %")

        # ── Instrument parameters ──────────────────────────────────────────────
        ip = self.hist["Instrument Parameters"][0]
        profile_type = ip.get("Type", ["?"])[0]
        print(f"\nInstrument parameters  (profile: {profile_type}):")
        all_inst_params = [
            "Lam",
            "Lam1",
            "Lam2",
            "Zero",
            "Azimuth",
            "Polariz.",
            "U",
            "V",
            "W",
            "Z",
            "X",
            "Y",
            "SH/L",
            "alpha-0",
            "alpha-1",
            "beta-0",
            "beta-1",
        ]
        for p in all_inst_params:
            if p not in ip:
                continue
            entry = ip[p]
            val = entry[1] if isinstance(entry, list) else entry
            if not isinstance(val, (int, float)):
                continue
            refine = (
                entry[2] if (isinstance(entry, list) and len(entry) >= 3) else False
            )
            flag = "refine" if refine else "fixed"
            print(f"  {p:<12} = {val:14.6g}  ({flag})")

        # ── Sample parameters ──────────────────────────────────────────────────
        sp = self.hist.SampleParameters
        print("\nSample parameters:")
        for key in (
            "Scale",
            "Absorption",
            "Shift",
            "DisplaceX",
            "DisplaceY",
            "Transparency",
            "SurfRoughA",
            "SurfRoughB",
        ):
            if key not in sp:
                continue
            entry = sp[key]
            val, refine = entry[0], entry[1]
            flag = "refine" if refine else "fixed"
            print(f"  {key:<14} = {val:14.6g}  ({flag})")

        # ── Background ────────────────────────────────────────────────────────
        bkg = self.hist["Background"]
        bkg0 = bkg[0]
        print(f"\nBackground:")
        print(f"  Function   : {bkg0[0]}")
        print(f"  Refine     : {bkg0[1]}")
        print(f"  # coeffs   : {bkg0[2]}")
        print(f"  Coefficients: {[f'{c:.4g}' for c in bkg0[3:3+bkg0[2]]]}")
        n_debye = bkg[1].get("nDebye", 0)
        if n_debye:
            print(f"  Debye terms: {n_debye}")
            for i, t in enumerate(bkg[1].get("debyeTerms", [])):
                A, refA, R, refR, U, refU = t
                print(
                    f"    [{i}]  A={A:.4g}(ref={refA})  R={R:.4g}(ref={refR})  U={U:.4g}(ref={refU})"
                )

        # ── Per-phase results ──────────────────────────────────────────────────
        print("\nPhase results:")
        for ph in self.gpx.phases():
            print(f"\n  {'─'*54}")
            print(f"  Phase: {ph.name}")
            cell = ph.get_cell()
            cell_refine = ph.data["General"]["Cell"][0]
            print(f"  Cell ({'refine' if cell_refine else 'fixed'}):")
            print(
                f"    a={cell['length_a']:.5f}  b={cell['length_b']:.5f}  c={cell['length_c']:.5f}  Å"
            )
            print(
                f"    α={cell['angle_alpha']:.4f}  β={cell['angle_beta']:.4f}  γ={cell['angle_gamma']:.4f}  °"
            )
            print(f"    V={cell['volume']:.4f}  Å³")

            hap = ph.data["Histograms"].get(self.hist.name, {})
            if hap:
                sc = hap.get("Scale", [1.0, False])
                print(
                    f"  HAP Scale     : {sc[0]:.6g}  ({'refine' if sc[1] else 'fixed'})"
                )

                ext = hap.get("Extinction", [0.0, False])
                print(
                    f"  Extinction    : {ext[0]:.6g}  ({'refine' if ext[1] else 'fixed'})"
                )

                hs = hap.get("HStrain")
                if hs:
                    vals = hs[0]
                    flags = hs[1]
                    dij_str = "  ".join(
                        f"D{i}={v:.4g}({'R' if f else 'F'})"
                        for i, (v, f) in enumerate(zip(vals, flags))
                    )
                    print(f"  HStrain       : {dij_str}")

                sz = hap.get("Size")
                if sz:
                    model = sz[0]
                    ref_iso = any(sz[2]) if isinstance(sz[2], list) else sz[2]
                    if model == "isotropic":
                        print(
                            f"  Size          : {model}  val={sz[1][0]:.4g}  refine={ref_iso}"
                        )
                    elif model == "uniaxial":
                        print(
                            f"  Size          : {model}  eq={sz[1][0]:.4g}  ax={sz[1][1]:.4g}  axis={sz[3]}  refine={ref_iso}"
                        )
                    else:
                        print(f"  Size          : {model}  refine={any(sz[5])}")

                ms = hap.get("Mustrain")
                if ms:
                    model = ms[0]
                    ref_iso = any(ms[2]) if isinstance(ms[2], list) else ms[2]
                    if model == "isotropic":
                        print(
                            f"  Mustrain      : {model}  val={ms[1][0]:.4g}  refine={ref_iso}"
                        )
                    elif model == "uniaxial":
                        print(
                            f"  Mustrain      : {model}  eq={ms[1][0]:.4g}  ax={ms[1][1]:.4g}  axis={ms[3]}  refine={ref_iso}"
                        )
                    else:
                        print(f"  Mustrain      : {model}  refine={any(ms[5])}")

                po = hap.get("Pref.Ori.")
                if po:
                    model = po[0]
                    refine = po[2]
                    if model == "MD":
                        print(
                            f"  Pref.Ori.     : MD  ratio={po[1]:.4g}  axis={po[3]}  refine={refine}"
                        )
                    else:
                        print(
                            f"  Pref.Ori.     : SH  ord={po[4]}  axis={po[3]}  refine={refine}"
                        )

            # Atoms summary
            atoms = ph.atoms()
            if atoms:
                print(f"  Atoms ({len(atoms)}):")
                print(
                    f"    {'Label':<8} {'Element':<6} {'x':>9} {'y':>9} {'z':>9}"
                    f" {'Occ':>6} {'Uiso':>8} {'Flags'}"
                )
                print(f"    {'─'*68}")
                for atom in atoms:
                    x, y, z = atom.coordinates
                    uiso = atom.ADP if atom.adp_flag == "I" else "aniso"
                    uiso_str = f"{uiso:.5f}" if isinstance(uiso, float) else uiso
                    print(
                        f"    {atom.label:<8} {atom.element:<6}"
                        f" {x:9.5f} {y:9.5f} {z:9.5f}"
                        f" {atom.occupancy:6.4f} {uiso_str:>8}"
                        f"  {atom.refinement_flags!r}"
                    )

    # ------------------------------------------------------------------
    # Convenience / utility methods
    # ------------------------------------------------------------------

    def get_Rwp(self) -> float | None:
        """
        Return the current weighted-profile R-factor (Rwp) in percent.

        Rwp is the principal figure-of-merit for a Rietveld refinement.  It
        is defined as::

            Rwp = 100 · √[ Σ w·(yobs−ycalc)² / Σ w·yobs² ]

        where *w* = 1/σ² are the per-point weights.

        Returns:
            float or None: Rwp in percent, or ``None`` if no refinement has been run yet.
        """
        return self.hist.get_wR()

    def get_chi2(self) -> float | None:
        """
        Return the reduced chi-squared (goodness-of-fit, GOF) of the last cycle.

        The reduced χ² is::

            χ² = Σ w·(yobs−ycalc)² / (N_obs − N_vars)

        where *N_obs* is the number of data points and *N_vars* is the number
        of free parameters.  A value near 1.0 indicates a statistically ideal
        fit; values ≫ 1 suggest systematic misfit or underestimated errors.

        Returns:
            float or None: Reduced χ², or ``None`` if no refinement has been run yet.
        """
        return self.hist["data"][0].get("GOF")

    def save(self) -> None:
        """
        Save the GSAS-II project file to disk.

        Wraps :meth:`G2Project.save`.  Call this after any manual parameter
        edits (e.g. direct dictionary access) to ensure the ``.gpx`` file
        reflects the current in-memory state.
        """
        self.gpx.save()
        print(f"Project saved: {self.gpx.filename}")

    def export_pattern(self, path: str | Path = "pattern.csv") -> None:
        """
        Export the observed, calculated, difference, and background arrays
        to a plain-text CSV file.

        The output file contains five columns::

            2theta, yobs, ycalc, diff, background

        with a one-line header.  Values are written with six significant
        figures.  The file can be read by any spreadsheet or plotting tool.

        Args:
            path (str or Path, optional): Output file path (default ``"pattern.csv"``).  The
                extension determines the format:

                - ``.csv`` — comma-separated (default)
                - any other extension — space-separated (suitable for most
                  plotting packages that accept ``.xy`` or ``.dat``)
        """
        tth = self.hist.getdata("x")
        yobs = self.hist.getdata("yobs")
        ycalc = self.hist.getdata("ycalc")
        ybkg = self.hist.getdata("background")
        diff = yobs - ycalc

        path = Path(path)
        sep = "," if path.suffix.lower() == ".csv" else "  "
        header = sep.join(["2theta", "yobs", "ycalc", "diff", "background"])

        data = np.column_stack([tth, yobs, ycalc, diff, ybkg])
        fmt = "%.6g"
        np.savetxt(str(path), data, delimiter=sep, header=header, comments="", fmt=fmt)
        print(f"Pattern exported to: {path}")

    def print_HAP_parameters(self, phase: str | list[str] | None = None) -> None:
        """
        Print a focused table of all Histogram-And-Phase (HAP) parameters for
        one or more phases linked to the current histogram.

        HAP parameters control the per-phase contribution to the powder
        pattern and live in the ``Histograms`` sub-dictionary of each phase.
        This method shows the current *values* together with their refinement
        flags so that the state of the model is immediately visible.

        Args:
            phase (str, list of str, or None, optional): Phase name(s) to inspect.
                ``None`` (default) prints all phases linked to the current histogram.

        **Printed quantities:**

        Scale
            Phase fraction scale factor (dimensionless).  Relates to the
            weight fraction via the Brindley–Hill formula.
        Extinction
            Isotropic extinction coefficient (dimensionless, Lorentz model).
        HStrain (D_ij)
            Hydrostatic / deviatoric strain tensor components in the crystal
            coordinate system.  Up to six independent D_ij values, labelled
            D11, D22, D33, D12, D13, D23 (symmetry-allowed subset depends on
            the Laue class).
        Size
            Apparent crystallite size parameters in µm.  Model is one of
            ``isotropic`` (one scalar), ``uniaxial`` (equatorial + axial along
            a specified crystallographic axis), or ``generalized`` (full
            symmetry-adapted harmonic expansion).
        Mustrain
            Microstrain parameters in µε (parts per million).  Same three
            models as Size.  The Lorentzian peak broadening from microstrain
            is proportional to tan θ.
        Pref.Ori.
            Preferred orientation.  Either ``MD`` (March–Dollase scalar
            distribution) or ``SH`` (spherical harmonics expansion).
        Babinet
            Babinet solvent-correction parameters ``BabA`` (amplitude, Å²)
            and ``BabU`` (thermal factor, Å²).  Used for macromolecular or
            porous samples where a featureless electron-density background
            contributes diffuse scattering.
        """
        available = {ph.name: ph for ph in self.gpx.phases()}
        if phase is None:
            targets = list(available.values())
        else:
            names = [phase] if isinstance(phase, str) else list(phase)
            for name in names:
                if name not in available:
                    raise ValueError(
                        f"Phase '{name}' not found. "
                        f"Available phases: {list(available)}"
                    )
            targets = [available[n] for n in names]

        for ph in targets:
            hap = ph.data["Histograms"].get(self.hist.name, {})
            if not hap:
                print(
                    f"\nPhase '{ph.name}' is not linked to histogram '{self.hist.name}'."
                )
                continue

            print("\n" + "=" * 60)
            print(f"HAP parameters  —  phase: {ph.name}")
            print(f"                   histogram: {self.hist.name}")
            print("=" * 60)

            sc = hap.get("Scale", [1.0, False])
            print(f"  Scale      : {sc[0]:.6g}  (refine={sc[1]})")

            ext = hap.get("Extinction", [0.0, False])
            print(f"  Extinction : {ext[0]:.6g}  (refine={ext[1]})")

            hs = hap.get("HStrain")
            if hs:
                vals, flags = hs[0], hs[1]
                dij_labels = ["D11", "D22", "D33", "D12", "D13", "D23"]
                dij_parts = [
                    f"{dij_labels[i]}={v:.4g}({'R' if f else 'F'})"
                    for i, (v, f) in enumerate(zip(vals, flags))
                ]
                print(f"  HStrain    : {', '.join(dij_parts)}")

            sz = hap.get("Size")
            if sz:
                model = sz[0]
                ref_flag = any(sz[2]) if isinstance(sz[2], list) else bool(sz[2])
                if model == "isotropic":
                    print(
                        f"  Size       : {model}  value={sz[1][0]:.4g} µm  refine={ref_flag}"
                    )
                elif model == "uniaxial":
                    print(
                        f"  Size       : {model}  eq={sz[1][0]:.4g}  ax={sz[1][1]:.4g} µm"
                        f"  axis={sz[3]}  refine={ref_flag}"
                    )
                else:
                    gen_vals = sz[4]
                    print(
                        f"  Size       : {model}  {len(gen_vals)} gen. coeffs  refine={any(sz[5])}"
                    )

            ms = hap.get("Mustrain")
            if ms:
                model = ms[0]
                ref_flag = any(ms[2]) if isinstance(ms[2], list) else bool(ms[2])
                if model == "isotropic":
                    print(
                        f"  Mustrain   : {model}  value={ms[1][0]:.4g} µε  refine={ref_flag}"
                    )
                elif model == "uniaxial":
                    print(
                        f"  Mustrain   : {model}  eq={ms[1][0]:.4g}  ax={ms[1][1]:.4g} µε"
                        f"  axis={ms[3]}  refine={ref_flag}"
                    )
                else:
                    gen_vals = ms[4]
                    print(
                        f"  Mustrain   : {model}  {len(gen_vals)} gen. coeffs  refine={any(ms[5])}"
                    )

            po = hap.get("Pref.Ori.")
            if po:
                model = po[0]
                ref = po[2]
                if model == "MD":
                    print(
                        f"  Pref.Ori.  : MD  ratio={po[1]:.4g}  axis={po[3]}  refine={ref}"
                    )
                else:
                    print(
                        f"  Pref.Ori.  : SH  order={po[4]}  axis={po[3]}  refine={ref}"
                    )

            bab = hap.get("Babinet", {})
            if bab:
                for key in ("BabA", "BabU"):
                    entry = bab.get(key, {})
                    val = entry.get("BabVal", 0.0)
                    ref = entry.get("refine", False)
                    print(f"  Babinet {key[-1]}  : {val:.4g}  (refine={ref})")

    def set_HAP_parameter(
        self,
        parameter: str,
        value: float,
        phase: str | list[str] | None = None,
    ) -> None:
        """
        Set a Histogram-And-Phase (HAP) parameter to a fixed value and freeze it.

        The supported *parameter* strings and the HAP entries they control are:

        +--------------+-------------------------------------------------------+
        | parameter    | HAP entry                                             |
        +==============+=======================================================+
        | ``"Scale"``  | Phase fraction scale factor ``hap["Scale"][0]``      |
        +--------------+-------------------------------------------------------+
        | ``"Extinction"`` | Extinction coefficient ``hap["Extinction"][0]``   |
        +--------------+-------------------------------------------------------+
        | ``"D11"`` … ``"D33"``, ``"D12"``, ``"D13"``, ``"D23"``             |
        |              | Individual HStrain tensor components                  |
        +--------------+-------------------------------------------------------+
        | ``"Size"``   | Isotropic crystallite size value (µm)                 |
        +--------------+-------------------------------------------------------+
        | ``"Mustrain"`` | Isotropic microstrain value (µε)                    |
        +--------------+-------------------------------------------------------+
        | ``"MD"``     | March–Dollase preferred orientation ratio             |
        +--------------+-------------------------------------------------------+
        | ``"BabA"``   | Babinet amplitude ``BabA``                            |
        +--------------+-------------------------------------------------------+
        | ``"BabU"``   | Babinet thermal factor ``BabU``                       |
        +--------------+-------------------------------------------------------+

        Call :meth:`print_HAP_parameters` first to check available keys and
        current values for a given phase.

        Args:
            parameter (str): HAP parameter to set (see table above).
            value (float): Value to assign and freeze.
            phase (str, list of str, or None, optional): Phase name(s) to update.
                ``None`` (default) updates all phases linked to the current histogram.

        Raises:
            ValueError: If *parameter* is not a recognised key, or if a model-dependent
                parameter (e.g. ``"Size"`` in uniaxial/generalized mode) cannot be
                set as a single scalar.
        """
        _DIJ = {"D11": 0, "D22": 1, "D33": 2, "D12": 3, "D13": 4, "D23": 5}
        _VALID = {
            "Scale",
            "Extinction",
            "Size",
            "Mustrain",
            "MD",
            "BabA",
            "BabU",
        } | _DIJ.keys()

        if parameter not in _VALID:
            raise ValueError(
                f"'{parameter}' is not a supported HAP parameter. "
                f"Valid options: {sorted(_VALID)}"
            )

        available = {ph.name: ph for ph in self.gpx.phases()}
        if phase is None:
            targets = list(available.values())
        else:
            names = [phase] if isinstance(phase, str) else list(phase)
            for name in names:
                if name not in available:
                    raise ValueError(
                        f"Phase '{name}' not found. "
                        f"Available phases: {list(available)}"
                    )
            targets = [available[n] for n in names]

        for ph in targets:
            hap = ph.data["Histograms"].get(self.hist.name)
            if hap is None:
                print(
                    f"  Skipping phase '{ph.name}': not linked to '{self.hist.name}'."
                )
                continue

            if parameter == "Scale":
                hap["Scale"][0] = value
                hap["Scale"][1] = False

            elif parameter == "Extinction":
                hap["Extinction"][0] = value
                hap["Extinction"][1] = False

            elif parameter in _DIJ:
                idx = _DIJ[parameter]
                hs = hap.get("HStrain")
                if hs is None:
                    raise ValueError(
                        f"Phase '{ph.name}' has no HStrain entry. "
                        "Enable HStrain first."
                    )
                hs[0][idx] = value
                hs[1][idx] = False

            elif parameter == "Size":
                sz = hap.get("Size")
                if sz is None:
                    raise ValueError(f"Phase '{ph.name}' has no Size entry.")
                if sz[0] != "isotropic":
                    raise ValueError(
                        f"Phase '{ph.name}' Size model is '{sz[0]}'. "
                        "set_HAP_parameter only supports isotropic Size."
                    )
                sz[1][0] = value
                sz[2][0] = False

            elif parameter == "Mustrain":
                ms = hap.get("Mustrain")
                if ms is None:
                    raise ValueError(f"Phase '{ph.name}' has no Mustrain entry.")
                if ms[0] != "isotropic":
                    raise ValueError(
                        f"Phase '{ph.name}' Mustrain model is '{ms[0]}'. "
                        "set_HAP_parameter only supports isotropic Mustrain."
                    )
                ms[1][0] = value
                ms[2][0] = False

            elif parameter == "MD":
                po = hap.get("Pref.Ori.")
                if po is None:
                    raise ValueError(f"Phase '{ph.name}' has no Pref.Ori. entry.")
                if po[0] != "MD":
                    raise ValueError(
                        f"Phase '{ph.name}' preferred orientation model is '{po[0]}', not 'MD'."
                    )
                po[1] = value
                po[2] = False

            elif parameter in ("BabA", "BabU"):
                bab = hap.get("Babinet")
                if bab is None or parameter not in bab:
                    raise ValueError(
                        f"Phase '{ph.name}' has no Babinet '{parameter}' entry."
                    )
                bab[parameter]["BabVal"] = value
                bab[parameter]["refine"] = False

            print(
                f"  HAP '{parameter}' for phase '{ph.name}' set to {value} and frozen."
            )

        self.gpx.save()

    def fix_all_parameters(self) -> None:
        """
        Fix (freeze) every refinement flag in the project.

        Clears refinement flags for: background (polynomial + Debye terms),
        all instrument parameters, histogram scale, and per-phase cell,
        atoms, HAP scale, extinction, hydrostatic strain, crystallite size,
        microstrain, and preferred orientation.  Does not run a refinement cycle.
        """
        # Background polynomial
        bkg = self.hist["Background"]
        bkg[0][1] = False
        for term in bkg[1].get("debyeTerms", []):
            term[1] = term[3] = term[5] = False  # refA, refR, refU

        # Instrument parameters
        ip = self.hist["Instrument Parameters"][0]
        for val in ip.values():
            if isinstance(val, list) and len(val) >= 3:
                val[2] = False

        # Histogram scale and sample parameters
        sp = self.hist.SampleParameters
        for key in ("Scale", "Absorption", "Shift", "DisplaceX", "DisplaceY"):
            if key in sp and isinstance(sp[key], list) and len(sp[key]) >= 2:
                sp[key][1] = False

        # Per-phase
        for ph in self.gpx.phases():
            ph.set_refinements({"Cell": False})
            for atom in ph.atoms():
                atom.refinement_flags = ""
            ph.set_HAP_refinements(
                {
                    "Scale": False,
                    "Extinction": False,
                    "HStrain": False,
                    "Size": {"refine": False},
                    "Mustrain": {"refine": False},
                    "Pref.Ori.": False,
                },
                histograms=[self.hist],
            )

        self.gpx.save()
        print("All parameters fixed.")

    def print_refined_variables(self) -> None:
        """Print all currently refined variables with their values and esds."""
        cov_data = self.gpx["Covariance"]["data"]
        vary_list = cov_data.get("varyList", [])
        variables = cov_data.get("variables", [])
        sigmas    = cov_data.get("sig", [])

        if not vary_list:
            print("No refined variables found (run a refinement first).")
            return

        print("\n" + "=" * 60)
        print("REFINED VARIABLES")
        print("=" * 60)
        print(f"  {'Parameter':<40} {'Value':>14} {'Esd':>14}")
        print("  " + "-" * 68)
        for i, var in enumerate(vary_list):
            val     = variables[i] if i < len(variables) else float("nan")
            sig     = sigmas[i]    if i < len(sigmas)    else None
            esd_str = f"{sig:.6g}" if sig is not None else "n/a"
            print(f"  {var:<40} {val:>14.6g} {esd_str:>14}")
        print(f"\n  Total refined parameters: {len(vary_list)}")

    def print_covariance_matrix(self) -> None:
        """Print the correlation matrix (normalised covariance) of all refined variables."""
        cov_data = self.gpx["Covariance"]["data"]
        vary_list = cov_data.get("varyList", [])
        cov_matrix = cov_data.get("covMatrix")

        if not vary_list or cov_matrix is None or not len(cov_matrix):
            print("No covariance data found (run a refinement first).")
            return

        sigmas = np.sqrt(np.diag(cov_matrix))
        with np.errstate(invalid="ignore"):
            corr = cov_matrix / np.outer(sigmas, sigmas)
        corr = np.nan_to_num(corr)

        n = len(vary_list)
        col_w = 10
        label_w = max(len(v) for v in vary_list) + 2

        print("\n" + "=" * 60)
        print("CORRELATION MATRIX")
        print("=" * 60)

        # Header row with short indices
        header = " " * label_w + "".join(f"{i:>{col_w}}" for i in range(n))
        print(header)
        print(" " * label_w + "-" * (col_w * n))

        for i, var in enumerate(vary_list):
            row = f"{var:<{label_w}}"
            for j in range(n):
                row += f"{corr[i, j]:>{col_w}.3f}"
            print(row)

        print("\n  Index → parameter mapping:")
        for i, var in enumerate(vary_list):
            print(f"    {i:3d}  {var}")

    def plot_covariance_matrix(
        self, show: bool = True, figsize: tuple = (6, 4)
    ) -> tuple:
        """
        Plot the correlation matrix (normalised covariance) as a heatmap.

        The correlation matrix is derived by normalising the covariance matrix
        so that diagonal entries are 1 and off-diagonal entries are Pearson
        correlation coefficients in [-1, 1].

        Args:
            show (bool, optional): Call ``plt.show()`` after creating the figure
                (default ``True``).
            figsize (tuple of (float, float), optional): Figure size in inches as
                ``(width, height)`` (default ``(12, 9)``).

        Returns:
            tuple: ``(fig, ax)`` — the :class:`matplotlib.figure.Figure` and
            :class:`matplotlib.axes.Axes` objects so the caller can further
            customise or save the plot.
        """
        cov_data = self.gpx["Covariance"]["data"]
        vary_list = cov_data.get("varyList", [])
        cov_matrix = cov_data.get("covMatrix")

        if not vary_list or cov_matrix is None or not len(cov_matrix):
            raise RuntimeError("No covariance data found (run a refinement first).")

        sigmas = np.sqrt(np.diag(cov_matrix))
        with np.errstate(invalid="ignore"):
            corr = cov_matrix / np.outer(sigmas, sigmas)
        corr = np.nan_to_num(corr)

        n = len(vary_list)
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        fig.colorbar(im, ax=ax, label="Correlation coefficient")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(vary_list, rotation=90, fontsize=8)
        ax.set_yticklabels(vary_list, fontsize=8)
        ax.set_title("Correlation matrix")

        fig.tight_layout()

        if show:
            plt.show()

        return fig, ax

    def print_atoms(self, phase: str | list[str] | None = None) -> None:
        """
        Print a table of all atoms in one or more phases with their current
        refinement state.

        Args:
            phase (str, list of str, or None, optional): Phase name(s) to inspect.
                ``None`` (default) prints all phases in the project.  Names must match
                those used in :meth:`add_phase`.
        """
        available = {ph.name: ph for ph in self.gpx.phases()}
        if phase is None:
            targets = list(available.values())
        else:
            names = [phase] if isinstance(phase, str) else list(phase)
            for name in names:
                if name not in available:
                    raise ValueError(
                        f"Phase '{name}' not found. "
                        f"Available phases: {list(available)}"
                    )
            targets = [available[n] for n in names]

        for ph in targets:
            atoms = ph.atoms()
            print("\n" + "=" * 72)
            print(f"Phase: {ph.name}  ({len(atoms)} atoms)")
            print("=" * 72)
            print(
                f"  {'Label':<8} {'Element':<8} {'Type':<8}"
                f" {'x':>8} {'y':>8} {'z':>8}"
                f" {'Occ':>6} {'Mult':>5} {'ADP':>4} {'Flags'}"
            )
            print("  " + "-" * 70)
            for atom in atoms:
                x, y, z = atom.coordinates
                print(
                    f"  {atom.label:<8} {atom.element:<8} {atom.type:<8}"
                    f" {x:8.5f} {y:8.5f} {z:8.5f}"
                    f" {atom.occupancy:6.4f} {atom.mult:5d}"
                    f" {atom.adp_flag:>4}  {atom.refinement_flags!r}"
                )

    def print_parameter_info(self, category: str | None = None) -> None:
        """
        Print a reference summary of every GSAS-II refinable parameter,
        grouped by category.

        Parameters follow the naming pattern ``p:h:<var>:n``, where ``p`` is
        the phase number, ``h`` is the histogram number, ``<var>`` is the
        variable name shown below, and ``n`` is the atom index.  Components
        that do not apply are omitted (e.g. ``:h:<var>`` for histogram-only
        parameters).

        Args:
            category (str or None, optional): Restrict output to one category.
                One of ``"instrument"``, ``"hap"``, ``"phase"``, ``"sample"``,
                or ``"background"``.  ``None`` (default) prints all categories.
        """
        _SECTIONS = {
            "instrument": (
                "Instrument Parameters (CW)",
                [
                    ("Lam",         "Wavelength",                                                   "Å"),
                    ("Zero",        "Two-theta zero-point correction",                              "degrees"),
                    ("U",           "Cagliotti Gaussian – U  (FWHM²=Utan²θ+Vtanθ+W)",             "degrees²"),
                    ("V",           "Cagliotti Gaussian – V",                                       "degrees²"),
                    ("W",           "Cagliotti Gaussian – W",                                       "degrees²"),
                    ("X",           "Lorentzian broadening – X  (scales as 1/cosθ)",               "degrees"),
                    ("Y",           "Lorentzian broadening – Y  (scales as tanθ)",                 "degrees"),
                    ("Z",           "Lorentzian broadening – Z  (constant term)",                  "degrees"),
                    ("SH/L",        "Finger-Cox-Jephcoat axial asymmetry = S/L + H/L",             "dimensionless"),
                    ("Polariz.",    "Beam polarization fraction",                                   "0 – 1"),
                    ("I(L2)/I(L1)","Kα₂/Kα₁ intensity ratio (dual-wavelength sources only)",      "dimensionless"),
                ],
            ),
            "hap": (
                "HAP Parameters  (Histogram-And-Phase)",
                [
                    ("Scale",      "Phase fraction scale factor",                                        "dimensionless"),
                    ("Extinction", "Primary extinction coefficient",                                     "dimensionless"),
                    ("HStrain",    "Anisotropic strain tensor Dij – hkl-dependent peak shifts",         "Å⁻²"),
                    ("Mustrain",   "Microstrain broadening (isotropic / uniaxial / generalized)",        "×10⁻⁶"),
                    ("Size",       "Crystallite size / Scherrer broadening (isotropic / uniaxial / ellipsoidal)", "Å"),
                    ("Pref.Ori.", "Preferred orientation: March-Dollase ratio or SH coefficients",      "dimensionless"),
                    ("LeBail",     "Le Bail extraction flag – bypasses structure factors (bool)",        "—"),
                ],
            ),
            "phase": (
                "Phase Parameters",
                [
                    ("Cell / a", "Lattice length a",    "Å"),
                    ("Cell / b", "Lattice length b",    "Å"),
                    ("Cell / c", "Lattice length c",    "Å"),
                    ("Cell / α", "Lattice angle alpha", "degrees"),
                    ("Cell / β", "Lattice angle beta",  "degrees"),
                    ("Cell / γ", "Lattice angle gamma", "degrees"),
                    ("Cell / V", "Unit-cell volume",    "Å³"),
                ],
            ),
            "sample": (
                "Sample Parameters",
                [
                    ("Scale",         "Overall histogram scale factor",                          "dimensionless"),
                    ("Absorption",    "Linear absorption coeff. μr (Debye–Scherrer only)",       "dimensionless"),
                    ("DisplaceX",     "Sample displacement perpendicular to beam",               "μm"),
                    ("DisplaceY",     "Sample displacement along beam direction",                "μm"),
                    ("SurfaceRoughA", "Surface roughness A – Surotti 1972 (Bragg–Brentano only)","dimensionless"),
                    ("SurfaceRoughB", "Surface roughness B (Bragg–Brentano only)",               "dimensionless"),
                ],
            ),
            "background": (
                "Background Parameters",
                [
                    ("Coefficients", "Background function coefficients (e.g. Chebyshev polynomial)", "counts"),
                    ("Debye terms",  "Optional diffuse Debye scattering terms",                       "—"),
                    ("Peaks",        "Optional background peak positions and widths",                  "—"),
                ],
            ),
        }

        categories = (
            list(_SECTIONS.keys())
            if category is None
            else [category.lower()]
        )

        for cat in categories:
            if cat not in _SECTIONS:
                raise ValueError(
                    f"Unknown category '{cat}'. "
                    f"Valid options: {list(_SECTIONS)}"
                )
            title, params = _SECTIONS[cat]
            col_w  = max(len(name)  for name, _,  _     in params) + 2
            unit_w = max(len(units) for _,    _, units   in params) + 2
            print("\n" + "=" * 72)
            print(title)
            print("=" * 72)
            print(f"  {'Parameter':<{col_w}} {'Units':<{unit_w}} Description")
            print(f"  {'-'*col_w} {'-'*unit_w} {'-'*20}")
            for name, desc, units in params:
                print(f"  {name:<{col_w}} {units:<{unit_w}} {desc}")

        print()
        print("Naming convention:  p:h:<var>:n")
        print("  p = phase number   h = histogram number   n = atom number")
        print("  Omit any component that does not apply.")

    def print_phases(self, phase: str | None = None) -> None:
        """
        Print a summary of phases in the project, including cell
        parameters, space group, composition, and the current refinement
        state of all phase-level and HAP parameters.

        Args:
            phase (str or None, optional): Name of a single phase to print.
                ``None`` (default) prints all phases.

        **HAP parameters** (Histogram-And-Phase, i.e. parameters that are
        specific to the combination of a phase and a powder histogram):

        ``Scale``
            Phase fraction scale factor (dimensionless).  Proportional to the
            weight fraction of the phase in the mixture.  Refined
            independently for each phase/histogram pair.

        ``Extinction``
            Primary extinction correction coefficient (dimensionless).
            Accounts for the reduction in diffracted intensity caused by
            re-diffraction within large, perfect crystalline domains.  Only
            significant for large single-domain crystallites; usually left
            fixed at 0.

        ``HStrain`` (Dij)
            Hydrostatic / anisotropic strain broadening coefficients
            (units: Å⁻² for *d*-space, or dimensionless in reciprocal
            lattice units depending on the GSAS-II convention).  The Dij
            tensor components model a homogeneous lattice strain that shifts
            peak positions in an hkl-dependent way without changing peak
            widths.  The number of independent Dij terms is determined by
            the Laue symmetry of the phase (e.g. 1 for cubic, up to 6 for
            triclinic).  Typical use: residual-stress or
            solid-solution-induced lattice distortions.

        ``Size``
            Mean apparent crystallite (domain) size, related to Scherrer
            broadening (units: Å).  Three models are available:

            * ``isotropic`` — single scalar size, same in all directions.
            * ``uniaxial`` — equatorial size and axial size along a
              specified crystallographic axis.
            * ``generalized`` — full set of symmetry-allowed spherical
              harmonic coefficients for arbitrary anisotropic broadening.

        ``Mustrain``
            Microstrain broadening coefficient (dimensionless, ×10⁻⁶).
            Models peak broadening caused by local, non-uniform lattice
            distortions (dislocations, defects).  Three models are
            available with the same isotropic / uniaxial / generalized
            choice as Size.  Mustrain and Size broadening are added in
            quadrature in the peak-profile function.

        ``Pref.Ori.`` (preferred orientation / texture)
            Correction for non-random crystallite orientation (texture).
            Two models are available:

            * ``MD`` (March-Dollase) — single ratio parameter *r*
              (dimensionless) along a preferred axis.  *r* = 1 means no
              texture; *r* < 1 needle texture; *r* > 1 plate texture.
            * ``SH`` (spherical harmonics) — series of harmonic
              coefficients (dimensionless) up to order *L*, capturing
              complex multi-component textures.
        """
        available = {ph.name: ph for ph in self.gpx.phases()}
        if phase is not None:
            if phase not in available:
                raise ValueError(
                    f"Phase '{phase}' not found. "
                    f"Available phases: {list(available)}"
                )
            phases = [available[phase]]
        else:
            phases = list(available.values())

        for ph in phases:
            gen = ph.data["General"]
            cell = gen["Cell"]  # [refine, a, b, c, α, β, γ, V]
            sg = gen["SGData"]["SpGrp"]
            phase_type = gen["Type"]
            c = ph.get_cell()

            print("\n" + "=" * 60)
            print(f"Phase : {ph.name}")
            print(f"Type  : {phase_type}")
            print(f"SG    : {sg}")
            print(f"Atoms : {len(ph.atoms())}")
            comp = ", ".join(f"{el}{n:.2g}" for el, n in ph.composition.items())
            print(f"Comp  : {comp}")
            print(f"Linked histograms: {ph.histograms()}")

            refine_flag = "refine" if cell[0] else "fixed"
            print(f"\n  Cell parameters ({refine_flag}):")
            print(
                f"    a={c['length_a']:.5f}  b={c['length_b']:.5f}  c={c['length_c']:.5f}  Å"
            )
            print(
                f"    α={c['angle_alpha']:.4f}  β={c['angle_beta']:.4f}  γ={c['angle_gamma']:.4f}  °"
            )
            print(f"    V={c['volume']:.3f}  Å³")

            hap = ph.data["Histograms"].get(self.hist.name, {})
            if hap:
                print(f"\n  HAP refinement state ({self.hist.name}):")

                sc = hap.get("Scale", [1.0, False])
                print(f"    Scale      : {sc[0]:.6g}  (refine={sc[1]})")

                ext = hap.get("Extinction", [0.0, False])
                print(f"    Extinction : {ext[0]:.6g}  (refine={ext[1]})")

                hs = hap.get("HStrain")
                if hs:
                    vals = hs[0]
                    flags = hs[1]
                    dij_str = "  ".join(
                        f"D{i}={v:.4g}({'R' if f else 'F'})"
                        for i, (v, f) in enumerate(zip(vals, flags))
                    )
                    print(f"    HStrain    : {dij_str if dij_str else 'none'}")

                sz = hap.get("Size")
                if sz:
                    model = sz[0]
                    ref_iso = any(sz[2]) if isinstance(sz[2], list) else sz[2]
                    if model == "isotropic":
                        print(
                            f"    Size       : {model}  val={sz[1][0]:.4g}  refine={ref_iso}"
                        )
                    elif model == "uniaxial":
                        print(
                            f"    Size       : {model}  eq={sz[1][0]:.4g}  ax={sz[1][1]:.4g}  axis={sz[3]}  refine={ref_iso}"
                        )
                    else:
                        print(f"    Size       : {model}  refine={any(sz[5])}")

                ms = hap.get("Mustrain")
                if ms:
                    model = ms[0]
                    ref_iso = any(ms[2]) if isinstance(ms[2], list) else ms[2]
                    if model == "isotropic":
                        print(
                            f"    Mustrain   : {model}  val={ms[1][0]:.4g}  refine={ref_iso}"
                        )
                    elif model == "uniaxial":
                        print(
                            f"    Mustrain   : {model}  eq={ms[1][0]:.4g}  ax={ms[1][1]:.4g}  axis={ms[3]}  refine={ref_iso}"
                        )
                    else:
                        print(f"    Mustrain   : {model}  refine={any(ms[5])}")

                po = hap.get("Pref.Ori.")
                if po:
                    model = po[0]  # 'MD' or 'SH'
                    refine = po[2]
                    if model == "MD":
                        print(
                            f"    Pref.Ori.  : MD  ratio={po[1]:.4g}  axis={po[3]}  refine={refine}"
                        )
                    else:
                        print(
                            f"    Pref.Ori.  : SH  ord={po[4]}  axis={po[3]}  refine={refine}"
                        )

    def plot_results(
        self,
        image_path: Path = "calibration_plot.png",
        show: bool = True,
        figsize: tuple = (9, 6),
    ) -> None:
        """
        Plot the Rietveld fit (observed / calculated / difference) and save to disk.

        Args:
            image_path (Path, optional): Output image file (default ``"calibration_plot.png"``).
            show (bool, optional): If ``True``, call ``plt.show()`` after saving (default ``True``).
            figsize (tuple of (float, float), optional): Figure size in inches as
                ``(width, height)`` (default ``(12, 9)``).
        """
        wR = self.hist.get_wR()
        print("\n" + "=" * 60)
        print("Generating calibration plot")
        print("=" * 60)

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            2, 1, figure=fig, height_ratios=[3, 1], hspace=0.08, wspace=0.35
        )

        ax_main = fig.add_subplot(gs[0, 0])
        ax_diff = fig.add_subplot(gs[1, 0], sharex=ax_main)

        # --- Observed / Calculated / Difference ---
        tth = self.hist.getdata("x")
        yobs = self.hist.getdata("yobs")
        ycalc = self.hist.getdata("ycalc")
        ybkg = self.hist.getdata("background")
        diff = yobs - ycalc

        ax_main.plot(tth, yobs, "k.", ms=2, label="Observed")
        ax_main.plot(tth, ycalc, "r-", lw=1, label="Calculated")
        ax_main.plot(tth, ybkg, "b--", lw=0.8, label="Background")

        # Reflection tick marks for both phases
        # colours_ticks = {"calibrant": "magenta", "Al_holder": "darkorange"}
        yrange = yobs.max() - yobs.min()
        tick_y0 = yobs.min() - 0.02 * yrange
        for ii, ph in enumerate(self.gpx.phases()):
            try:
                reflist = self.hist.reflections()[ph.name]["RefList"]
                ref_tth = reflist[:, 5]
                # colour = colours_ticks.get(ph.name, "green")
                ax_main.vlines(
                    ref_tth,
                    tick_y0,
                    tick_y0 + 0.04 * yrange,
                    color=COLORS[ii],
                    lw=0.8,
                    label=f"{ph.name} reflections",
                )
            except Exception:
                pass

        ax_main.set_ylabel("Intensity")
        residuals = self.hist.residuals
        chi2 = residuals.get("GOF")
        stat_parts = []
        if wR is not None:
            stat_parts.append(f"Rwp = {wR:.2f} %")
        if chi2 is not None:
            stat_parts.append(f"χ² = {chi2:.4f}")
        stats_str = "   ".join(stat_parts)
        ax_main.set_title(
            f"{self.calibrant_composition}\n{stats_str}"
            if stats_str
            else "Rietveld fit"
        )
        ax_main.legend(fontsize=7, markerscale=2)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        ax_diff.plot(tth, diff, color="forestgreen", lw=0.8)
        ax_diff.axhline(0, color="k", lw=0.5, ls="--")
        ax_diff.set_ylabel("Obs−Calc")
        ax_diff.set_xlabel("2θ (degrees)")

        plt.suptitle(
            f"Refinement results of {self.sample_name}.",
            fontsize=12,
            fontweight="bold",
            y=1.01,
        )
        plt.savefig(str(image_path), dpi=150, bbox_inches="tight")
        if show:
            plt.show()


class InstrumentCalibration(BaseRefinement):
    """
    Specialised refinement class for instrument calibration using a known calibrant.

    Extends :class:`BaseRefinement` with output paths routed to a ``calibration/``
    sub-folder, a dedicated method to export the calibrated ``.instprm`` file, and
    a richer diagnostic plot (parameter bar chart, FWHM model, parameter table).
    """

    def __init__(
        self,
        acquisition_file,
        sample_name,
        scan_type="half-turn",
        translation_motor="dty",
        rotation_motor="rot",
        outer_loop_motor="translation",
        beam_size=0.0001,
        beam_energy=44,
        tth_lims=(None, None),
        xy_file=Path("integrated_data.xy"),
        param_file=Path("calibrated_instrument.instprm"),
        polarization=0.99,
        image_file: Path = Path("calibration_results.png"),
    ) -> None:
        """
        Args:
            acquisition_file (Path): Raw acquisition data file.
            sample_name (str): Sample / calibrant identifier.
            scan_type (str, optional): Scan geometry (default ``"half-turn"``).
            translation_motor (str, optional): Inner-loop translation motor name (default ``"dty"``).
            rotation_motor (str, optional): Rotation motor name (default ``"rot"``).
            outer_loop_motor (str, optional): Outer-loop motor name (default ``"translation"``).
            beam_size (float, optional): Beam size in metres (default 100 µm).
            beam_energy (float, optional): Beam energy in keV (default 44).
            tth_lims (tuple, optional): ``(low, high)`` 2θ limits in degrees (default ``(None, None)``).
            xy_file (Path, optional): Integrated calibrant pattern (default ``"integrated_data.xy"``).
            param_file (Path, optional): Base name for the calibrated ``.instprm`` output inside
                ``calibration/`` (default ``"calibrated_instrument.instprm"``).
            polarization (float, optional): Beam polarization fraction (default 0.99).
            image_file (Path, optional): Base name for the calibration plot inside ``calibration/``
                (default ``"calibration_results.png"``).
        """
        super().__init__(
            acquisition_file,
            sample_name,
            scan_type,
            translation_motor,
            rotation_motor,
            outer_loop_motor,
            beam_size,
            beam_energy,
            tth_lims,
            xy_file,
            param_file,
            polarization,
        )

        os.makedirs("calibration", exist_ok=True)

        self.xy_file = xy_file
        self.param_file = param_file
        self.low_lim, self.high_lim = tth_lims
        self.tth, self.intensity = read_xy_file(str(self.xy_file))
        self.phases = []

        if self.low_lim == None:
            self.low_lim = self.tth.min()
        if self.high_lim == None:
            self.hih_lim = self.tth.max()

        self.param_file_init = write_starting_instrument_pars(
            polarization=polarization, wavelength=self.wavelength
        )

        self.calibration_file = Path("calibration") / self.param_file
        self.calibration_image = Path("calibration") / image_file

    def refine_instrument_parameters(
        self,
        profile_params: list[str] = ["W", "X", "Y"],
        profile: str = "FCJVoigt",
        n_background_coeff: int = 12,
        background_function: str = "chebyschev",
        use_lebail: bool = True,
    ) -> None:
        """
        Run the recommended instrument-calibration refinement sequence.

        Performs the following steps in order, each building on the converged
        result of the previous one:

        1. **Background** — fit the powder-pattern baseline.
        2. **Scale** — establish the overall intensity scale (or LeBail
           intensities when ``use_lebail=True``).
        3. **Zero shift** — correct the 2θ zero-point offset.
        4. **Peak profile** — refine the width parameters listed in
           ``profile_params`` sequentially, one per refinement cycle.
        5. **Export** — write the calibrated ``.instprm`` file via
           :meth:`write_calibrated_instrument_pars`.

        At every step the refined parameter is left *free* (``freeze=False``)
        so that subsequent steps can absorb any residual correlation.  Only
        after the full sequence is the project saved with all flags in their
        final refined state.

        This method assumes the GSAS-II project has already been created
        (:meth:`create_model` or :meth:`load_model`) and the calibrant phase
        has been added (:meth:`add_phase`) with its cell and atomic positions
        fixed (``block_cell=True``, the default).

        Args:
            profile_params (list of str, optional): Ordered list of instrument peak-profile
                parameters to refine.  Refined sequentially, one per GSAS-II cycle.  Must be valid
                for the chosen ``profile``.  Default ``["W", "X", "Y"]`` is the recommended starting
                set for synchrotron data with a 2-D integrating detector (``U``, ``V``, and ``SH/L``
                are typically fixed at 0/0/0.0001 for such data).
            profile (str, optional): Peak-profile model passed to :meth:`refine_peak_profile`.
                One of ``"FCJVoigt"`` (default), ``"ExpFCJVoigt"``, ``"EpsVoigt"``.
            n_background_coeff (int, optional): Number of background polynomial coefficients (default 12).
            background_function (str, optional): Background function type passed to
                :meth:`refine_background` (default ``"chebyschev"``).
            use_lebail (bool, optional): If ``True`` (default), activate LeBail extraction for the
                calibrant phase before refining the scale, so that the integrated intensities are free
                parameters rather than structure-factor predictions.  Recommended for calibrants whose
                exact structure is well-known and whose scale should not contaminate the profile fit.
        """
        print("\n" + "=" * 60)
        print("INSTRUMENT CALIBRATION REFINEMENT SEQUENCE")
        print("=" * 60)

        # Step 1 — Background
        print("\n--- Step 1: Background ---")
        self.refine_background(
            number_coeff=n_background_coeff,
            function=background_function,
        )

        # Step 2 — Scale (LeBail or Rietveld)
        print("\n--- Step 2: Scale ---")
        if use_lebail:
            self.set_LeBail(enable=True)
            self.refine_histogram_scale()
        else:
            self.refine_histogram_scale()

        # Step 3 — Zero shift
        print("\n--- Step 3: Zero shift ---")
        self.refine_zero_shift(freeze=True)

        # Step 4 — Peak profile
        print("\n--- Step 4: Peak profile ---")
        self.refine_peak_profile(profile=profile, parameters=profile_params)

        # Step 5 — Export
        print("\n--- Step 5: Export ---")
        self.write_calibrated_instrument_pars()
        print("\nCalibration sequence complete.")

    def write_calibrated_instrument_pars(self) -> None:
        """
        Export the refined instrument parameters to a GSAS-II ``.instprm`` file.

        The file is written to ``calibration/<param_file>`` and the calibrated
        values are also printed to stdout for quick verification.
        """
        print("\n" + "=" * 60)
        print("EXPORTING CALIBRATED INSTPRM")
        print("=" * 60)

        ip = self.hist["Instrument Parameters"][0]
        lines = ["#GSAS-II instrument parameter file\n"]

        key_order_single = [
            "Type",
            "Bank",
            "Zero",
            "Polariz.",
            "Azimuth",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "SH/L",
        ]

        lines.append(f"Lam:{self.wavelength}\n")

        for p in key_order_single:
            if p in ip:
                val = ip[p][1] if isinstance(ip[p], list) else ip[p]
                lines.append(f"{p}:{val}\n")

        with open(str(self.calibration_file), "w") as f:
            f.writelines(lines)

        print(f"Calibrated instprm saved to:\n  {str(self.calibration_file)}")
        print("\nFinal calibrated parameters:")
        for line in lines[1:]:  # skip the header comment
            print(f"  {line.rstrip()}")

        print(f"Calibrated instprm saved to:\n  {str(self.calibration_file)}")
        print("\nIn your sample refinements:")
        print("  - Use this file as INST_PARAMS")
        print("  - Fix Zero, W, X, Y (carry from calibration)")
        print(
            "  - U, V, SH/L remain fixed at 0 / 0.0001 for synchrotron and 2D detectors"
        )

    def plot_calibration_results(
        self, show: bool = True, figsize: tuple = (12, 9)
    ) -> None:
        """
        Generate and save a five-panel calibration diagnostic figure.

        The figure is laid out on a 3-row × 2-column grid and saved to
        ``calibration/<image_file>`` (set in :meth:`__init__` via
        ``image_file``).  Each panel is described below.

        **Figure layout:**

        **Top-left — Rietveld fit** (``ax_main``)
            Observed intensities (black dots), calculated profile (red line),
            and fitted background (blue dashed line) plotted against 2θ.
            Vertical tick marks below the pattern show the position of every
            allowed Bragg reflection for each phase; colours cycle through the
            ``colours_ticks`` dict (``"calibrant"`` → magenta,
            ``"Al_holder"`` → dark orange, others → green).  The title shows
            the phase/calibrant name, Rwp (%), and χ² (goodness-of-fit).

        **Bottom-left — Difference plot** (``ax_diff``, shares x-axis)
            Observed minus calculated residuals (I\ :sub:`obs` − I\ :sub:`calc`)
            in intensity units.  A horizontal dashed line marks zero.  A good
            calibration should show a flat, featureless residual with amplitude
            well below the peak heights.  Systematic wiggles indicate
            remaining peak-shape errors; asymmetric residuals around strong
            peaks suggest that ``Zero`` or ``SH/L`` need attention.

        **Top-right — Refined parameter bar chart** (``ax_ip``)
            Bar chart of the four key refined instrument parameters:
            ``Zero``, ``W``, ``X``, ``Y``.  Blue bars are positive values,
            red bars are negative.  Each bar is labelled with its numerical
            value to four decimal places.

            * **Zero** — 2θ zero-point offset (degrees).  A non-zero value
              means the diffractometer's mechanical zero does not coincide
              with the true 2θ = 0°.  Should be small (|Zero| ≲ 0.05°) for
              a well-aligned instrument.
            * **W** — angle-independent Gaussian width coefficient
              (centideg²).  For a synchrotron beam with a 2-D detector this
              is typically the dominant peak-width contribution.
            * **X** — Lorentzian width, 1/cosθ term (centideg).  Related to
              sample crystallite size and instrumental contributions along
              the beam direction.
            * **Y** — Lorentzian width, tanθ term (centideg).  Related to
              microstrain and other angle-dependent broadening.

            U, V, and SH/L are fixed at 0 / 0.0001 for synchrotron data with
            a 2-D detector and are not shown here.

        **Bottom-right — FWHM model** (``ax_fw``)
            Predicted peak FWHM as a function of 2θ, decomposed into its
            Gaussian (orange, driven by ``W``) and Lorentzian (blue, driven
            by ``X`` and ``Y``) components, plus the Thompson-Cox-Hastings
            (TCH) combined total (black).  The TCH pseudo-Voigt combination
            rule is:

                FWHM\ :sub:`total`\ ⁵ = FWHM\ :sub:`G`\ ⁵ + FWHM\ :sub:`L`\ ⁵

            Use this panel to judge whether the peak-width model is
            physically reasonable across the full angular range.  If the
            Lorentzian contribution dominates strongly, consider whether
            sample broadening (size or strain) is contaminating the
            instrumental calibration.

        **Bottom strip — Parameter table** (``ax_text``, spans both columns)
            Tabulated values of all nine instrument parameters reported:
            Lam, Zero, U, V, W, X, Y, SH/L, and Polariz.  Values that were
            fixed during calibration (U = V = 0, SH/L = 0.0001) are included
            for completeness but carry no physical meaning for 2-D detector
            synchrotron data.

        Args:
            show (bool, optional): If ``True``, call ``plt.show()`` after saving (default ``True``).
                Set to ``False`` when running in a batch/headless environment.
            figsize (tuple of (float, float), optional): Figure size in inches as
                ``(width, height)`` (default ``(12, 9)``).

        Note:
            The output image is written to the path set by
            ``self.calibration_image``, which resolves to
            ``calibration/<image_file>`` relative to the working directory.
            The ``calibration/`` directory is created automatically in
            :meth:`__init__`.
        """
        ip = self.hist["Instrument Parameters"][0]
        params_to_report = ["Lam", "Zero", "U", "V", "W", "X", "Y", "SH/L", "Polariz."]
        calibrated = {}
        for p in params_to_report:
            if p in ip:
                val = ip[p][1] if isinstance(ip[p], list) else ip[p]
                calibrated[p] = val

        wR = self.hist.get_wR()
        print("\n" + "=" * 60)
        print("Generating calibration plot")
        print("=" * 60)

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            3, 2, figure=fig, height_ratios=[3, 1, 1.2], hspace=0.08, wspace=0.35
        )

        ax_main = fig.add_subplot(gs[0, 0])
        ax_diff = fig.add_subplot(gs[1, 0], sharex=ax_main)
        ax_ip = fig.add_subplot(gs[0, 1])
        ax_fw = fig.add_subplot(gs[1, 1])
        ax_text = fig.add_subplot(gs[2, :])
        ax_text.axis("off")

        # --- Observed / Calculated / Difference ---
        tth = self.hist.getdata("x")
        yobs = self.hist.getdata("yobs")
        ycalc = self.hist.getdata("ycalc")
        ybkg = self.hist.getdata("background")
        diff = yobs - ycalc

        ax_main.plot(tth, yobs, "k.", ms=2, label="Observed")
        ax_main.plot(tth, ycalc, "r-", lw=1, label="Calculated")
        ax_main.plot(tth, ybkg, "b--", lw=0.8, label="Background")

        # Reflection tick marks for both phases
        colours_ticks = {"calibrant": "magenta", "Al_holder": "darkorange"}
        yrange = yobs.max() - yobs.min()
        tick_y0 = yobs.min() - 0.02 * yrange
        for ph in self.gpx.phases():
            try:
                reflist = self.hist.reflections()[ph.name]["RefList"]
                ref_tth = reflist[:, 5]
                colour = colours_ticks.get(ph.name, "green")
                ax_main.vlines(
                    ref_tth,
                    tick_y0,
                    tick_y0 + 0.04 * yrange,
                    color=colour,
                    lw=0.8,
                    label=f"{ph.name} reflections",
                )
            except Exception:
                pass

        ax_main.set_ylabel("Intensity")
        residuals = self.hist.residuals
        chi2 = residuals.get("GOF")
        stat_parts = []
        if wR is not None:
            stat_parts.append(f"Rwp = {wR:.2f} %")
        if chi2 is not None:
            stat_parts.append(f"χ² = {chi2:.4f}")
        stats_str = "   ".join(stat_parts)
        ax_main.set_title(
            f"{self.calibrant_composition}\n{stats_str}"
            if stats_str
            else "Calibration fit"
        )
        ax_main.legend(fontsize=7, markerscale=2)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        ax_diff.plot(tth, diff, color="forestgreen", lw=0.8)
        ax_diff.axhline(0, color="k", lw=0.5, ls="--")
        ax_diff.set_ylabel("Obs−Calc")
        ax_diff.set_xlabel("2θ (degrees)")

        # --- Bar chart of calibrated parameters ---
        bar_params = ["Zero", "W", "X", "Y"]
        bar_vals = [calibrated.get(p, 0.0) for p in bar_params]
        colours_bar = ["steelblue" if v >= 0 else "tomato" for v in bar_vals]

        bars = ax_ip.bar(bar_params, bar_vals, color=colours_bar, edgecolor="k", lw=0.6)
        ax_ip.axhline(0, color="k", lw=0.5)
        ax_ip.set_title("Refined parameters\n(U, V, SH/L fixed at 0)")
        ax_ip.set_ylabel("Value")
        spread = max(abs(v) for v in bar_vals) if any(bar_vals) else 1
        for bar, val in zip(bars, bar_vals):
            ax_ip.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * spread,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        # --- FWHM vs 2theta ---
        try:
            W_v = calibrated.get("W", 0)
            X_v = calibrated.get("X", 0)
            Y_v = calibrated.get("Y", 0)
            tth_range = np.linspace(self.low_lim, self.high_lim, 300)
            tan_th = np.tan(np.radians(tth_range / 2))
            cos_th = np.cos(np.radians(tth_range / 2))
            fwhm_G = np.sqrt(np.abs(W_v)) * np.ones_like(tth_range)
            fwhm_L = np.abs(X_v) / cos_th + np.abs(Y_v) * tan_th
            fwhm_total = (fwhm_G**5 + fwhm_L**5) ** (1 / 5)
            ax_fw.plot(tth_range, fwhm_G, "darkorange", lw=1.5, label="Gaussian (W)")
            ax_fw.plot(tth_range, fwhm_L, "steelblue", lw=1.5, label="Lorentzian (X+Y)")
            ax_fw.plot(tth_range, fwhm_total, "k-", lw=1.5, label="TCH total")
            ax_fw.set_xlabel("2θ (degrees)")
            ax_fw.set_ylabel("FWHM (degrees)")
            ax_fw.set_title("Peak width model")
            ax_fw.legend(fontsize=7)
            ax_fw.grid(True, alpha=0.3)
        except Exception as e:
            ax_fw.text(
                0.5,
                0.5,
                f"FWHM plot error:\n{e}",
                ha="center",
                va="center",
                transform=ax_fw.transAxes,
                fontsize=8,
            )

        # --- Parameter table ---
        table_params = ["Lam", "Zero", "U", "V", "W", "X", "Y", "SH/L", "Polariz."]
        table_vals = [f"{calibrated.get(p, 0.0):.6f}" for p in table_params]
        tbl = ax_text.table(
            cellText=[table_vals],
            colLabels=table_params,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.8)
        ax_text.set_title(
            "Refined instrument parameter values\n"
            "(U=V=0, SH/L=0.0001 fixed — not meaningful for 2D detector)",
            pad=4,
            fontsize=9,
        )

        plt.suptitle(
            "GSAS-II Instrument Calibration", fontsize=12, fontweight="bold", y=1.01
        )
        plt.savefig(str(self.calibration_image), dpi=150, bbox_inches="tight")
        if show:
            plt.show()
