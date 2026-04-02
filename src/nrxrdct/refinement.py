"""
GSAS-II Rietveld refinement wrappers for XRD-CT data.

Provides :class:`BaseRefinement`, a :class:`~nrxrdct.parameters.Scan` subclass
that drives sequential GSAS-II refinement steps (background, scale, zero shift,
peak broadening, cell, preferred orientation, crystallite size, microstrain,
extinction), and :class:`InstrumentCalibration`, a specialised subclass for
calibrant-based instrument parameter calibration with dedicated plotting.
"""
import os
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
        Parameters
        ----------
        acquisition_file : Path
            Raw acquisition data file (passed to :class:`~nrxrdct.parameters.Scan`).
        sample_name : str
            Sample identifier used in output file names.
        scan_type : str, optional
            Scan geometry (default ``"half-turn"``).
        translation_motor : str, optional
            Inner-loop translation motor name (default ``"dty"``).
        rotation_motor : str, optional
            Rotation motor name (default ``"rot"``).
        outer_loop_motor : str, optional
            Outer-loop motor name (default ``"translation"``).
        beam_size : float, optional
            Beam size in metres (default 100 µm).
        beam_energy : float, optional
            Beam energy in keV (default 44).
        tth_lims : tuple of (float or None, float or None), optional
            ``(low, high)`` 2θ limits in degrees.  ``None`` means use the
            pattern extent (default ``(None, None)``).
        xy_file : Path, optional
            Integrated pattern to fit (default ``"integrated_data.xy"``).
        param_file : Path, optional
            Calibrated ``.instprm`` file (default ``"calibrated_instrument.instprm"``).
        polarization : float, optional
            Beam polarization fraction used when writing the starting instrument
            parameter file (default 0.99).
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

    def create_model(self, gpx_file: Path = Path("model.gpx")) -> tuple:
        """
        Create a new GSAS-II project and add the powder histogram.

        Parameters
        ----------
        gpx_file : Path, optional
            Output ``.gpx`` file path (default ``"model.gpx"``).

        Returns
        -------
        tuple
            ``(gpx, hist)`` — the :class:`G2Project` and the added histogram object.
        """
        self.gpx = G2sc.G2Project(newgpx=str(gpx_file))

        self.hist = self.gpx.add_powder_histogram(
            datafile=self.xy_file, iparams=self.param_file_init, phases="all"
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

        Parameters
        ----------
        cif_file : Path, optional
            Path to the CIF file (default ``"cif_file"``).
        phase_name : str, optional
            Name to assign the phase in GSAS-II (default ``"LaB6"``).
        block_cell : bool, optional
            If ``True``, fixes all atom positions and the unit cell so only
            instrumental parameters are refined in subsequent steps (default ``True``).

        Returns
        -------
        G2Phase
            The newly added GSAS-II phase object.
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

    def refine_background(
        self,
        number_coeff: int = 12,
        do_refine: bool = True,
        function: str = "chebyschev",
        debye_terms: list | None = None,
    ) -> None:
        """
        Refine the powder pattern background.

        Parameters
        ----------
        number_coeff : int, optional
            Number of background function coefficients (default 12).
        do_refine : bool, optional
            Whether to activate the background refinement flag (default ``True``).
        function : str, optional
            Background function type.  Must be one of:

            * ``"chebyschev"``        — Chebyshev polynomial (default)
            * ``"chebyschev-1"``      — Chebyshev polynomial (1st kind)
            * ``"cosine"``            — Cosine series
            * ``"Q^2 power series"``  — Q² power series
            * ``"Q^-2 power series"`` — Q⁻² power series
            * ``"lin interpolate"``   — Linear interpolation
            * ``"inv interpolate"``   — Inverse interpolation
            * ``"log interpolate"``   — Logarithmic interpolation
        debye_terms : list of dict, optional
            Debye–scattering components for amorphous content.  Each entry is a
            dict with the following keys (all optional; unset keys fall back to
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
        print(f"Background refinement performed (function='{function}'"
              + (f", {len(debye_terms)} Debye term(s)" if debye_terms else "")
              + ")")

    def refine_histogram_scale(self) -> None:
        """Refine the overall histogram scale factor, then freeze it."""
        self.hist.SampleParameters["Scale"][1] = True
        self.gpx.save()
        self.gpx.do_refinements([{}])
        self.hist.SampleParameters["Scale"][1] = False
        self.gpx.save()
        print("Histogram scale refinement done (parameter frozen)")

    def refine_phase_scale(self, phase: str | list[str]) -> None:
        """
        Refine the HAP scale factor for one or more phases.

        Parameters
        ----------
        phase : str or list of str
            Phase name, or list of phase names, whose HAP scale factor should
            be refined.  Names must match those used in :meth:`add_phase`.
        """
        names = [phase] if isinstance(phase, str) else list(phase)
        available = {ph.name: ph for ph in self.gpx.phases()}
        for name in names:
            if name not in available:
                raise ValueError(
                    f"Phase '{name}' not found. "
                    f"Available phases: {list(available)}"
                )
            available[name].set_HAP_refinements({"Scale": True}, histograms=[self.hist])
            self.gpx.save()
            self.gpx.do_refinements([{}])
            available[name].set_HAP_refinements({"Scale": False}, histograms=[self.hist])
            self.gpx.save()
            print(f"Phase scale refinement done for '{name}' (parameter frozen)")

    def refine_zero_shift(self) -> None:
        """Refine the 2θ zero-shift instrument parameter, then freeze it."""
        self.hist.set_refinements({"Instrument Parameters": ["Zero"]})
        self.gpx.save()
        self.gpx.do_refinements([{}])
        self.hist["Instrument Parameters"][0]["Zero"][2] = False
        self.gpx.save()
        print("Zero shift refinement done (parameter frozen)")

    def refine_sample_displacement(self, parameter: str = "Shift") -> None:
        """
        Refine a sample displacement parameter, then freeze it.

        Parameters
        ----------
        parameter : str, optional
            Sample parameter to refine.  Choose according to the geometry:

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
        self.hist.clear_refinements({"Sample Parameters": [parameter]})
        self.gpx.save()
        print(f"Sample displacement refinement done for '{parameter}' (parameter frozen)")

    def refine_gaussian_broadening(self, refine: list = ["U", "V", "W", "SH/L"]) -> None:
        """
        Possible parameters are U, V, W, Z, SH/L
        For synchrotron light with 2D detectors, go only for W.
        """
        for param in refine:
            self.hist.set_refinements({"Instrument Parameters": [param]})
            self.gpx.save()
            self.gpx.do_refinements([{}])
            print(f"Refined {param} (Gaussian broadening)")

    def refine_lorentzian_broadening(self, refine: list = ["X", "Y"]) -> None:
        """
        Possible parameters are X, and Y
        """
        for param in refine:
            self.hist.set_refinements({"Instrument Parameters": [param]})
            self.gpx.save()
            self.gpx.do_refinements([{}])
            print(f"Refined {param} (Lorentzian broadening)")

    def refine_peak_profile(
        self,
        profile: str = "FCJVoigt",
        parameters: list[str] | None = None,
    ) -> None:
        """
        Refine instrument peak-profile parameters for the chosen profile model.

        Profile models
        --------------
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

        Parameters
        ----------
        profile : ``"FCJVoigt"`` | ``"ExpFCJVoigt"`` | ``"EpsVoigt"``, optional
            Peak profile model to activate and refine (default
            ``"FCJVoigt"``).
        parameters : list of str, optional
            Instrument parameters to refine, refined sequentially.
            Must be valid for the chosen ``profile`` — see table above.
            If ``None`` (default), a sensible starting set is used:

            * FCJVoigt   → ``["W", "X", "Y"]``
            * ExpFCJVoigt → ``["W", "alpha-0", "alpha-1", "beta-0", "beta-1"]``
            * EpsVoigt   → ``["W", "alpha-0", "alpha-1", "beta-0", "beta-1"]``
        """
        _TYPE_CHAR = {"FCJVoigt": "C", "ExpFCJVoigt": "A", "EpsVoigt": "B"}
        _VALID_PARAMS = {
            "FCJVoigt":    {"U", "V", "W", "X", "Y", "Z", "SH/L"},
            "ExpFCJVoigt": {"U", "V", "W", "X", "Y", "Z", "SH/L",
                            "alpha-0", "alpha-1", "beta-0", "beta-1"},
            "EpsVoigt":    {"U", "V", "W", "X", "Y", "Z",
                            "alpha-0", "alpha-1", "beta-0", "beta-1"},
        }
        _DEFAULT_PARAMS = {
            "FCJVoigt":    ["W", "X", "Y"],
            "ExpFCJVoigt": ["W", "alpha-0", "alpha-1", "beta-0", "beta-1"],
            "EpsVoigt":    ["W", "alpha-0", "alpha-1", "beta-0", "beta-1"],
        }
        _EXTRA_DEFAULTS = {
            "alpha-0": [0.1, 0.1, False],
            "alpha-1": [0.0, 0.0, False],
            "beta-0":  [0.1, 0.1, False],
            "beta-1":  [0.0, 0.0, False],
        }

        if profile not in _TYPE_CHAR:
            raise ValueError(
                f"Unknown profile '{profile}'. "
                f"Valid options: {list(_TYPE_CHAR)}"
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
        current = ip["Type"][0]          # e.g. 'PXC'
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

    def free_and_refine_cell(self) -> None:
        """
        Frees and performs cell refinement. Tends to converge fast for higher symmetries provided that the spacegroup is correctly given.
        """

        self.phase.set_refinements({"Cell": True})
        self.gpx.save()
        self.gpx.do_refinements([{}])

    def refine_preferential_orientation(
        self,
        model: str = "MD",
        phase: str | list[str] | None = None,
        parsMD: dict = MD_DICT,
        parsSH: dict = SH_DICT,
    ) -> None:
        """
        Refine preferred orientation (texture) for one or more phases.

        Parameters
        ----------
        model : ``"MD"`` | ``"SH"``, optional
            Texture model (default ``"MD"``).  See model descriptions below.
        phase : str, list of str, or None, optional
            Phase name(s) to apply the refinement to.  ``None`` (default)
            applies to all phases in the project.
        parsMD : dict, optional
            Full parameter dictionary for the March-Dollase model.
            Default is ``MD_DICT`` from ``refine_dict``.  You must supply
            the complete dict when overriding any value — partial updates
            are not merged automatically.
        parsSH : dict, optional
            Full parameter dictionary for the spherical-harmonics model.
            Default is ``SH_DICT`` from ``refine_dict``.  Same caveat as
            ``parsMD``.

        Models
        ------
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

        Parameters
        ----------
        flags : list of str, optional
            Sequence of GSAS-II atom-refinement flag strings applied in
            order.  Each string is any combination of the tokens below;
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

            .. note::
                Anisotropic displacement parameters (ADP / U\ :sub:`ij`
                tensor) are **not** controlled through this flag string.
                They require changing the atom's displacement model from
                isotropic (``'I'``) to anisotropic (``'A'``) via the
                ``cia`` field directly in the GSAS-II project, and are
                only justified for single-crystal-quality powder data.

        phase : str, list of str, or None, optional
            Phase name(s) to apply the refinement to.  ``None`` (default)
            applies to all phases in the project.  Names must match those
            passed to :meth:`add_phase`.

        atoms : list of str, or None, optional
            Restrict refinement to a subset of atoms.  Each entry is
            matched against atom **labels** (e.g. ``"Fe1"``, ``"O2"``)
            first, then against **element symbols** (e.g. ``"Fe"``,
            ``"O"``).  If an entry matches neither in a given phase it is
            silently skipped for that phase.  ``None`` (default) refines
            all atoms.
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
            print(f"Atomic refinement flag '{flag}' applied to {phase_names}{atom_info}")

    def refine_occupancy(
        self,
        phase: str | list[str] | None = None,
        atoms: list[str] | None = None,
    ) -> None:
        """
        Refine site occupancy for selected atoms, then freeze the occupancy flag.

        The ``"F"`` refinement flag is added to the current flags of every
        target atom before the refinement cycle and removed afterwards, so
        that positional and displacement flags that were already active are
        left untouched.

        Parameters
        ----------
        phase : str, list of str, or None, optional
            Phase name(s) to apply the refinement to.  ``None`` (default)
            applies to all phases in the project.  Names must match those
            used in :meth:`add_phase`.
        atoms : list of str, or None, optional
            Restrict refinement to a subset of atoms.  Each entry is matched
            against atom **labels** (e.g. ``"Fe1"``) first, then against
            **element symbols** (e.g. ``"Fe"``).  ``None`` (default) refines
            the occupancy of all atoms in the target phases.

        Notes
        -----
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

        # Freeze: restore flags without "F"
        for atom in target_atoms:
            atom.refinement_flags = original_flags[id(atom)]
        self.gpx.save()

        phase_names = [ph.name for ph in targets]
        atom_info = f", atoms={atoms}" if atoms is not None else ""
        print(f"Occupancy refinement done for {phase_names}{atom_info} (flag frozen)")

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

        Parameters
        ----------
        refine_type : ``"isotropic"`` | ``"uniaxial"`` | ``"generalized"``, optional
            Size broadening model (default ``"isotropic"``).  Ignored when
            ``refine_dict`` is supplied.
        refine_dict : dict or None, optional
            Custom parameter dictionary following the same structure as the
            predefined ``SIZE_*_DICT`` constants (i.e. the top-level key must
            be ``"Size"``).  When provided, ``refine_type`` is ignored.
        phase : str, list of str, or None, optional
            Phase name(s) to refine.  ``None`` (default) refines all phases.

        Models
        ------
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

        Notes
        -----
        Crystallite size broadening and microstrain broadening both contribute
        to Lorentzian peak widths and are strongly correlated.  Refine them
        sequentially — size first if the phase is nanocrystalline, mustrain
        first if the sample has significant lattice distortions — or use
        Williamson-Hall analysis to decide which dominates before starting the
        Rietveld refinement.
        """
        if isinstance(refine_dict, dict):
            refine_type = "personalized"

        if refine_type.lower() not in ("isotropic", "uniaxial", "generalized", "personalized"):
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

        Parameters
        ----------
        refine_type : ``"isotropic"`` | ``"uniaxial"`` | ``"generalized"``, optional
            Microstrain model (default ``"isotropic"``).  Ignored when
            ``refine_dict`` is supplied.
        refine_dict : dict or None, optional
            Custom parameter dictionary following the same structure as the
            predefined ``MUSTRAIN_*_DICT`` constants (i.e. no top-level
            ``"Mustrain"`` key — the inner dict is passed directly).  When
            provided, ``refine_type`` is ignored.
        phase : str, list of str, or None, optional
            Phase name(s) to refine.  ``None`` (default) refines all phases.

        Models
        ------
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

        Notes
        -----
        Microstrain and crystallite-size broadening are highly correlated
        because both affect the Lorentzian component of the peak profile.  A
        Williamson-Hall plot (βcosθ vs sinθ) can help determine which
        contribution dominates before committing to a model.  Avoid refining
        both simultaneously unless the data quality and angular range clearly
        support it.
        """
        if isinstance(refine_dict, dict):
            refine_type = "personalized"

        if refine_type.lower() not in ("isotropic", "uniaxial", "generalized", "personalized"):
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

        Parameters
        ----------
        phase : str, list of str, or None, optional
            Phase name(s) to refine.  ``None`` (default) refines all phases.

        Notes
        -----
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

    def refine_extinction(self) -> None:
        """Refine the extinction parameter for all phases."""
        for ph in self.gpx.phases():
            ph.set_HAP_refinements({"Extinction": True}, histograms=[self.hist])
            self.gpx.save()
            self.gpx.do_refinements([{}])

    def print_refinement_results(self) -> None:
        """Print a summary of calibrated instrument parameters and per-phase cell results to stdout."""
        print("\n" + "=" * 60)
        print("REFINEMENT RESULTS")
        print("=" * 60)

        ip = self.hist["Instrument Parameters"][0]
        params_to_report = ["Lam", "Zero", "U", "V", "W", "X", "Y", "SH/L", "Polariz."]
        calibrated = {}
        for p in params_to_report:
            if p in ip:
                val = ip[p][1] if isinstance(ip[p], list) else ip[p]
                calibrated[p] = val

        print("\nCalibrated instrument parameters:")
        for p, v in calibrated.items():
            fixed = "" if (isinstance(ip[p], list) and ip[p][2]) else " (fixed)"
            print(f"  {p:12s} = {v:.6f}{fixed}")

        wR = self.hist.get_wR()
        print(
            f"\n  Rwp = {wR:.2f} %" if wR is not None else "\n  Rwp = refinement failed"
        )
        print("\nPhase results:")
        for ph in self.gpx.phases():
            print(f"\n  Phase: {ph.name}")
            cell = ph.get_cell()
            for k, v in cell.items():
                try:
                    print(f"    {k} = {v:.5f}")
                except Exception:
                    print(f"    {k} = {v}")

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

        # Histogram scale
        self.hist.SampleParameters["Scale"][1] = False

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
        parm_dict = cov_data.get("parmDict", {})
        cov_matrix = cov_data.get("covMatrix")

        if not vary_list:
            print("No refined variables found (run a refinement first).")
            return

        sigmas = (
            np.sqrt(np.diag(cov_matrix))
            if cov_matrix is not None and len(cov_matrix)
            else [None] * len(vary_list)
        )

        print("\n" + "=" * 60)
        print("REFINED VARIABLES")
        print("=" * 60)
        print(f"  {'Parameter':<40} {'Value':>14} {'Esd':>14}")
        print("  " + "-" * 68)
        for var, sig in zip(vary_list, sigmas):
            val = parm_dict.get(var, float("nan"))
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

    def print_atoms(self, phase: str | list[str] | None = None) -> None:
        """
        Print a table of all atoms in one or more phases with their current
        refinement state.

        Parameters
        ----------
        phase : str, list of str, or None, optional
            Phase name(s) to inspect.  ``None`` (default) prints all phases
            in the project.  Names must match those used in :meth:`add_phase`.
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

    def print_phases(self) -> None:
        """
        Print a summary of every phase in the project, including cell
        parameters, space group, composition, and the current refinement
        state of all phase-level and HAP parameters.
        """
        for ph in self.gpx.phases():
            gen = ph.data["General"]
            cell = gen["Cell"]           # [refine, a, b, c, α, β, γ, V]
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
            print(f"    a={c['length_a']:.5f}  b={c['length_b']:.5f}  c={c['length_c']:.5f}  Å")
            print(f"    α={c['angle_alpha']:.4f}  β={c['angle_beta']:.4f}  γ={c['angle_gamma']:.4f}  °")
            print(f"    V={c['volume']:.3f}  Å³")

            hap = ph.data["Histograms"].get(self.hist.name, {})
            if hap:
                print(f"\n  HAP refinement state ({self.hist.name}):")
                for key in ("Scale", "Extinction", "HStrain", "Size",
                            "Mustrain", "Pref.Ori."):
                    if key not in hap:
                        continue
                    val = hap[key]
                    if key == "Scale":
                        print(f"    Scale      : {val[0]:.6g}  (refine={val[1]})")
                    elif key == "Extinction":
                        print(f"    Extinction : {val[0]:.6g}  (refine={val[1]})")
                    elif key == "HStrain":
                        flags = val[1]
                        print(f"    HStrain    : refine={flags}")
                    elif key in ("Size", "Mustrain"):
                        model = val[0]
                        refine = val[2]
                        print(f"    {key:<10} : model={model!r}  refine={refine}")
                    elif key == "Pref.Ori.":
                        model = val[2]
                        refine = bool(val[6]) if len(val) > 6 else val[2]
                        print(f"    Pref.Ori.  : model={model!r}  refine={val[6] if len(val) > 6 else '?'}")

    def plot_results(
        self, image_path: Path = "calibration_plot.png", show: bool = True
    ) -> None:
        """
        Plot the Rietveld fit (observed / calculated / difference) and save to disk.

        Parameters
        ----------
        image_path : Path, optional
            Output image file (default ``"calibration_plot.png"``).
        show : bool, optional
            If ``True``, call ``plt.show()`` after saving (default ``True``).
        """
        wR = self.hist.get_wR()
        print("\n" + "=" * 60)
        print("Generating calibration plot")
        print("=" * 60)

        fig = plt.figure(figsize=(15, 10))
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
        ax_main.set_title(
            f"{self.calibrant_composition}\nRwp = {wR:.2f} %"
            if wR
            else "Calibration fit"
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
        Parameters
        ----------
        acquisition_file : Path
            Raw acquisition data file.
        sample_name : str
            Sample / calibrant identifier.
        scan_type : str, optional
            Scan geometry (default ``"half-turn"``).
        translation_motor : str, optional
            Inner-loop translation motor name (default ``"dty"``).
        rotation_motor : str, optional
            Rotation motor name (default ``"rot"``).
        outer_loop_motor : str, optional
            Outer-loop motor name (default ``"translation"``).
        beam_size : float, optional
            Beam size in metres (default 100 µm).
        beam_energy : float, optional
            Beam energy in keV (default 44).
        tth_lims : tuple, optional
            ``(low, high)`` 2θ limits in degrees (default ``(None, None)``).
        xy_file : Path, optional
            Integrated calibrant pattern (default ``"integrated_data.xy"``).
        param_file : Path, optional
            Base name for the calibrated ``.instprm`` output inside ``calibration/``
            (default ``"calibrated_instrument.instprm"``).
        polarization : float, optional
            Beam polarization fraction (default 0.99).
        image_file : Path, optional
            Base name for the calibration plot inside ``calibration/``
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

    def print_refinement_results(self) -> None:
        print("\n" + "=" * 60)
        print("REFINEMENT RESULTS")
        print("=" * 60)

        ip = self.hist["Instrument Parameters"][0]
        params_to_report = ["Lam", "Zero", "U", "V", "W", "X", "Y", "SH/L", "Polariz."]
        calibrated = {}
        for p in params_to_report:
            if p in ip:
                val = ip[p][1] if isinstance(ip[p], list) else ip[p]
                calibrated[p] = val

        print("\nCalibrated instrument parameters:")
        for p, v in calibrated.items():
            fixed = "" if (isinstance(ip[p], list) and ip[p][2]) else " (fixed)"
            print(f"  {p:12s} = {v:.6f}{fixed}")

        wR = self.hist.get_wR()
        print(
            f"\n  Rwp = {wR:.2f} %" if wR is not None else "\n  Rwp = refinement failed"
        )
        print("\nPhase results:")
        for ph in self.gpx.phases():
            print(f"\n  Phase: {ph.name}")
            cell = ph.get_cell()
            for k, v in cell.items():
                try:
                    print(f"    {k} = {v:.5f}")
                except Exception:
                    print(f"    {k} = {v}")

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

    def plot_calibration_results(self, show: bool = True) -> None:
        """
        Generate and save a multi-panel calibration diagnostic figure.

        The figure contains: the Rietveld fit with reflection markers; an
        observed-minus-calculated difference plot; a bar chart of the key
        refined parameters; a FWHM-vs-2θ model plot; and a parameter table.
        The image is saved to ``calibration/<image_file>``.

        Parameters
        ----------
        show : bool, optional
            If ``True``, call ``plt.show()`` after saving (default ``True``).
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

        fig = plt.figure(figsize=(15, 10))
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
        ax_main.set_title(
            f"{self.calibrant_composition}\nRwp = {wR:.2f} %"
            if wR
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
