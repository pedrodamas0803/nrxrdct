"""
Pre-built GSAS-II refinement dictionary templates for common microstructure models.

These dictionaries are the defaults of
:meth:`~nrxrdct.rietveld.refinement.BaseRefinement.refine_crystallite_size`,
:meth:`~nrxrdct.rietveld.refinement.BaseRefinement.refine_mustrain` and
:meth:`~nrxrdct.rietveld.refinement.BaseRefinement.refine_preferential_orientation`.
Copy and modify one to change starting values, axes or mixing coefficients.

GSAS-II's own ``G2Phase.set_HAP_refinements`` only reads part of these keys
(e.g. it ignores ``"equatorial"``/``"axial"`` and the March–Dollase ratio and
axis), so the ``BaseRefinement`` methods write the model definition into the
project directly.

Preferred orientation
---------------------
MD_DICT   : March–Dollase model along [1 1 1]
SH_DICT   : Spherical-harmonics model of order 4 (cylindrical sample symmetry)

Crystallite size
----------------
SIZE_ISO_DICT  : Isotropic size model
SIZE_UNI_DICT  : Uniaxial size model along [0 0 1]
SIZE_ELL_DICT  : Ellipsoidal (symmetry-constrained tensor) size model
SIZE_GEN_DICT  : Alias of ``SIZE_ELL_DICT`` kept for backward compatibility

Microstrain (mustrain)
----------------------
MUSTRAIN_ISO_DICT  : Isotropic strain model
MUSTRAIN_UNI_DICT  : Uniaxial strain model along [0 0 1]
MUSTRAIN_GEN_DICT  : Generalized (Stephens) strain model

Recognised keys
---------------
Size / Mustrain:
    ``type``        model name (see above)
    ``refine``      refine the model parameters (bool)
    ``value``       starting isotropic value (µm for Size, µε for Mustrain);
                    also used to initialise the uniaxial, ellipsoidal and
                    generalized parameters
    ``equatorial``  uniaxial value perpendicular to the axis
    ``axial``       uniaxial value along the axis
    ``axis``        unique axis ``[h, k, l]`` (``direction`` is accepted too)
    ``LGmix``       Lorentzian fraction, a float or ``{"value": .., "refine": ..}``
    ``terms``       ellipsoidal Size only: which of ``S11`` … ``S23`` to refine
                    (default the diagonal ``S11``, ``S22``, ``S33``; GSAS-II
                    does not constrain the ellipsoid by symmetry)

Pref.Ori.:
    ``Model``  ``"MD"`` or ``"SH"``
    ``Ref``    refine the ratio (MD) or the coefficients (SH)
    ``Ratio``  March–Dollase ratio *r* (MD only)
    ``Axis``   March–Dollase axis ``[h, k, l]`` (MD only)
    ``SHord``  even spherical-harmonics order (SH only)
    ``SHcoef`` starting coefficients keyed by GSAS-II name (SH only, optional)
"""
# The templates deliberately carry no starting values: the current value in
# the project is kept, so calling a refine_* method again continues from the
# refined state. When a model is switched (e.g. isotropic -> uniaxial), the
# new parameters are initialised from the current isotropic value.
MD_DICT = {
    "Pref.Ori.": {
        "Model": "MD",
        "Axis": [1, 1, 1],
        "Ref": True,
    }
}
SH_DICT = {
    "Pref.Ori.": {
        "Model": "SH",
        "SHord": 4,
        "Ref": True,
    }
}

SIZE_ISO_DICT = {"Size": {"type": "isotropic", "refine": True}}

SIZE_UNI_DICT = {
    "Size": {
        "type": "uniaxial",
        "refine": True,
        "axis": [0, 0, 1],
    }
}

SIZE_ELL_DICT = {"Size": {"type": "ellipsoidal", "refine": True}}
SIZE_GEN_DICT = SIZE_ELL_DICT
#
# Isotropic
MUSTRAIN_ISO_DICT = {"type": "isotropic", "refine": True}
# Uniaxial
MUSTRAIN_UNI_DICT = {
    "type": "uniaxial",
    "refine": True,
    "axis": [0, 0, 1],
}
# Generalized (Stephens model)
MUSTRAIN_GEN_DICT = {"type": "generalized", "refine": True}
