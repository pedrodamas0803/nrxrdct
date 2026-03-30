"""
Pre-built GSAS-II refinement dictionary templates for common microstructure models.

These dictionaries can be passed directly to GSAS-II phase/histogram objects to
enable specific refinement models without hand-crafting the parameter structure.

Preferred orientation
---------------------
MD_DICT   : March–Dollase model along [1 1 1]
SH_DICT   : Spherical-harmonics model with cylindrical symmetry along [0 0 1]

Crystallite size
----------------
SIZE_ISO_DICT  : Isotropic size model
SIZE_UNI_DICT  : Uniaxial size model along [0 0 1]
SIZE_GEN_DICT  : Generalized (anisotropic) size model

Microstrain (mustrain)
----------------------
MUSTRAIN_ISO_DICT  : Isotropic strain model (1000 µε)
MUSTRAIN_UNI_DICT  : Uniaxial strain model along [0 0 1] (1000 µε)
MUSTRAIN_GEN_DICT  : Generalized (Stephens) strain model
"""
MD_DICT = {
    "Pref.Ori.": {
        "Model": "MD",
        "Axis": [1, 1, 1],
        "Ratio": 1.0,
        "Ref": True,
    }
}
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

SIZE_ISO_DICT = {"Size": {"type": "isotropic", "refine": True, "value": 1.0}}

SIZE_UNI_DICT = {
    "Size": {
        "type": "uniaxial",
        "refine": True,
        "equatorial": 1.0,
        "axial": 1.0,
        "axis": [0, 0, 1],
    }
}

SIZE_GEN_DICT = {"Size": {"type": "generalized", "refine": True}}
#
# Isotropic
MUSTRAIN_ISO_DICT = {"type": "isotropic", "refine": True, "value": 1000.0}
# Uniaxial
MUSTRAIN_UNI_DICT = {
    "type": "uniaxial",
    "refine": True,
    "equatorial": 1000.0,
    "axial": 1000.0,
    "axis": [0, 0, 1],
}
# Generalized (Stephens model)
MUSTRAIN_GEN_DICT = {"type": "generalized", "refine": True}
