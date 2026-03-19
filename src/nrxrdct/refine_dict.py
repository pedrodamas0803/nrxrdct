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
