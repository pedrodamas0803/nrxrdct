"""
nrxrdct.laue.compare — statistical comparison of two GrainMap reconstructions.

Two independent micro-Laue scans (different samples, different scan grids,
possibly different shapes) cannot be compared pixel-by-pixel or grain-slot
by grain-slot.  This module instead reduces each map to one scalar per
*physical grain* — using only quantities that are invariant to the
per-grain crystal-frame convention — and compares the two grain
populations with standard two-sample statistics, following the way EBSD
tools (e.g. MTEX) compare independent samples through scalar/distributional
summaries rather than registered maps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["compare_grain_populations"]


def _grain_ids_and_masks(gmap, grain, min_pixels: int) -> dict:
    """Map each sufficiently large physical grain to a boolean pixel mask."""
    if grain == "merged":
        if gmap.best_grain_map is None:
            raise ValueError(
                "GrainMap has no merge result — call apply_merge() first, "
                "or pass an explicit grain index via grain_a/grain_b."
            )
        bgm = gmap.best_grain_map
        ids = sorted(int(g) for g in np.unique(bgm) if g >= 0)
        masks = {g: bgm == g for g in ids}
    else:
        valid = ~np.any(np.isnan(gmap.U[grain]), axis=(-2, -1))
        masks = {grain: valid}
    return {g: m for g, m in masks.items() if m.sum() >= min_pixels}


def _selected_field(gmap, arr: np.ndarray, grain) -> np.ndarray:
    """Per-pixel array for the requested grain slot ('merged' or an int)."""
    return gmap._select_merged(arr) if grain == "merged" else arr[grain]


def _per_grain_medians(field_2d: np.ndarray, masks: dict) -> dict:
    """One median value per grain, dropping grains with no finite pixels."""
    out = {}
    for g, mask in masks.items():
        vals = field_2d[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[g] = float(np.median(vals))
    return out


def _principal_strains(gmap, grain) -> np.ndarray:
    """Descending-sorted eigenvalues of the deviatoric strain tensor, (ny, nx, 3).

    Frame-invariant, unlike the raw tensor components: ``strain_tensor_deviatoric``
    is expressed in each grain's own crystal frame, so e.g. ``e_xx`` is not
    comparable across grains (let alone across samples), while the eigenvalues
    of the tensor are.
    """
    eps = _selected_field(gmap, gmap.strain_tensor_deviatoric, grain)
    return np.linalg.eigvalsh(eps)[..., ::-1]  # eigvalsh is ascending


def _grain_mean_orientations(gmap, grain, symmetry: str) -> dict:
    """Mean orientation matrix (3, 3), keyed by physical grain id."""
    ori, iy, ix = gmap.get_orientations(grain=grain, symmetry=symmetry)
    grain_id = (
        gmap.best_grain_map[iy, ix] if grain == "merged"
        else np.full(len(iy), grain)
    )
    means = {}
    for g in np.unique(grain_id):
        mask = grain_id == g
        if mask.any():
            means[int(g)] = gmap._mean_rotation(ori[mask].to_matrix())
    return means


def _orientation_spread_deg(gmap, grain, symmetry: str, masks: dict) -> dict:
    """Per-grain misorientation (deg) between each grain's mean orientation and
    the sample-wide mean orientation.

    A texture-spread scalar analogous to grain orientation spread (GOS), but
    applied across grains within a sample instead of across pixels within a
    grain — i.e. "how scattered is this sample's set of grain orientations".

    *masks* (as returned by :func:`_grain_ids_and_masks`) restricts the
    result to grains that passed the ``min_pixels`` cut.
    """
    from orix.quaternion import Orientation

    means = _grain_mean_orientations(gmap, grain, symmetry)
    means = {g: U for g, U in means.items() if g in masks}
    if not means:
        return {}

    sym = gmap._orix_symmetry(symmetry)
    global_mean = gmap._mean_rotation(np.stack(list(means.values())))
    global_ori = Orientation.from_matrix(global_mean[None], symmetry=sym)

    spread = {}
    for g, U in means.items():
        o = Orientation.from_matrix(U[None], symmetry=sym)
        mori = (~o).outer(global_ori)
        mori.symmetry = sym
        mori = mori.reduce()
        spread[g] = float(np.degrees(mori.angle[0, 0]))
    return spread


def _two_sample_stats(a, b, *, n_bootstrap: int, rng) -> dict:
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    out = {"n_a": a.size, "n_b": b.size}

    if a.size < 2 or b.size < 2:
        out.update(
            median_a=float(np.median(a)) if a.size else np.nan,
            median_b=float(np.median(b)) if b.size else np.nan,
            median_diff=np.nan, ci_low=np.nan, ci_high=np.nan,
            mannwhitney_p=np.nan, ks_stat=np.nan, ks_p=np.nan, levene_p=np.nan,
        )
        return out

    out["median_a"] = float(np.median(a))
    out["median_b"] = float(np.median(b))
    out["median_diff"] = out["median_a"] - out["median_b"]

    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        boot[i] = (
            np.median(rng.choice(a, size=a.size, replace=True))
            - np.median(rng.choice(b, size=b.size, replace=True))
        )
    out["ci_low"], out["ci_high"] = (float(v) for v in np.percentile(boot, [2.5, 97.5]))

    out["mannwhitney_p"] = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    ks = stats.ks_2samp(a, b)
    out["ks_stat"], out["ks_p"] = float(ks.statistic), float(ks.pvalue)
    out["levene_p"] = float(stats.levene(a, b, center="median").pvalue)
    return out


def compare_grain_populations(
    map_a,
    map_b,
    *,
    grain_a: "int | str" = "merged",
    grain_b: "int | str" = "merged",
    symmetry: str = "cubic",
    label_a: str = "A",
    label_b: str = "B",
    min_pixels: int = 5,
    n_bootstrap: int = 2000,
    random_state: "int | None" = None,
) -> pd.DataFrame:
    """
    Compare two :class:`~nrxrdct.laue.map.GrainMap` reconstructions from
    independent scans (e.g. different samples), without assuming any
    pixel-to-pixel or grid correspondence between them.

    Each map is reduced to one value per physical grain (the median over
    that grain's pixels), using only quantities that are invariant to the
    arbitrary per-grain crystal-frame convention:

    - ``rms_px``, ``match_rate`` — fit-quality sanity check.  A large gap
      here means an apparent physical difference could just be a fitting
      artifact rather than a real difference between the samples.
    - ``orientation_spread_deg`` — see :func:`_orientation_spread_deg`;
      a texture-spread scalar, analogous to grain orientation spread (GOS)
      but computed across a sample's grains rather than across one grain's
      pixels.
    - ``equivalent_strain``, ``max_principal_strain``, ``min_principal_strain``,
      ``max_shear_strain`` — derived from the eigenvalues of
      ``strain_tensor_deviatoric``.  Only the deviatoric part is used
      because white-beam Laue cannot resolve the hydrostatic/volumetric
      strain (that needs an absolute lattice parameter from an
      energy-resolved measurement); the eigenvalues are frame-invariant,
      unlike the raw tensor components which are expressed per-grain in
      each grain's own crystal frame.

    For each quantity, the two samples' per-grain distributions are compared
    with:

    - **Mann-Whitney U** (``mannwhitney_p``) — shift in median.
    - **two-sample KS** (``ks_stat``, ``ks_p``) — any distribution-shape
      difference.
    - **Levene, median-centred / Brown-Forsythe** (``levene_p``) — difference
      in spread (heterogeneity), independent of any shift in the median.
    - a bootstrap confidence interval on the median difference
      (``ci_low``, ``ci_high``), resampling grains with replacement, as an
      effect-size estimate — useful since grain counts are often small
      enough that a p-value alone is not very informative.

    Grains are aggregated to the physical-grain level (one value per grain,
    not per pixel) before any test is run: neighbouring pixels within a
    grain are not independent samples, so testing on raw per-pixel arrays
    would understate the true p-values.

    Args:
        map_a, map_b (GrainMap): The two reconstructions to compare.
        grain_a, grain_b (int or 'merged'): Grain slot to use in each map.
            ``'merged'`` (default) uses every physical grain found in
            :attr:`GrainMap.best_grain_map` (requires :meth:`GrainMap.apply_merge`
            to have been called on that map).
        symmetry (str): Crystal point-group symmetry passed to orix — one of
            ``'cubic'``, ``'hexagonal'``, ``'tetragonal'``, ``'orthorhombic'``.
            Assumes both samples share the same crystal structure.
        label_a, label_b (str): Sample labels, stored in the returned
            DataFrame's ``attrs`` for readability.
        min_pixels (int): Drop grains with fewer than this many valid pixels
            before aggregating, to avoid spurious/tiny fits skewing the
            per-grain distributions.  Default ``5``.
        n_bootstrap (int): Bootstrap resamples for the median-difference CI.
            Default ``2000``.
        random_state (int or None): Seed for the bootstrap RNG.

    Returns:
        pandas.DataFrame: One row per quantity (index name ``quantity``),
        with columns ``n_a, n_b, median_a, median_b, median_diff, ci_low,
        ci_high, mannwhitney_p, ks_stat, ks_p, levene_p``.

    Example::

        gmap_a.apply_merge(*gmap_a.merge(min_match_rate=0.3))
        gmap_b.apply_merge(*gmap_b.merge(min_match_rate=0.3))

        df = compare_grain_populations(
            gmap_a, gmap_b, symmetry='cubic',
            label_a='as-grown', label_b='annealed',
        )
        print(df)
"""
    rng = np.random.default_rng(random_state)

    masks_a = _grain_ids_and_masks(map_a, grain_a, min_pixels)
    masks_b = _grain_ids_and_masks(map_b, grain_b, min_pixels)

    eig_a = _principal_strains(map_a, grain_a)
    eig_b = _principal_strains(map_b, grain_b)

    quantities = {
        "rms_px": (
            _per_grain_medians(_selected_field(map_a, map_a.rms_px, grain_a), masks_a),
            _per_grain_medians(_selected_field(map_b, map_b.rms_px, grain_b), masks_b),
        ),
        "match_rate": (
            _per_grain_medians(_selected_field(map_a, map_a.match_rate, grain_a), masks_a),
            _per_grain_medians(_selected_field(map_b, map_b.match_rate, grain_b), masks_b),
        ),
        "orientation_spread_deg": (
            _orientation_spread_deg(map_a, grain_a, symmetry, masks_a),
            _orientation_spread_deg(map_b, grain_b, symmetry, masks_b),
        ),
        "equivalent_strain": (
            _per_grain_medians(map_a.equivalent_strain(grain_a), masks_a),
            _per_grain_medians(map_b.equivalent_strain(grain_b), masks_b),
        ),
        "max_principal_strain": (
            _per_grain_medians(eig_a[..., 0], masks_a),
            _per_grain_medians(eig_b[..., 0], masks_b),
        ),
        "min_principal_strain": (
            _per_grain_medians(eig_a[..., -1], masks_a),
            _per_grain_medians(eig_b[..., -1], masks_b),
        ),
        "max_shear_strain": (
            _per_grain_medians((eig_a[..., 0] - eig_a[..., -1]) / 2.0, masks_a),
            _per_grain_medians((eig_b[..., 0] - eig_b[..., -1]) / 2.0, masks_b),
        ),
    }

    rows = {
        name: _two_sample_stats(vals_a.values(), vals_b.values(),
                                 n_bootstrap=n_bootstrap, rng=rng)
        for name, (vals_a, vals_b) in quantities.items()
    }

    df = pd.DataFrame(rows).T
    df.index.name = "quantity"
    df.attrs["label_a"] = label_a
    df.attrs["label_b"] = label_b
    return df
