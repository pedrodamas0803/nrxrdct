"""
Scanning 3DXRD (s3DXRD) processing for spotty XRD-CT diffraction data.

Reuses the exact raw detector frames read by
:func:`nrxrdct.azimuthal.integration.integrate_powder_parallel` for powder
(azimuthal) integration, but instead of averaging the Bragg spots away, this
module segments them and indexes them point-by-point across the scanned
rotation/translation grid (Henningsson & Hall's scanning-3DXRD method) to
recover a per-voxel single-crystal orientation map on the same real-space
cross-section ("slice") that :mod:`nrxrdct.xrdct.reconstruction` reconstructs
from the powder signal.

This wraps low-level building blocks from ImageD11 directly (segmentation
primitives from :mod:`ImageD11.sinograms.lima_segmenter`, geometry from
:mod:`ImageD11.columnfile`/:mod:`ImageD11.transform`, and point-by-point
indexing from :mod:`ImageD11.sinograms.point_by_point`) rather than ImageD11's
own :class:`~ImageD11.sinograms.dataset.DataSet`, since that class assumes an
ESRF Bliss ``{dataroot}/{sample}/{sample}_{dset}/...`` folder layout that
does not match this repo's single-``master_file`` convention.

Geometry note: peak positions are used directly as raw pixel coordinates
(no spline/distortion correction), which is appropriate for pixel-array
detectors such as Eiger. Indexing requires an ImageD11 parameter file
(``.par``, with detector geometry and unit cell). This is independent of
the ``.poni`` file used for azimuthal integration, since ImageD11's geometry
model and pyFAI's are not directly interchangeable — but :func:`poni_to_par`
converts one to the other following the closed-form conversion documented at
https://pyfai.readthedocs.io/en/stable/geometry_conversion.html.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import fabio
import h5py
import numpy as np
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from tqdm import tqdm

_SEG_AVAILABLE: "bool | None" = None    # cImageD11 + sparseframe + lima_segmenter
_IDX_AVAILABLE: "bool | None" = None    # point_by_point (imports tensor_map via Numba)

# Module-level names populated lazily; declared here so static analysers don't
# flag undefined names in the functions that use them.
ImageD11 = None
cImageD11 = None
sparseframe = None
lima_segmenter = None
PBP = None
PBPMap = None

_ERR = (
    "ImageD11 is required for scanning-3DXRD (s3dxrd) processing. "
    "Install it with: pip install nrxrdct[xrdct]"
)


def _require_seg() -> None:
    """Load the segmentation C extensions (cImageD11, sparseframe, lima_segmenter).

    Kept separate from _require_idx() so SLURM segmentation workers never
    import point_by_point → tensor_map, which triggers a Numba JIT compile
    that can fail with stale caches on heterogeneous cluster nodes.
    """
    global _SEG_AVAILABLE, cImageD11, sparseframe, lima_segmenter
    if _SEG_AVAILABLE is True:
        return
    if _SEG_AVAILABLE is False:
        raise ImportError(_ERR)
    try:
        import ImageD11.cImageD11 as _cid11
        import ImageD11.sparseframe as _sf
        from ImageD11.sinograms import lima_segmenter as _ls

        cImageD11      = _cid11
        sparseframe    = _sf
        lima_segmenter = _ls
        _SEG_AVAILABLE = True
    except ImportError:
        _SEG_AVAILABLE = False
        raise ImportError(_ERR)


def _require_idx() -> None:
    """Load the indexing modules (ImageD11.columnfile + point_by_point).

    point_by_point imports tensor_map which has Numba-compiled gufuncs —
    only load this when indexing is actually requested, never in workers.
    """
    global _IDX_AVAILABLE, ImageD11, PBP, PBPMap
    _require_seg()   # columnfile and indexing also need the seg layer
    if _IDX_AVAILABLE is True:
        return
    if _IDX_AVAILABLE is False:
        raise ImportError(_ERR)
    try:
        import ImageD11 as _id11
        import ImageD11.columnfile    # noqa: F401 — registers sub-module on _id11
        from ImageD11.sinograms.point_by_point import PBP as _PBP, PBPMap as _PBPMap

        ImageD11       = _id11
        PBP            = _PBP
        PBPMap         = _PBPMap
        _IDX_AVAILABLE = True
    except ImportError:
        _IDX_AVAILABLE = False
        raise ImportError(_ERR)



def poni_to_par(
    poni_file: Union[str, Path],
    par_file: Union[str, Path],
    cell_params: Optional[dict] = None,
    o11: int = 1,
    o12: int = 0,
    o21: int = 0,
    o22: int = -1,
) -> None:
    """
    Convert a pyFAI ``.poni`` calibration into an ImageD11 ``.par`` parameter
    file, implementing the closed-form conversion documented at
    https://pyfai.readthedocs.io/en/stable/geometry_conversion.html
    (verified by round-tripping pyFAI -> ImageD11 -> pyFAI numerically —
    see the module's test suite).

    pyFAI and ImageD11 describe the same detector geometry with different
    conventions (pyFAI applies the sample-to-detector distance before its
    rotations, ImageD11 after; the axis ordering and rotation signs differ
    too), so this is not a trivial unit change — but it is an exact,
    documented closed-form conversion, not an approximation.

    Two things it does *not* derive from ``poni_file``, and why:

    1. **Detector flip matrix** (``o11``/``o12``/``o21``/``o22``). The
       default (``o11=1, o22=-1, o12=o21=0``) is pyFAI's own documented
       value for "no additional flip", and is correct *if* the raw frames
       used to calibrate ``poni_file`` were read with the same row/column
       orientation as the frames :func:`segment_scan` reads for
       segmentation. That holds by construction in this repo (both read
       ``{entry}/measurement/{camera_name}`` from the same master file), but
       if your ``.poni`` was calibrated from frames read or transposed
       differently, override these.
    2. **Sample/rotation-axis parameters with no PONI equivalent**
       (``chi``, ``wedge``, ``t_x``, ``t_y``, ``t_z``) are written as
       ImageD11's own defaults (``0.0``), not derived from ``poni_file``.

    Always verify the result against a few known/expected reflections before
    trusting it for real indexing — a wrong flip or sign here produces
    plausible-looking but incorrect UBIs.

    Args:
        poni_file (Path): pyFAI ``.poni`` calibration file.
        par_file (Path): Output ImageD11 ``.par`` file.
        cell_params (dict, optional): Unit cell parameters to merge in, with
            keys such as ``cell__a``, ``cell__b``, ``cell__c``,
            ``cell_alpha``, ``cell_beta``, ``cell_gamma``, and
            ``cell_lattice_[P,A,B,C,I,F,R]``. If not given, *par_file* will
            contain detector geometry only (no usable unit cell for indexing).
        o11 (int, optional): Detector flip matrix element — see above.
        o12 (int, optional): Detector flip matrix element — see above.
        o21 (int, optional): Detector flip matrix element — see above.
        o22 (int, optional): Detector flip matrix element — see above.
    """
    import ImageD11.parameters as _id11_params  # pure Python — safe before C extensions load
    ai = AzimuthalIntegrator()
    ai.load(str(poni_file))
    theta1, theta2, theta3 = ai.rot1, ai.rot2, ai.rot3

    distance = (ai.dist / (np.cos(theta1) * np.cos(theta2))) * 1e6  # m -> um
    y_center = (ai.poni2 - ai.dist * np.tan(theta1)) / ai.pixel2
    z_center = (ai.poni1 + ai.dist * np.tan(theta2) / np.cos(theta1)) / ai.pixel1

    geom = {
        "distance": distance,
        "y_center": y_center,
        "z_center": z_center,
        "tilt_x": theta3,
        "tilt_y": theta2,
        "tilt_z": -theta1,
        "wavelength": ai.wavelength * 1e10,  # m -> Angstrom
        "y_size": ai.pixel2 * 1e6,  # m -> um
        "z_size": ai.pixel1 * 1e6,
        "o11": o11,
        "o12": o12,
        "o21": o21,
        "o22": o22,
        "chi": 0.0,
        "wedge": 0.0,
        "t_x": 0.0,
        "t_y": 0.0,
        "t_z": 0.0,
        "omegasign": 1.0,
        "fit_tolerance": 0.05,
        "min_bin_prob": 1e-05,
        "no_bins": 10000,
        "weight_hist_intensities": 0,
    }
    if cell_params:
        geom.update(cell_params)

    pars = _id11_params.parameters(**geom)
    pars.saveparameters(str(par_file))


@dataclass
class SegmentationOptions:
    """
    Peak-segmentation thresholds for scanning-3DXRD spot finding.

    Mirrors :class:`ImageD11.sinograms.lima_segmenter.SegmenterOptions`, but
    takes an already-loaded mask array (same ``1 = masked`` convention used
    by :mod:`nrxrdct.azimuthal.integration`) instead of a mask file path.

    Attributes:
        cut (float): Minimum pixel intensity kept during the first pass
            over a frame.
        howmany (int): Maximum number of pixels kept per frame.
        pixels_in_spot (int): Minimum number of connected pixels for a group
            to be kept as a Bragg spot.
    """

    cut: float = 1.0
    howmany: int = 100_000
    pixels_in_spot: int = 3

    def to_imaged11(self, active_mask: np.ndarray):
        """
        Build an ImageD11 ``SegmenterOptions`` bound to an in-memory mask.

        ImageD11's own :func:`~ImageD11.sinograms.lima_segmenter.clean`
        always dereferences ``options.mask.shape``, so the mask must be set
        directly here rather than routed through ``SegmenterOptions.setup()``
        (which only loads a mask from ``maskfile`` and otherwise leaves it
        ``None``).
        """
        _require_seg()
        opts = lima_segmenter.SegmenterOptions(
            cut=self.cut, howmany=self.howmany, pixels_in_spot=self.pixels_in_spot
        )
        opts.thresholds = tuple(self.cut * pow(2, i) for i in range(6))
        opts.mask = active_mask
        return opts


@dataclass
class SegmentationResult:
    """Segmented Bragg-spot peaks for one scan (one rotation sweep at fixed dty)."""

    entry: str
    dty: float
    sc: np.ndarray
    fc: np.ndarray
    omega: np.ndarray
    sum_intensity: np.ndarray
    n_pixels: np.ndarray

    @property
    def n_peaks(self) -> int:
        return len(self.sc)

    def __str__(self) -> str:
        return (
            f"SegmentationResult(entry={self.entry!r}, dty={self.dty:.4f}, "
            f"n_peaks={self.n_peaks})"
        )


@dataclass
class IndexingResult:
    """
    Per-voxel single-crystal orientation map for one XRD-CT slice, produced
    by scanning-3DXRD point-by-point indexing.

    Attributes:
        grains_file (str): Path to the raw point-by-point grains text file
            written by ImageD11 (multiple candidate UBIs per pixel).
        best_ubi (np.ndarray): Best UBI matrix per pixel, shape ``(NI, NJ, 3, 3)``.
        best_nuniq (np.ndarray): Number of uniquely-indexed peaks for the best
            UBI at each pixel, shape ``(NI, NJ)``; ``0`` where no grain was found.
        best_npks (np.ndarray): Total number of peaks used for the best UBI
            at each pixel, shape ``(NI, NJ)``.
        symmetry (str): Crystal symmetry used for indexing (e.g. ``"cubic"``).
        phase_name (str, optional): Phase name, if the parameter file defines
            more than one phase.
    """

    grains_file: str
    best_ubi: np.ndarray
    best_nuniq: np.ndarray
    best_npks: np.ndarray
    symmetry: str
    phase_name: Optional[str] = None

    def __str__(self) -> str:
        n_indexed = int(np.sum(self.best_nuniq > 0))
        n_total = self.best_nuniq.size
        return (
            f"IndexingResult(phase={self.phase_name!r}, symmetry={self.symmetry!r}, "
            f"indexed {n_indexed}/{n_total} pixels)"
        )

    def save(self, h5file: Union[str, Path], group: str = "s3dxrd") -> None:
        """Save the orientation map to an HDF5 file under ``group``."""
        with h5py.File(h5file, "a") as hout:
            grp = hout.require_group(group)
            grp.attrs["symmetry"] = self.symmetry
            if self.phase_name is not None:
                grp.attrs["phase_name"] = self.phase_name
            grp.attrs["grains_file"] = self.grains_file
            for name in ("best_ubi", "best_nuniq", "best_npks"):
                if name in grp:
                    del grp[name]
                grp.create_dataset(
                    name, data=getattr(self, name), compression="gzip"
                )


@dataclass
class _MiniDataset:
    """
    Minimal ``(ybincens, ystep)`` stand-in for
    :class:`ImageD11.sinograms.dataset.DataSet`, since
    :class:`ImageD11.sinograms.point_by_point.PBP` only ever reads those two
    attributes off its ``dset`` argument.
    """

    ybincens: np.ndarray
    ystep: float


def segment_frame(
    image: np.ndarray, active_mask: np.ndarray, options
) -> Optional[np.ndarray]:
    """
    Segment one detector frame into Bragg-spot peak centroids.

    Args:
        image (np.ndarray): 2D detector frame.
        active_mask (np.ndarray): 2D array, ``1`` = active pixel, ``0`` =
            masked — the inverse of the pyFAI/fabio mask convention
            (``1`` = masked) used elsewhere in this repo.
        options (ImageD11.sinograms.lima_segmenter.SegmenterOptions): see
            :meth:`SegmentationOptions.to_imaged11`.

    Returns:
        np.ndarray or None: ``(n_peaks, 4)`` array of columns
            ``(sc, fc, sum_intensity, n_pixels)``, or ``None`` if no peaks
            were found on this frame.
    """
    _require_seg()
    to_sparse = lima_segmenter.frmtosparse(active_mask, image.dtype)
    nnz, row, col, val = to_sparse(image, options.cut)
    sf = lima_segmenter.clean(nnz, row, col, val, config_options=options)
    if sf is None:
        return None
    nlabel = sf.meta["cp"]["nlabel"]
    if nlabel == 0:
        return None
    moments = sparseframe.sparse_moments(sf, intensity_name="f32", labels_name="cp")
    sum_intensity = moments[:, cImageD11.s2D_I]
    sc = moments[:, cImageD11.s2D_sI] / sum_intensity
    fc = moments[:, cImageD11.s2D_fI] / sum_intensity
    n_pixels = moments[:, cImageD11.s2D_1]
    return np.column_stack([sc, fc, sum_intensity, n_pixels])


def segment_scan(
    master_file: Union[str, Path],
    entry: str,
    mask: np.ndarray,
    options: SegmentationOptions,
    camera_name: str = "eiger",
    translation_motor: str = "dty",
    rotation_motor: str = "rot",
) -> SegmentationResult:
    """
    Segment every frame of one scan (HDF5 entry) into Bragg-spot peaks.

    Reads the same raw image stack consumed by
    :func:`nrxrdct.azimuthal.integration.integrate_powder_parallel` for
    powder integration, but keeps the Bragg-spot signal instead of
    integrating it away.

    Args:
        master_file (Path): Path to the master HDF5 file containing all scan entries.
        entry (str): HDF5 entry key for this scan (e.g. ``"1.1"``).
        mask (np.ndarray): Mask array, ``1`` = masked, ``0`` = valid (pyFAI/fabio convention).
        options (SegmentationOptions): Peak-segmentation thresholds.
        camera_name (str, optional): Detector dataset name under ``measurement/``
            (default ``"eiger"``).
        translation_motor (str, optional): Translation motor name under
            ``instrument/positioners/`` (default ``"dty"``).
        rotation_motor (str, optional): Rotation motor dataset name under
            ``measurement/`` (default ``"rot"``).

    Returns:
        SegmentationResult: Segmented peaks for this scan.
    """
    _require_seg()
    active_mask = (1 - mask).astype(np.uint8)
    id11_options = options.to_imaged11(active_mask)

    with h5py.File(master_file, "r") as hin:
        images = hin[f"{entry}/measurement/{camera_name}"][:].astype(np.float32)
        omega = hin[f"{entry}/measurement/{rotation_motor}"][:]
        dty = float(hin[f"{entry}/instrument/positioners/{translation_motor}"][()])

    if omega[-1] < omega[0]:
        images = images[::-1]
        omega = omega[::-1]

    sc_chunks, fc_chunks, omega_chunks = [], [], []
    sumi_chunks, npix_chunks = [], []
    for frame_idx in range(len(images)):
        peaks = segment_frame(images[frame_idx], active_mask, id11_options)
        if peaks is None:
            continue
        n = len(peaks)
        sc_chunks.append(peaks[:, 0])
        fc_chunks.append(peaks[:, 1])
        sumi_chunks.append(peaks[:, 2])
        npix_chunks.append(peaks[:, 3])
        omega_chunks.append(np.full(n, omega[frame_idx]))

    if sc_chunks:
        sc = np.concatenate(sc_chunks)
        fc = np.concatenate(fc_chunks)
        omega_out = np.concatenate(omega_chunks)
        sum_intensity = np.concatenate(sumi_chunks)
        n_pixels = np.concatenate(npix_chunks)
    else:
        sc = fc = omega_out = sum_intensity = n_pixels = np.empty(0, dtype=np.float64)

    return SegmentationResult(
        entry=entry,
        dty=dty,
        sc=sc,
        fc=fc,
        omega=omega_out,
        sum_intensity=sum_intensity,
        n_pixels=n_pixels,
    )


def _segment_scan_worker(args: tuple) -> tuple:
    """Top-level worker so ProcessPoolExecutor can pickle it."""
    ii, master_file, entry, mask, options, camera_name, translation_motor, rotation_motor = args
    result = segment_scan(
        master_file, entry, mask, options,
        camera_name=camera_name,
        translation_motor=translation_motor,
        rotation_motor=rotation_motor,
    )
    return ii, result


def segment_slice(
    master_file: Union[str, Path],
    output_file: Union[str, Path],
    mask_file: Union[str, Path],
    options: Optional[SegmentationOptions] = None,
    camera_name: str = "eiger",
    translation_motor: str = "dty",
    rotation_motor: str = "rot",
    n_workers: int = 1,
) -> List[SegmentationResult]:
    """
    Segment every scan of one XRD-CT slice (a full rotation+translation
    sweep at fixed sample height) into Bragg-spot peaks, writing each scan
    to ``output_file`` as soon as it is segmented.

    A scan already present in *output_file* is skipped without re-segmenting
    it, so an interrupted run can be resumed. A scan that fails to read or
    segment is skipped with a warning instead of aborting the whole slice —
    matching :func:`nrxrdct.azimuthal.integration.integrate_powder_parallel`.

    Args:
        master_file (Path): Path to the master HDF5 file containing all scan entries.
        output_file (Path): Path to the output HDF5 file for segmented peaks.
        mask_file (Path): Path to the mask file (fabio-readable, ``1`` = masked).
        options (SegmentationOptions, optional): Peak-segmentation thresholds
            (defaults used if not given).
        camera_name (str, optional): Detector dataset name (default ``"eiger"``).
        translation_motor (str, optional): Translation motor name (default ``"dty"``).
        rotation_motor (str, optional): Rotation motor dataset name (default ``"rot"``).
        n_workers (int, optional): Number of parallel worker processes for
            segmentation (default ``1``). Each worker processes one scan
            independently; HDF5 writes are always serialised in the main
            process. Set to ``os.cpu_count()`` to use all available cores.

    Returns:
        list[SegmentationResult]: Segmented peaks for every scan segmented
            in this call or already present in *output_file*, in entry order.
    """
    _require_seg()
    if options is None:
        options = SegmentationOptions()
    mask = fabio.open(mask_file).data

    print("Reading entries from master file...")
    valid_entries = []
    with h5py.File(master_file, "r") as hin:
        all_entries = list(hin.keys())
        for entry in tqdm(all_entries, desc="Validating entries"):
            try:
                _ = hin[f"{entry}/measurement/{camera_name}"].shape
                _ = hin[f"{entry}/instrument/positioners/{translation_motor}"][()]
                valid_entries.append(entry)
            except KeyError as e:
                print(f"  ⚠  Entry {entry} missing expected dataset ({e}) — skipping")

    print(f"\n✓  {len(valid_entries)}/{len(all_entries)} entries OK")

    results_map: dict = {}

    # Scans already on disk — read them back without re-submitting.
    pending_args = []
    with h5py.File(output_file, "a") as hout:
        for ii, entry in enumerate(valid_entries):
            group_path = f"segmented/scan_{ii:04d}"
            if group_path in hout:
                results_map[ii] = _read_scan_group(hout[group_path], translation_motor)
            else:
                pending_args.append(
                    (ii, str(master_file), entry, mask, options,
                     camera_name, translation_motor, rotation_motor)
                )

    n_cached = len(results_map)
    if n_cached:
        print(f"  {n_cached} scans already done — skipping")

    if pending_args:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_segment_scan_worker, a): a[0] for a in pending_args}
            with tqdm(total=len(futures), desc="Segmenting scans") as pbar:
                for future in as_completed(futures):
                    ii = futures[future]
                    entry = valid_entries[ii]
                    group_path = f"segmented/scan_{ii:04d}"
                    try:
                        _, result = future.result()
                    except (OSError, KeyError) as e:
                        print(f"  ✗ Failed to segment entry {entry}: {e} — skipping")
                        pbar.update()
                        continue
                    with h5py.File(output_file, "a") as hout:
                        _write_scan_group(hout, group_path, result, translation_motor)
                    results_map[ii] = result
                    pbar.update()

    return [results_map[ii] for ii in sorted(results_map)]


def _write_scan_group(
    hout: h5py.File,
    group_path: str,
    result: SegmentationResult,
    translation_motor: str,
) -> None:
    grp = hout.create_group(group_path)
    grp.create_dataset("sc", data=result.sc, compression="gzip")
    grp.create_dataset("fc", data=result.fc, compression="gzip")
    grp.create_dataset("omega", data=result.omega, compression="gzip")
    grp.create_dataset("sum_intensity", data=result.sum_intensity, compression="gzip")
    grp.create_dataset("n_pixels", data=result.n_pixels, compression="gzip")
    grp.attrs["entry"] = result.entry
    grp.attrs[translation_motor] = result.dty
    grp.attrs["n_peaks"] = result.n_peaks


def _read_scan_group(grp: h5py.Group, translation_motor: str) -> SegmentationResult:
    return SegmentationResult(
        entry=grp.attrs["entry"],
        dty=float(grp.attrs[translation_motor]),
        sc=grp["sc"][:],
        fc=grp["fc"][:],
        omega=grp["omega"][:],
        sum_intensity=grp["sum_intensity"][:],
        n_pixels=grp["n_pixels"][:],
    )


def save_segmentation(
    output_file: Union[str, Path],
    results: Sequence[SegmentationResult],
    translation_motor: str = "dty",
) -> None:
    """Save a list of :class:`SegmentationResult` to ``segmented/scan_XXXX`` groups."""
    with h5py.File(output_file, "a") as hout:
        for ii, r in enumerate(results):
            group_path = f"segmented/scan_{ii:04d}"
            if group_path in hout:
                continue
            _write_scan_group(hout, group_path, r, translation_motor)


def load_segmentation(
    segmentation_file: Union[str, Path], translation_motor: str = "dty"
) -> List[SegmentationResult]:
    """Reload the list of :class:`SegmentationResult` written by :func:`save_segmentation`."""
    with h5py.File(segmentation_file, "r") as hin:
        scan_keys = sorted(hin["segmented"].keys())
        return [
            _read_scan_group(hin[f"segmented/{key}"], translation_motor)
            for key in scan_keys
        ]


def build_columnfile(
    segmentation: Union[str, Path, Sequence[SegmentationResult]],
    par_file: Union[str, Path],
    phase_name: Optional[str] = None,
) -> "object":  # ImageD11.columnfile.columnfile
    """
    Assemble segmented per-scan peaks into a single ImageD11 columnfile with
    full diffraction geometry (tth, eta, g-vectors, d-spacing) computed from
    the supplied ImageD11 parameter file.

    Args:
        segmentation: Either a path to a segmentation HDF5 file written by
            :func:`segment_slice`, or an already-loaded list of
            :class:`SegmentationResult`.
        par_file (Path): ImageD11 parameter file (detector geometry + unit cell).
        phase_name (str, optional): Phase name, if ``par_file`` defines more
            than one phase.

    Returns:
        ImageD11.columnfile.columnfile: Columnfile with columns ``sc``,
            ``fc``, ``omega``, ``dty``, ``sum_intensity``,
            ``Number_of_pixels``, plus the geometry columns added by
            ``updateGeometry()``.
    """
    _require_idx()
    if isinstance(segmentation, (str, Path)):
        results = load_segmentation(segmentation)
    else:
        results = list(segmentation)

    results = [r for r in results if r.n_peaks > 0]
    if not results:
        raise ValueError("No segmented peaks found across any scan.")

    colf = ImageD11.columnfile.colfile_from_dict(
        {
            "sc": np.concatenate([r.sc for r in results]),
            "fc": np.concatenate([r.fc for r in results]),
            "omega": np.concatenate([r.omega for r in results]),
            "dty": np.concatenate([np.full(r.n_peaks, r.dty) for r in results]),
            "sum_intensity": np.concatenate([r.sum_intensity for r in results]),
            "Number_of_pixels": np.concatenate([r.n_pixels for r in results]),
        }
    )
    colf.parameters.loadparameters(str(par_file), phase_name=phase_name)
    colf.updateGeometry()
    return colf


def index_slice(
    colf: "object",  # ImageD11.columnfile.columnfile
    par_file: Union[str, Path],
    grains_file: Union[str, Path],
    symmetry: str = "cubic",
    phase_name: Optional[str] = None,
    y0: float = 0.0,
    hkl_tol: float = 0.05,
    fpks: float = 0.7,
    ds_tol: float = 0.005,
    etacut: float = 0.1,
    ifrac: Optional[float] = None,
    gmax: int = 5,
    uniqcut: float = 0.75,
    foridx: Optional[Sequence[int]] = None,
    forgen: Optional[Sequence[int]] = None,
    gridstep: int = 1,
    n_procs: Optional[int] = None,
    minpeaks: int = 6,
) -> IndexingResult:
    """
    Run scanning-3DXRD point-by-point indexing across one XRD-CT slice.

    For every ``(i, j)`` pixel of the sample cross-section — the same
    real-space cross-section reconstructed from the powder signal by
    :func:`nrxrdct.xrdct.reconstruction.reconstruct_slice` — finds the
    single-crystal orientation(s) consistent with the Bragg peaks observed
    at that pixel, i.e. the peaks whose sinusoidal trajectory in
    ``(omega, dty)`` space passes through it.

    Args:
        colf: Columnfile built by :func:`build_columnfile`.
        par_file (Path): ImageD11 parameter file (detector geometry + unit cell).
        grains_file (Path): Output path for the raw point-by-point grains
            text file.
        symmetry (str, optional): Crystal symmetry (default ``"cubic"``).
        phase_name (str, optional): Phase name, if ``par_file`` defines more
            than one phase.
        y0 (float, optional): Value of the ``dty`` motor when the rotation
            axis is centred in the beam (default ``0.0``).
        hkl_tol (float, optional): HKL matching tolerance in reciprocal space.
        fpks (float, optional): Minimum fraction (or count, if ``>= 1``) of
            expected peaks required for an indexing candidate.
        ds_tol (float, optional): d-spacing tolerance for ring assignment.
        etacut (float, optional): Minimum ``|sin(eta)|`` for a peak to be used.
        ifrac (float, optional): Minimum intensity fraction (of the strongest
            peak on a ring) for a peak to be used; defaults to ``1/n_dty_positions``.
        gmax (int, optional): Maximum number of candidate grains kept per pixel.
        uniqcut (float, optional): Minimum fraction of uniquely-matched peaks
            for a candidate to be accepted.
        foridx (Sequence[int], optional): Ring indices to use for peak selection.
        forgen (Sequence[int], optional): Ring indices to use for orientation generation.
        gridstep (int, optional): Sampling step of the ``(i, j)`` reconstruction grid.
        n_procs (int, optional): Number of worker processes (defaults to all
            available cores).
        minpeaks (int, optional): Minimum number of uniquely-matched peaks
            for :meth:`~ImageD11.sinograms.point_by_point.PBPMap.choose_best`
            to accept a pixel's best candidate.

    Returns:
        IndexingResult: Per-voxel single-crystal orientation map for this slice.
    """
    _require_idx()
    dty_values = np.asarray(colf.dty)
    ybincens = np.array(sorted(set(np.round(dty_values, 6))))
    if len(ybincens) < 2:
        raise ValueError("Need at least two distinct dty positions to index a slice.")
    ystep = float(np.median(np.diff(ybincens)))
    dset = _MiniDataset(ybincens=ybincens, ystep=ystep)

    pbp = PBP(
        str(par_file),
        dset,
        hkl_tol=hkl_tol,
        fpks=fpks,
        ds_tol=ds_tol,
        etacut=etacut,
        ifrac=ifrac,
        gmax=gmax,
        y0=y0,
        symmetry=symmetry,
        foridx=list(foridx) if foridx is not None else None,
        forgen=list(forgen) if forgen is not None else None,
        uniqcut=uniqcut,
        phase_name=phase_name,
    )
    icolf_file = str(Path(grains_file).with_suffix("")) + "_icolf.h5"
    pbp.setpeaks(colf, icolf_filename=icolf_file)
    pbp.point_by_point(str(grains_file), nprocs=n_procs, gridstep=gridstep)

    pbp_map = PBPMap(str(grains_file))
    pbp_map.choose_best(minpeaks=minpeaks)

    return IndexingResult(
        grains_file=str(grains_file),
        best_ubi=pbp_map.best_ubi,
        best_nuniq=pbp_map.best_nuniq,
        best_npks=pbp_map.best_npks,
        symmetry=symmetry,
        phase_name=phase_name,
    )


def combine_with_powder(
    s3dxrd_output_file: Union[str, Path],
    reconstruction_file: Union[str, Path],
    tth_index: int,
    pixel_size_mm: float,
) -> None:
    """
    Copy the powder (azimuthally-integrated) tomographic reconstruction of
    one 2-theta channel into the same HDF5 file as the scanning-3DXRD
    orientation map (written by :meth:`IndexingResult.save`), so the powder
    background and the single-crystal grain map for this XRD-CT slice live
    side by side in one file, under ``powder/`` and ``s3dxrd/`` respectively.

    Args:
        s3dxrd_output_file (Path): Output HDF5 file, already containing an
            ``s3dxrd`` group written by :meth:`IndexingResult.save`.
        reconstruction_file (Path): HDF5 file produced by
            ``nrxrdct.xrdct.slurm_reconstruction`` (or
            :func:`nrxrdct.xrdct.reconstruction.reconstruct_slice`), containing
            ``reconstruction/slice_XXXX`` datasets.
        tth_index (int): Index of the 2-theta channel/slice to copy.
        pixel_size_mm (float): Real-space pixel size (mm) of the powder
            reconstruction, recorded so the powder image and the
            point-by-point orientation map (which has its own ``ystep``
            recorded under ``s3dxrd``, see :func:`index_slice`) can be
            registered against each other downstream.
    """
    with h5py.File(reconstruction_file, "r") as hin:
        powder_slice = hin[f"reconstruction/slice_{tth_index:04d}"][:]

    with h5py.File(s3dxrd_output_file, "a") as hout:
        grp = hout.require_group("powder")
        if "image" in grp:
            del grp["image"]
        grp.create_dataset("image", data=powder_slice, compression="gzip")
        grp.attrs["tth_index"] = tth_index
        grp.attrs["source"] = str(reconstruction_file)
        grp.attrs["pixel_size_mm"] = pixel_size_mm


class S3DXRDSlice:
    """
    Scanning-3DXRD processing of one XRD-CT slice (a full rotation +
    translation sweep at fixed sample height).

    Drives the pipeline segment -> assemble columnfile -> point-by-point
    index -> save, producing a per-voxel single-crystal orientation map
    that can then be superposed with the powder reconstruction of the same
    slice via :func:`combine_with_powder`.

    Example:
        >>> sl = S3DXRDSlice(master_file, mask_file, par_file, sample_name="sample1")
        >>> sl.segment("segmented.h5")
        >>> sl.build_columnfile()
        >>> sl.index("grains.txt")
        >>> sl.save("s3dxrd_slice.h5")
    """

    def __init__(
        self,
        master_file: Union[str, Path],
        mask_file: Union[str, Path],
        par_file: Union[str, Path],
        sample_name: str = "sample",
        phase_name: Optional[str] = None,
        symmetry: str = "cubic",
    ):
        self.master_file = str(master_file)
        self.mask_file = str(mask_file)
        self.par_file = str(par_file)
        self.sample_name = sample_name
        self.phase_name = phase_name
        self.symmetry = symmetry

        self.segmentation: Optional[List[SegmentationResult]] = None
        self.columnfile = None
        self.indexing: Optional[IndexingResult] = None

    def segment(
        self,
        output_file: Union[str, Path],
        options: Optional[SegmentationOptions] = None,
        camera_name: str = "eiger",
        translation_motor: str = "dty",
        rotation_motor: str = "rot",
        n_workers: int = 1,
    ) -> List[SegmentationResult]:
        """Segment Bragg-spot peaks from every scan of this slice. See :func:`segment_slice`."""
        self.segmentation = segment_slice(
            self.master_file,
            output_file,
            self.mask_file,
            options=options,
            camera_name=camera_name,
            translation_motor=translation_motor,
            rotation_motor=rotation_motor,
            n_workers=n_workers,
        )
        return self.segmentation

    def build_columnfile(self):
        """Assemble segmented peaks into an ImageD11 columnfile. See :func:`build_columnfile`."""
        if self.segmentation is None:
            raise RuntimeError("Call segment() before build_columnfile().")
        self.columnfile = build_columnfile(
            self.segmentation, self.par_file, phase_name=self.phase_name
        )
        return self.columnfile

    def index(self, grains_file: Union[str, Path], **kwargs) -> IndexingResult:
        """Run point-by-point indexing on this slice. See :func:`index_slice`."""
        if self.columnfile is None:
            raise RuntimeError("Call build_columnfile() before index().")
        self.indexing = index_slice(
            self.columnfile,
            self.par_file,
            grains_file,
            symmetry=self.symmetry,
            phase_name=self.phase_name,
            **kwargs,
        )
        return self.indexing

    def save(self, output_file: Union[str, Path]) -> None:
        """Save the orientation map to ``output_file``. See :meth:`IndexingResult.save`."""
        if self.indexing is None:
            raise RuntimeError("Call index() before save().")
        self.indexing.save(output_file)
        with h5py.File(output_file, "a") as hout:
            hout.attrs["sample_name"] = self.sample_name

    def __str__(self) -> str:
        status = self.indexing if self.indexing is not None else "not indexed yet"
        return f"S3DXRDSlice(sample={self.sample_name!r}, {status})"
