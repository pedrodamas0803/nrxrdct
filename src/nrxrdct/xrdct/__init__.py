"""
XRD-CT reconstruction, volume analysis, and visualization tools.

Subpackage covering the full powder-diffraction CT pipeline: sinogram
assembly, ASTRA-backed reconstruction, per-voxel GSAS-II refinement via
:class:`ReconstructedVolume`, and interactive Jupyter/napari viewers.
"""

from .io import (
    add_array_to_output,
    get_array_from_file,
    read_sinogram_from_file,
    read_volume_from_file,
    read_xy_file,
    save_sinogram,
    save_volume,
    save_xy_file,
    write_calibrated_intrument_pars,
    write_starting_instrument_pars,
)
from .parameters import Scan
from .preprocessing import NTHREAD, dezinger, zinger_remove
from .reconstruction import (
    HAS_GPU,
    NTHREADS,
    forward_project_gpu,
    reconstruct_astra_cpu,
    reconstruct_astra_gpu,
    reconstruct_astra_gpu_3d,
    reconstruct_slice,
)
from .s3dxrd import (
    IndexingResult,
    S3DXRDSlice,
    SegmentationOptions,
    SegmentationResult,
    build_columnfile,
    combine_with_powder,
    index_slice,
    load_segmentation,
    poni_to_par,
    save_segmentation,
    segment_frame,
    segment_scan,
    segment_slice,
)
from .sinogram import assemble_sinogram, get_fluo_full_spectra, get_fluo_roi
from .visualization import (
    ZProfilePlot,
    plot_integrated_cake,
    select_roi,
    visualize_slices,
    visualize_slices_with_profile,
    visualize_slices_with_profile_jupyter,
    visualize_volume,
)
from .volume import ReconstructedVolume

__all__ = [
    # I/O
    "save_sinogram",
    "save_volume",
    "save_xy_file",
    "read_xy_file",
    "add_array_to_output",
    "get_array_from_file",
    "read_sinogram_from_file",
    "read_volume_from_file",
    "write_starting_instrument_pars",
    "write_calibrated_intrument_pars",
    # Scan parameters
    "Scan",
    # Preprocessing (backwards compat)
    "NTHREAD",
    "zinger_remove",
    "dezinger",
    # Reconstruction
    "HAS_GPU",
    "NTHREADS",
    "reconstruct_astra_gpu_3d",
    "reconstruct_astra_gpu",
    "reconstruct_astra_cpu",
    "forward_project_gpu",
    "reconstruct_slice",
    # Sinogram assembly
    "assemble_sinogram",
    "get_fluo_roi",
    "get_fluo_full_spectra",
    # Scanning 3DXRD (s3dxrd)
    "SegmentationOptions",
    "SegmentationResult",
    "IndexingResult",
    "S3DXRDSlice",
    "segment_frame",
    "segment_scan",
    "segment_slice",
    "save_segmentation",
    "load_segmentation",
    "build_columnfile",
    "index_slice",
    "combine_with_powder",
    "poni_to_par",
    # Visualization
    "visualize_volume",
    "visualize_slices",
    "ZProfilePlot",
    "visualize_slices_with_profile",
    "visualize_slices_with_profile_jupyter",
    "select_roi",
    "plot_integrated_cake",
    # Volume analysis
    "ReconstructedVolume",
]
