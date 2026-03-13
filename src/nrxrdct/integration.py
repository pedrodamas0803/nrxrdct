import os
import numpy as np
import fabio
import pyFAI
import pyFAI.multi_geometry

def integrate_multigeo(images, poni_files, n_bins=2000, unit = "2th_deg", polarization = 0.5, radial_range = None):
   
    print("=" * 60)
    print("STEP 1: pyFAI multi-geometry integration")
    print("=" * 60)

    print("Loading images...")
    imgs = []
    masks  = []
    for img_path, poni_path in zip(images, poni_files):
        img = fabio.open(img_path).data.astype(np.float32)
        imgs.append(img)
        print(f"  Loaded: {os.path.basename(img_path)}  shape={img.shape}")

        # Use detector gap mask if available
        ai = pyFAI.load(poni_path)
        det_mask = ai.detector.mask
        if det_mask is not None:
            masks.append(det_mask.astype(bool))
            print(f"    Detector mask applied: {det_mask.sum()} pixels masked")
        else:
            masks.append(np.zeros(img.shape, dtype=bool))
            print(f"    No detector mask found")

    print("\nBuilding MultiGeometry...")
    mg = pyFAI.multi_geometry.MultiGeometry(
        ais=poni_files,
        unit=unit,
        radial_range=radial_range,
        empty=0.0
    )

    print("Integrating...")
    result = mg.integrate1d(
        lst_data=imgs,
        npt=n_bins,
        lst_mask=masks,
        polarization_factor=polarization,
        error_model="poisson",
    )

    tth_integrated       = result.radial
    intensity_integrated = result.intensity
    sigma_integrated     = result.sigma if result.sigma is not None \
                        else np.zeros_like(intensity_integrated)

    return tth_integrated, intensity_integrated, sigma_integrated

def save_xy_file(tth, intensity, sigma, unit = '2th_deg', outfile = 'integrated.xy'):
    # Save as .xy file
    header = (
        f"# pyFAI multi-geometry azimuthal integration\n"
        f"# Unit: {unit}\n"
        f"# Columns: {unit}  Intensity  Sigma\n"
    )
    np.savetxt(outfile,
            np.column_stack([tth, intensity, sigma]),
            header=header, fmt="%.6f")
    print(f"Integrated pattern saved to:\n  {outfile}")