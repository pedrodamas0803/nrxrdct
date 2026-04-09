"""
Preprocessing routines for XRD-CT detector images.

Currently implements zinger (hot-pixel) removal using median filtering,
with a parallelised wrapper for processing image stacks.
"""
import os, time, concurrent.futures
import numpy as np
import scipy.ndimage as ndi

NTHREAD = os.cpu_count() - 1


def zinger_remove(dimg, medsize=3, nsigma=5) -> np.ndarray:
    """
    Remove zingers (hot pixels) from a single 2-D detector image.

    Pixels whose deviation from the local median exceeds ``nsigma`` standard
    deviations (computed over the whole residual image) are replaced by the
    median-filtered value.  The replacement region is dilated by one pixel to
    catch ringing artefacts around bright zingers.

    Args:
        dimg (np.ndarray): 2-D detector image.
        medsize (int, optional): Side length of the square median-filter kernel (default 3).
        nsigma (int or float, optional): Sigma threshold above which a pixel is considered
            a zinger (default 5).

    Returns:
        np.ndarray: Cleaned image with the same shape and dtype as *dimg*.
    """
    med = ndi.median_filter(dimg, medsize)
    err = dimg - med
    ds0 = err.std()
    msk = err > ds0 * nsigma
    gromsk = ndi.binary_dilation(msk)
    return np.where(gromsk, med, dimg)


def dezinger(image, medsize: int = 3, nsigma: int = 5) -> np.ndarray:
    """
    Apply zinger removal to a stack of detector images in parallel.

    Each frame is processed independently via :func:`zinger_remove` using a
    thread pool sized to ``os.cpu_count() - 1``.

    Args:
        image (np.ndarray): 3-D array of shape ``(N, rows, cols)`` containing *N* detector frames.
        medsize (int, optional): Median-filter kernel size passed to :func:`zinger_remove`
            (default 3).
        nsigma (int, optional): Sigma threshold passed to :func:`zinger_remove` (default 5).

    Returns:
        np.ndarray: Dezingered stack with the same shape and dtype as *image*.
    """
    t0 = time.time()
    N = image.shape[0]
    def dezing(im):
        return zinger_remove(im, medsize, nsigma)
    print(f"Will dezinger {N} images. Might take few seconds.")
    out_image = np.zeros_like(image)
    with concurrent.futures.ThreadPoolExecutor(NTHREAD) as pool:
        for i, result in enumerate(pool.map(dezing, image)):
            out_image[i] = result
    t1 = time.time()

    print(f"It took {(t1-t0):.2f}s to dezinger {N} images.")
    return out_image