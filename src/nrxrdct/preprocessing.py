import os, time, concurrent.futures
import numpy as np
import scipy.ndimage as ndi

NTHREAD = os.cpu_count() - 1


def zinger_remove(dimg, medsize=3, nsigma=5):
    """
    remove zingers. Anything which is >5 sigma after a 3x3 median filter is replaced by the filtered values
    """
    med = ndi.median_filter(dimg, medsize)
    err = dimg - med
    ds0 = err.std()
    msk = err > ds0 * nsigma
    gromsk = ndi.binary_dilation(msk)
    return np.where(gromsk, med, dimg)


def dezinger(image, medsize:int=3, nsigma:int=5):
    """
    Performs parallelized zinger removal.

    Inputs
    image
    medsize
    nsigma

    Written this way to have only one argument. Could be improved, but it works.
    """
    t0 = time.time()
    N = image.shape[0]
    def dezing(im):
        return zinger_remove(image, medsize, nsigma)
    print(f"Will dezinger {N} images. Might take few seconds.")
    out_image = np.zeros_like(image)
    with concurrent.futures.ThreadPoolExecutor(NTHREAD) as pool:
        for i, result in enumerate(pool.map(dezing, image)):
            out_image[i] = result
    t1 = time.time()

    print(f"It took {(t1-t0):.2f}s to dezinger {N} images.")
    return out_image