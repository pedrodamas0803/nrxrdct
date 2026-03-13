import numpy as np
import astra

HAS_GPU = True if "nvidia" in astra.get_gpu_info().lower() else False

def reconstruct_astra_gpu(data:np.ndarray, dty_step:float=1.0, angles_rad:np.array=np.empty((1,)), algo:str='SART_CUDA', num_iter:int=200):

    N = data.shape[0]
    data = data.T
    # Ensure correct sinogram shape:
    # ASTRA expects (num_angles, num_detectors)
    if data.shape[0] != len(angles_rad):
        raise ValueError(
            "Sinogram must have shape (num_angles, num_detectors)"
        )

    proj_geom = astra.create_proj_geom(
        'parallel', dty_step, N, angles_rad)

    vol_geom = astra.create_vol_geom(N, N)

    sinogram_id = astra.data2d.create('-sino', proj_geom, data)
    recon_id = astra.data2d.create('-vol', vol_geom)

    cfg = astra.astra_dict(algo)
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ReconstructionDataId'] = recon_id

    if algo in ['SIRT_CUDA', 'SART_CUDA']:
        cfg['option'] = {'MinConstraint': 0.0}

    algorithm_id = astra.algorithm.create(cfg)
    astra.algorithm.run(algorithm_id, num_iter)

    reconstruction = astra.data2d.get(recon_id)

    astra.algorithm.delete(algorithm_id)
    astra.data2d.delete([sinogram_id, recon_id])

    return reconstruction

def reconstruct_astra_cpu(data:np.ndarray, dty_step:float=1.0, angles_rad:np.array=np.empty((1,)), algo:str='FBP', num_iter:int=200):

    
    N = data.shape[0]
    data = data.T
    # Ensure correct sinogram shape:
    # ASTRA expects (num_angles, num_detectors)
    if data.shape[0] != len(angles_rad):
        raise ValueError(
            "Sinogram must have shape (num_angles, num_detectors)"
        )

    num_detectors = data.shape[1]

    # Create geometries
    proj_geom = astra.create_proj_geom(
        'parallel',
        dty_step,
        num_detectors,
        angles_rad
    )

    vol_geom = astra.create_vol_geom(num_detectors,
                                     num_detectors)

    # CPU projector (important!)
    projector_id = astra.create_projector(
        'linear',   # CPU projector
        proj_geom,
        vol_geom
    )

    # Create data objects
    sinogram_id = astra.data2d.create('-sino',
                                      proj_geom,
                                      data)

    recon_id = astra.data2d.create('-vol',
                                   vol_geom,
                                   data=0.0)

    # Configure algorithm
    cfg = astra.astra_dict(algo)
    cfg['ProjectorId'] = projector_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ReconstructionDataId'] = recon_id

    # Optional positivity constraint
    if algo in ['SIRT', 'SART', 'FBP']:
        cfg['option'] = {'MinConstraint': 0.0}

    algorithm_id = astra.algorithm.create(cfg)

    # Run reconstruction
    astra.algorithm.run(algorithm_id, num_iter)

    reconstruction = astra.data2d.get(recon_id)

    # Cleanup
    astra.algorithm.delete(algorithm_id)
    astra.data2d.delete([sinogram_id, recon_id])
    astra.projector.delete(projector_id)

    return reconstruction

def reconstruct_slice(data:np.ndarray, dty_step:float=1.0, angles_rad:np.array=np.empty((1,)), algo:str='SART_CUDA', num_iter:int=200):
    N = data.shape[0]
    if angles_rad.shape[0]< 10:
        angles_rad = np.linspace(0, np.pi, N)
    if HAS_GPU:
        print("Reconstructing data using GPU.")
        slc = reconstruct_astra_gpu(data, dty_step, angles_rad, algo, num_iter)
    else:
        print("Reconstructing data using CPU.")
        slc = reconstruct_astra_cpu(data, dty_step, angles_rad, algo, num_iter)

    return slc