from pathlib import Path
import numpy as np
import h5py

def save_sinogram(sinogram:np.ndarray, output_file:Path):

    with h5py.File(str(output_file), "a") as hout:
        hout['sinogram'] = sinogram

def save_volume(volume:np.ndarray, output_file:Path):

    with h5py.File(str(output_file), "a") as hout:
        hout['volume'] = volume

def add_array_to_output(array:np.ndarray, array_name: str, output_file:Path):
    with h5py.File(output_file, 'a') as hout:
        hout[array_name] = array

def read_sinogram_from_file(input_file:Path, slicing:tuple|None=None):

    with h5py.File(input_file, 'r') as hin:

        if isinstance(slicing, tuple):
            tthmin, tthmax, xmin, xmax, ymin, ymax = slicing
            sinogram = hin['sinogram'][tthmin:tthmax, xmin, xmax, ymin, ymax].astype(np.float32)
        else:
            sinogram = hin['sinogram'][:].astype(np.float32)

    return sinogram

def read_volume_from_file(input_file:Path, slicing:tuple|None=None):
    with h5py.File(input_file, 'r') as hin:
        if isinstance(slicing, tuple):
            tthmin, tthmax, xmin, xmax, ymin, ymax = slicing
            volume = hin['volume'][tthmin:tthmax, xmin, xmax, ymin, ymax].astype(np.float32)
        else:
            volume = hin['volume'][:].astype(np.float32)

    return volume