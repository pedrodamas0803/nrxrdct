
import os
from pathlib import Path

import numpy as np
import h5py

class Scan:

    """
    Class that holds data from each scan and their important parameters. 
    Important units:
    - Energy: keV
    - Distance: mm
    """

    def __init__(self, acquisition_file:Path, sample_name:str, scan_type:str="half-turn", translation_motor:str="dty", rotation_motor:str="rot", outer_loop_motor:str="translation", beam_size:float=100e-6, beam_energy:float=44):

        self.acquisition_file = acquisition_file
        self.sample_name = sample_name
        self.translation_motor = translation_motor
        self.rotation_motor = rotation_motor
        self.outer_loop_motor = outer_loop_motor
        self.beam_size = beam_size
        self.beam_energy = beam_energy
        self.scan_type = scan_type
        self.wavelength = 12.398 / self.beam_energy

    def __str__(self):
        msg = f"""
            XRDCT scan stored in {str(self.acquisition_file)}
            Translation motor name: {self.translation_motor}
            Rotation motor name: {self.rotation_motor}
            Scan outer loop: {self.outer_loop_motor}
            """
        return msg
    
    def save_parameter_file(self, output_file:Path=Path("xrdct_scan.h5")):

        with h5py.File(str(output_file), 'a') as hout:
            for flag, value in self.__dict__.items():
                hout[flag] = value

    @classmethod
    def get_scan_from_parameters(cls, parameter_file:Path=Path("xrdct_scan.h5")):

        scan_dict = {}
        with h5py.File(str(parameter_file), 'r') as hin:
            for key in hin.keys():
                value = hin[key][()]
                if not isinstance(value, (np.int64, np.float64)):
                    value = value.decode()
                scan_dict[key] = value

        return Scan(**scan_dict)



if __name__ == "__main__":

    try:
        os.remove("xrdct_scan.h5")
    except:
        pass
    scan = Scan("foo.h5", "sample")

    print(scan)

    scan.save_parameter_file()

    scan2 = Scan.get_scan_from_parameters()

    print("SCAN 2: ", scan2)




