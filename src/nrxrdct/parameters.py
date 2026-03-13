from pathlib import Path
import h5py

class Scan:

    """
    Class that holds data from each scan and their important parameters. 
    Important units:
    - Energy: keV
    - Distance: mm
    """

    def __init__(self, file:Path, sample_name:str, scan_type:str="half-turn", translation_motor:str="dty", rotation_motor:str="rot", outer_loop:str="translation", beam_size:float=100e-6, beam_energy:float=44):

        self.acquisition_file = file
        self.sample_name = sample_name
        self.translation_motor = translation_motor
        self.rotation_motor = rotation_motor
        self.outer_loop_motor = outer_loop
        self.beam_size = beam_size
        self.beam_energy = beam_energy
        self.scan_type = scan_type

    def __str__(self):
        msg = f"""
            XRDCT scan stored in {str(self.acquisition_file)}\n
            Translation motor name: {self.translation_motor}\n
            Rotation motor name: {self.rotation_motor}\n
            Scan outer loop: {self.outer_loop_motor}
            """
        return msg
    
    def save_parameter_file(self, output_file:Path="xrdct_scan.h5"):

        with h5py.File(output_file, 'a') as hout:
            for flag, value in self.__dict__.items():
                hout[flag] = value

if __name__ == "__main__":

    scan = Scan("foo.h5", "sample")

    print(scan)

    scan.save_parameter_file()



