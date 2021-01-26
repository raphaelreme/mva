# Unable to use matfile and be RAM efficient
import enum
import argparse
import os

import numpy as np
import scipy.io


# Meta data that cannot be extracted from the database directly
class SamplingRate(enum.IntEnum):
    """The Sampling Rate of the different DBs.

    To be completed as needed
    """
    DB1 = 100
    DB2 = 2000
    DB7 = 2000


class NinaProPreprocessor:
    """Preprocess matlab files into npz files.

    Keep only what we need. Much more quick to load. And ram efficient
    (for an unknown reason, we were RAM unefficient with the matlab files)

    Attrs:
        METADATA (List[str]): List of metadata key to keep
        DATA (List[str]): List of data key to keep
    """
    METADATA = {"subject", "exercise"}
    DATA = {"emg", "stimulus", "glove"}

    def __init__(self, data_folder: str, database:str, sampling_rate:int = 400):
        """Constructor:

        Args:
            data_folder (str): The directory where the mat files are stored
            database (str): The name of the Database it belongs to (Allow usage of extra metadata)
            sampling_rate (int): Downsample to this sampling rate (Upsampling not allowed)
                Useful to reduce data size
        """
        self.database = database
        self.data_folder = data_folder

        assert sampling_rate <= SamplingRate[self.database], "Will not perform upsampling"
        sampling_rate = SamplingRate[self.database] // sampling_rate
        self.sampling_rate = SamplingRate[self.database] // sampling_rate

    def preprocess(self):
        sampling_rate = SamplingRate[self.database] // self.sampling_rate

        files = os.listdir(self.data_folder)
        for file in sorted(filter(lambda file: file[-4:] == ".mat", files)):
            print(file)
            mat_file = scipy.io.loadmat(os.path.join(self.data_folder, file))

            # Filter non correct mat file
            skip = False
            for key in self.DATA.union(self.METADATA):
                if key not in mat_file:
                    print(f"WARNIING: Missing key {key} in {file}! Skipping it.")
                    skip = True
                    break
            if skip:
                continue

            # Down sampling
            for key in self.DATA:
                mat_file[key] = mat_file[key][::sampling_rate]

            data = {key: mat_file[key] for key in self.DATA.union(self.METADATA)}

            npz_file = os.path.join(self.data_folder, f"{file[:-4]}")
            np.savez(npz_file, **data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessor")
    parser.add_argument(
        "folder",
        help="Location of the data",
    )
    parser.add_argument(
        "--database",
        default="DB7",
        help="Database it belongs to. Supported: [DB1, DB2, DB7]. Default: DB7",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=200,
        help="Down sample to this sampling rate",
    )

    args = parser.parse_args()
    preprocessor = NinaProPreprocessor(args.folder, args.database, args.sampling_rate)
    preprocessor.preprocess()
