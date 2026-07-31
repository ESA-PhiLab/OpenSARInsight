"""l0 Patch Generator for TDS of project OPENSAR-INSIGHT"""

import glob
import os
import sys
import numpy as np
import argparse
import datetime
from typing import Literal, Optional, TextIO

from astropy.time import Time

from . import __version__

Polarization = Literal["vv", "vh"]
SensingType = np.dtype([("burstFilename", "U40"), ("sensingTime", "f8")])

PROG = __package__
VERSION = __version__

def _write_l0_patch(
        rawdata: np.ndarray,
        filename: str
) -> bool:
    """Write a L0 patch to a file

    Write the raw data matrix with the patch contain to a file with the file
    name provided.
    File format is:
        - Data is stored in Little Endian format
        - Complex data are stored first real part, then imaginary part
        - Complex data are stored in 32-bits float type
        - File header contains the size (rows, cols) in uint32 type

            <Number of pulses (az_size)> in 32-bits unsigned integer
            <Number of samples (rg_size)> in 32-bits unsigned integer
            <real(pulse0,sample0)>
            <imag(pulse0,sample0)>
            ....
            <real(pulse0,rg_size)>
            <imag(pulse0,rg_size)>
            <real(pulse1,sample0)>
            <imag(pulse1,sample0)>
            ....
            <real(pulse1,rg_size)>
            <imag(pulse1,rg_size)>
            ...............
            <real(az_size,rg_size)>
            <imag(az_size,rg_size)>

    :parameter
        rawdata:    array of complex. Matrix with the raw data echoes of a
                    single patch. Each row is a pulse and each column is a
                    sample of the pulse
        filename:   string. Name (path included) of the file to be written
    :return
        Boolean value showing the correct generation of the file
    """
    az_size, rg_size = rawdata.shape
    # Rewrite data matrix in a one-dimensional array, one pulse after the 
    # previous one
    complex_data = (rawdata.flatten()) 

    real = np.array(np.real(complex_data), dtype=">f4") 
    imag = np.array(np.imag(complex_data), dtype=">f4")

    # data will be stored as <real>,<imaginary> tuple values
    data = np.empty((2*len(real),), dtype=">f4")  
    data[0::2] = real
    data[1::2] = imag

    # Check and create output folder
    folder = os.path.dirname(filename)
    os.makedirs(folder,exist_ok=True)

    try:
        with open(filename,"w") as f:
            #
            # File format: Binary file in little-endian format
            #   <Number of pulses (az_size)> in 32-bits unsigned integer
            #   <Number of samples (rg_size)> in 32-bits unsigned integer
            #   <real(sample0,pulse0)>
            #   <imag(sample0,pulse0)>
            #   ....
            #   <real(sample rg_size,pulse0)>
            #   <imag(sample rg_size,pulse0)>
            #   <real(sample0,pulse1)>
            #   <imag(sample0,pulse1)>
            #   ....
            #   <real(sample rg_size,pulse1)>
            #   <imag(sample rg_size,pulse1)>
            #   ...............
            #   <real(sample rg_size,pulse az_size)>
            #   <imag(sample rg_size,pulse az_size)>
            # with all real and imaginary data in 32-bits float format

            np.array([az_size,rg_size],
                     dtype=">u4"
                     ).byteswap().tofile(f, format="")
            data.byteswap().tofile(f)

        return True
    except (IOError, OSError) as e:
        print(f"ERROR: an I/O error is detected: {e}")
    except Exception as e:
        print(f"ERROR: {e}")

    return False


def _extract_patch(
        rawdata, 
        patch_info, 
        fid_log
) -> np.ndarray:
    """Extract a submatrix of the burst raw data

    Extracts a section of the raw data defined by the coordinates provided in
    the patch info. If patch coordinates are outside the raw data matrix,
    the coordinates will be limited to the raw data ones

    :parameter
        rawdata:    array of complex. Matrix with the raw data echoes of a
                    single burst. Each row is an echo and each column is a
                    sample of the echo
        patch_info: tuple. Information of each patch to be processed:
                        [0]: Subswath where the patch is loacted
                        [1]: Patch identification
                        [2]: Patch coordinates in SLC product
                        [3]: Patch coordinates inside the burst
                            [Min pulse, Max pulse, Min sample, Max sample]
                        [4]: Burst sensing time as provided in L1B
                             annotation file
        fid_log:    TextIO (optional value). File descriptor where the log
                    messages will be written. If None, no log messages are
                    generated
    :return:
        Matrix with the section of rawdata defined by the coordinates provided
        in patch_info
    """
    az_size, rg_size = rawdata.shape
    coord = patch_info[3]
    patch_number = patch_info[1]

    # If pulse number is less than 0 (first valid pulse), patch will be 
    # limited to first pulse
    min_az = int(np.floor(max(0,coord[0]-1)))
    if coord[0] < 0: # warning message for patch truncation
        warning_text = (f"WARNING: Patch index {patch_number}: "
                        f"Min Azimuth coordinate < 0. "
                        f"Value received: {coord[0]}")
        print(warning_text)
        if fid_log:
            fid_log.write(warning_text + "\n")

    # if pulse number is greater than burst size, patch will be limited to 
    # last valid pulse
    max_az = int(np.ceil(min(az_size-1,coord[1]-1)))
    if coord[1] > (az_size-1):
        warning_text = (f"WARNING: Patch index {patch_number}: "
                        f"Maz Azimuth coordinate > {az_size-1}. "
                        f"Value received: {coord[1]}")
        print(warning_text)
        if fid_log:
            fid_log.write(warning_text + "\n")

    # if sample number is less than 0 (first valid sample), patch will be 
    # limited to first valid sample
    min_rg = int(np.floor(max(0, coord[2]-1)))
    if coord[3] < 0:
        warning_text = (f"WARNING: Patch index {patch_number}: "
                        f"Min Range coordinate < 0. "
                        f"Value received: {coord[2]}")
        print(warning_text)
        if fid_log:
            fid_log.write(warning_text + "\n")

    # if sample number is greater than pulse length, patche will be limited 
    # to last valid sample
    max_rg = int(np.ceil(min(rg_size - 1, coord[3]-1)))
    if coord[3] > (rg_size - 1):
        warning_text = (f"WARNING: Patch index {patch_number}: "
                        f"Maz Azimuth coordinate > {rg_size - 1}. "
                        f"Value received: {coord[3]}")
        print(warning_text)
        if fid_log:
            fid_log.write(warning_text + "\n")

    # extraction of rawdata block of the selected coordinates
    return np.array([fila[min_rg:max_rg+1] 
                     for fila in rawdata[min_az:max_az+1]])


def _gps_utc(gps_time: float) -> str:
    """Convert from GPS time to UTC time"""
    # conversion of GPS time to UTC time (for log annotations)
    t = Time(gps_time, format="gps", scale="utc")
    dt = t.to_datetime(timezone=None)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _load_sensing_times(
        l0_folder: str, 
        polarization: Polarization
) -> np.ndarray:
    """Load the start time of each burst of the raw data file

    This function loads the files "BurstStartTime*" corresponding to the beams
    used in an IW acquisition (IW1, IW2 and IW3). These files contain an array
    with the GPS time of the first pulse of each burst. See package l0_reader
    for more info
    After loading these files, generates an array with the start time of each
    burst of the L0 file and the file name where each burst is stored.

    :parameter
        l0_folder:      string. Folder path where the L0 burst are stored
        polarization:   literal ("vv" or "vh"). Polarization to be processed.
                        Only L0 burst data of the selected polarization will be
                        read
    :return
        sensing_times:  array of SensingType. Array with the start time of each
                        burst and the filename where it is stored
    """
    pattern = os.path.join(l0_folder, f"BurstStartTime_IW*{polarization}.npy")
    file_list = glob.glob(pattern)
    
    sensing_times = np.array([], dtype=SensingType)
    
    for file in file_list:
        times = np.load(file, allow_pickle=True)
        base_name = os.path.basename(file)
        pos = base_name.find("IW")
        if pos == -1:
            continue
        subswath = base_name[pos:pos + 3]
        for i, time in enumerate(times, start=1):
            filename = (f"SentinelData_{subswath}-Burst_{i:02d}-"
                        f"{polarization}.npy")
            line = np.array([(filename, time)], dtype=SensingType)
            sensing_times = np.append(sensing_times, line)

    return np.sort(sensing_times, order="sensingTime")


def _burst_processing(
    polarization: Polarization,
    sensing_times: np.ndarray,
    patch_time: float,
    patches_in_burst: list[tuple[int, int]],
    l0_folder: str,
    output_folder: str,
    patches_base_name: str,
    fid_log: Optional[TextIO] = None,
):
    """ Write all the raw data patches of the same burst.

    This function receives a list with the patches coordinates and the burst
    start time in which the patches are located and write in one file for each
    patch the raw data corresponding to the provided coordinates.

    :parameter
        polarization:   literal ("vv" or "vh"). Polarization to be processed.
                        Only L0 burst data of the selected polarization will be
                        read
        sensing_times:  array of SensingType. Array with the start time of each
                        burst and the filename where it is stored
        patch_time:     float. GPS start time of the burst in which the patches
                        are located
        patches_in_burst: list of patches. The list of the patches with the
                        same patch_time (patches in the same burst)
        l0_folder:      string. Folder path where the L0 burst are stored
        output_folder:  string. Folder path where the generated patches will be
                        stored.
        patches_base_name: string. First part of the output file names. This
                        text will be the same for all patches generated
        fid_log:        TextIO (optional value). File descriptor where the log
                        messages will be written. If None, no log messages are
                        generated
    """
    text = f"\t\tProcessing polarization {polarization.upper()}"
    print(text)
    if fid_log:
        fid_log.write(text + "\n")

    if len(sensing_times) == 0:
        return

    # Select the burst with start time nearest to the info provided with the
    # patch data and read the burst raw data
    pos = np.argmin(np.abs(sensing_times["sensingTime"] - patch_time))
    raw_file = os.path.join(l0_folder, sensing_times[pos]["burstFilename"])
    rawdata = np.load(raw_file)

    for patch in patches_in_burst:
        text = f"\t\t\tProcessing patch {patch[1]}"
        print(text)
        if fid_log:
            fid_log.write(text + "\n")
        patch_filename = (f"{patches_base_name}-IW{patch[0]}-{patch[1]}-"
                          f"{polarization.upper()}.dat")
        patch_rawdata = _extract_patch(rawdata, patch, fid_log)
        _write_l0_patch(patch_rawdata, 
                        os.path.join(output_folder, patch_filename))


def generate_l0_patch(
        l0_folder: str,
        patch_info_file: str,
        output_folder: str,
        patches_base_name: str,
        log_file: str
):
    """Generate raw data files with the size defined in the patches parameters.

    This function receives a file with the information related to the patches
    to be generated (coordinates, ID, ...) and stored in a file the raw data
    section corresponding to the specified coordinates.

    :parameter
        l0_folder:      string. Folder path where the L0 burst are stored
        patch_info_file:string. File name (path included) where the information
                        of each patch is stored. The patch are a list of tuples
                        with the following information
                            [0]: Subswath where the patch is loacted
                            [1]: Patch identification
                            [2]: Patch coordinates in SLC product
                            [3]: Patch coordinates inside the burst
                                [Min echo, Max echo, Min sample, Max sample]
                            [4]: Burst sensing time as provided in L1B
                                annotation file
        output_folder:  string. Folder path where the L0 patches will be stored
        patches_base_name: string. First part of the L0 patches file name.
                        The L0 patch filename will be composed as:
                        <patches_base_name>-<Swath>-<Patch_identification>-<Pol>.dat
        log_file:       string. File name (path included) where the log
                        messages will be written.

    :return
        This function does not return any value. However, it generates L0 files
        por each patch with the following format:
        File format is:
            - Data is stored in Little Endian format
            - Complex data are stored first real part, then imaginary part
            - Complex data are stored in 32-bits float type
            - File header contains the size (rows, cols) in uint32 type

                <Number of pulses (az_size)> in 32-bits unsigned integer
                <Number of samples (rg_size)> in 32-bits unsigned integer
                <real(pulse0,sample0)>
                <imag(pulse0,sample0)>
                ....
                <real(pulse0,rg_size)>
                <imag(pulse0,rg_size)>
                <real(pulse1,sample0)>
                <imag(pulse1,sample0)>
                ....
                <real(pulse1,rg_size)>
                <imag(pulse1,rg_size)>
                ...............
                <real(az_size,rg_size)>
                <imag(az_size,rg_size)>
    """
    # Create log file
    try:
        fid_log = open(log_file, "w")
    except IOError:
        print(f"WARNING: Log file {log_file} cannot be created/opened. "
              f"No log file will be generated")
        fid_log = None

    # Check if input data exist
    if not os.path.isdir(l0_folder):
        error_msg = (f"{datetime.datetime.now().isoformat()} ERROR: "
                     f"Decompressed L0 folder {l0_folder} does not exist. "
                     f"Returning")
        print(error_msg)
        if fid_log:
            fid_log.write(error_msg + "\n")
            fid_log.close()
        return

    if not os.path.isfile(patch_info_file):
        error_msg = (f"{datetime.datetime.now().isoformat()} ERROR: "
                     f"file {patch_info_file} with patches coordinates does "
                     f"not exist. Returning")
        print(error_msg)
        if fid_log:
            fid_log.write(error_msg + "\n")
            fid_log.close()
        return

    # Read start times for each burst of L0 decoded folder
    sensing_times_vv = _load_sensing_times(l0_folder, "vv")
    sensing_times_vh = _load_sensing_times(l0_folder, "vh")
    if (len(sensing_times_vv) == 0) and (len(sensing_times_vh) == 0):
        error_msg = ("ERROR: there must exist at least one file with burst "
                     "start times. Returning")
        print(error_msg)
        if fid_log:
            fid_log.write(error_msg + "\n")
            fid_log.close()
        return


    # Reading patch coordinates and extracting distinct burst start times.
    # This defines the bursts to be processed.
    patch_data = np.load(patch_info_file, allow_pickle=True)
    patches_start_times = np.unique(patch_data[:,4])

    # Process all patches of the same burst
    for patch_time in patches_start_times:
        text = (f"\tProcessing patches with burst start time "
                f"{_gps_utc(patch_time)}")
        print(text)
        if fid_log:
            fid_log.write(text + "\n")

        patches_in_burst = patch_data[patch_data[:,4] == patch_time]
        text = f"\t\tPatches with burst: {len(patches_in_burst)}"
        print(text)
        if fid_log:
            fid_log.write(text + "\n")

        _burst_processing(
            polarization="vv",
            sensing_times=sensing_times_vv,
            patch_time=patch_time,
            patches_in_burst=patches_in_burst,
            l0_folder=l0_folder,
            output_folder=output_folder,
            patches_base_name=patches_base_name,
            fid_log=fid_log
        )
        
        _burst_processing(
            polarization="vh",
            sensing_times=sensing_times_vh,
            patch_time=patch_time,
            patches_in_burst=patches_in_burst,
            l0_folder=l0_folder,
            output_folder=output_folder,
            patches_base_name=patches_base_name,
            fid_log=fid_log
        )

    fid_log.close()


def get_parser(subparsers=None) -> argparse.ArgumentParser:
    """Instantiate the command line argument (sub-)parser."""
    name = PROG
    synopsis = __doc__.splitlines()[0]
    doc = __doc__

    if subparsers is None:
        parser = argparse.ArgumentParser(prog=name, description=doc)
        parser.add_argument(
            "--version",
            action="version",
            version="%(prog)s v" + VERSION
        )
    else:
        parser = subparsers.add_parser(name, description=doc, help=synopsis)


    # Command line options
    parser.add_argument(
       "-o",
       "--output_folder",
       help="output folder where patches will be stored "
       "(if not present, output folder will be current working directory)",
    )
    parser.add_argument(
       "-l",
       "--log_file",
       help="log file name (if not present, the name will be 'Logfile.log' "
            "and located at working directory)",
    )
    parser.add_argument(
       "-b",
       "--base_name",
       help="First part of the L0 patches file name."
                        "The L0 patch filename will be composed as:"
                        "<base_name>-<Swath>-<Patch_identification>-<Pol>.dat "
                        "(if not present, base_name will be 'Patch')",
    )

    # Positional arguments
    parser.add_argument("l0_folder",
                        help="Decoded raw data folder location")
    parser.add_argument("patch_info",
                        help="File with patch info parameters")

    return parser


def parse_args(args=None, namespace=None, parser=None):
    """Parse command line arguments."""
    if parser is None:
        parser = get_parser()

    args = parser.parse_args(args, namespace)
    return args



def main (*argv):
    """Main CLI interface."""

    args = parse_args(argv if argv else None)
    l0_folder = args.l0_folder
    patch_info_file = args.patch_info
    if args.output_folder is None:
        output_folder = "."
    else:
        output_folder = args.output_folder
    if args.log_file is None:
        log_file = "./logfile.log"
    else:
        log_file = args.log_file
    if args.base_name is None:
        patches_base_name = "Patch"
    else:
        patches_base_name = args.base_name

    generate_l0_patch(l0_folder,
                      patch_info_file,
                      output_folder,
                      patches_base_name,
                      log_file
                      )


if __name__ == "__main__":
    sys.exit(main())
