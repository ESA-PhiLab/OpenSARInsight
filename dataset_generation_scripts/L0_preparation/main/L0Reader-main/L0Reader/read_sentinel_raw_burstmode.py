"""Sentinel-1 rawdata reader and decoder"""


import os
import sys
import datetime
import time
import argparse
import logging
import numpy as np
from pathlib import Path
from .decoding import decode_data
from .process_sentinel_header import process_sentinel_header
from .Huffman import *

from . import __version__

PROG = __package__

def read_sentinel_raw_burstmode(raw_data_file, output_dir, logger:logging.Logger):
    """Read and decode Sentinel-1 rawdata files

    This function receives the name of a rawdata file from Sentinel-1, reads
    each pulse, decodes the header and the data and stores the results,
    grouped by burst, in the output folder specified as input.

    :parameter
        raw_data_file (String): Name, path included, of the file to be read
        output_dir (String): Folder where the results will be stored
        logger (logging.Logger): python logger

    :return
        NONE

    :additional results
        For each burst read and decoded three files will be generated:
            SentinelData_ : File with the burst data decoded
            SentinelHeader_: File with the header dictionary for each pulse
            SentinelAuxData_: File with the parameters used in the decoding
                              process (BRC and THIDX)

            Naming convention
                SentinelData_<swath>-Burst_<burstNumber>-<pol>.npy
                SentinelHeader_<swath>-Burst_<burstNumber>-<pol>.npy
                SentinelAuxData_<swath>-Burst_<burstNumber>-<pol>.npy
                    <swath>: any of the swath defined in the swathMapList tag
                             for the IW mode in the Auxiliary Instrument file
                    <burstNumber>: consecutive number (starting 1) for each
                             burst of the same <swath>
                    <pol>: vv or vh, based in rawdata file name

        For each swath
            BurstStartTime_: File with the time of first pulse of each burst

            Naming convention:
                BurstStartTime_<swath>-<pol>.npy
    """
    header = []
    swath_id = None

    pulse_counter = 0

    if not os.path.isfile(raw_data_file):
        error_msg = (f"{datetime.datetime.now().isoformat()} ERROR: "
                     f"RawData file {raw_data_file} does not exist. Returning")
        logger.error(error_msg)
        return

    try:
        fid_raw = open(raw_data_file, 'rb')
    except IOError:
        error_msg = (f"{datetime.datetime.now().isoformat()} ERROR: "
                     f"RawData file {raw_data_file} cannot be opened. Returning")
        logger.error(error_msg)
        return

    file_name = os.path.basename(raw_data_file)

    pol = ""
    if "-vh-" in file_name:
        pol = "-vh"
    elif "-vv-" in file_name:
        pol = "-vv"

    # From instrument aux file
    swath_values = [10, 60, 43, 93, 11, 61, 44, 94, 12, 62, 45, 95]
    swath_test_list = ["IW1", "CAL1-IW1", "CAL2-IW1", "CAL3-IW1", "IW2",
                       "CAL1-IW2", "CAL2-IW2", "CAL3-IW2", "IW3", "CAL1-IW3",
                       "CAL2-IW3", "CAL3-IW3"]


    # List of 12 elements, one for each swath previously defined.
    # Each element stores a list with the start time of first pulse of each burst
    sensing_times_list = [[] for _ in range(12)]

    # Number of last burst processed for each swath type
    # (to be used in the filename to be writen)
    burst_count = np.ones(12, dtype=int)

    # List with the start time of first pulse of each burst,
    # for burst coded with value not defined in "swath_values"
    # In principle it must not be possible
    undefined_sensing_times = []

    # Number of last burst processed for burst coded with value not
    # defined in "swath_values"
    undefined_count = 1


    cont_pulses = 0

    while True:
        header_data = np.fromfile(fid_raw, dtype=np.uint8, count=68)
        if len(header_data) < 68:  # End of File is reached.
            break
        
        header_temp, error = process_sentinel_header(header_data)
        if error != 0:
            fid_raw.close()
            return
        
        num_data = header_temp['PrimaryHeader']['PacketDataLength'] + 1 - 62
        swath_temp = header_temp['RadarConfiguration']['SES_SSB']['SwathNumber']
        
        if swath_id is None or swath_temp != swath_id:
            #New burst is detected. Files of previous burst will be saved
            if swath_id is not None:
                # Not first pulse. It is really a burst change
                pos = (swath_values.index(swath_id)
                       if swath_id in swath_values else None)
                swath_test = (swath_test_list[pos]
                              if pos is not None else "Undefined")
                burst_num = (burst_count[pos]
                             if pos is not None else undefined_count)
                if pos is not None:
                    burst_count[pos] += 1
                    sensing_times_list[pos].append(
                        header[0]['TimeCode']['CoarseTime']
                        + header[0]['TimeCode']['FineTime'])
                else:
                    undefined_count += 1
                    undefined_sensing_times.append(
                        header[0]['TimeCode']['CoarseTime']
                        + header[0]['TimeCode']['FineTime'])

                # Save files with data of previous burst:
                burst_file = os.path.join(
                    output_dir,
                    f"SentinelHeader_{swath_test}-Burst_{burst_num:02d}{pol}")
                np.save(burst_file, header)
                logger.debug(f"Saved header to SentinelHeader_{swath_test}-Burst_{burst_num:02d}{pol}.npy")

            # Initialization of parameters for the new burst
            swath_id = swath_temp
            header = []
            cont_pulses = 0

        # Reading the data of the pulse
        data_temp = np.fromfile(fid_raw, dtype=np.uint8, count=num_data) #causing problems can't be commented out!

        # Add the data of the pulse to the burst data
        header.append(header_temp)
        cont_pulses += 1

        pulse_counter += 1
    fid_raw.close()


    # Saving last burst data. Same code as previous one
    pos = swath_values.index(swath_id) if swath_id in swath_values else None
    swath_test = swath_test_list[pos] if pos is not None else "Undefined"
    burst_num = burst_count[pos] if pos is not None else undefined_count
    if pos is not None:
        burst_count[pos] += 1
        sensing_times_list[pos].append(
            header[0]['TimeCode']['CoarseTime']
            + header[0]['TimeCode']['FineTime'])
    else:
        undefined_count += 1
        undefined_sensing_times.append(
            header[0]['TimeCode']['CoarseTime']
            + header[0]['TimeCode']['FineTime'])
    burst_file = os.path.join(
        output_dir,
        f"SentinelHeader_{swath_test}-Burst_{burst_num:02d}{pol}")
    np.save(burst_file, header)
    logger.debug(f"Saved header to SentinelHeader_{swath_test}-Burst_{burst_num:02d}{pol}.npy")

    # Saving the start times of decoded burst, for each swath type
    for i in range(0, len(swath_values)):
        if len(sensing_times_list[i]) > 0:
            burst_time_file = os.path.join(
                output_dir,f"BurstStartTime_{swath_test_list[i]}{pol}")
            np.save(burst_time_file, sensing_times_list[i])
            logger.debug(f"Saved burst times to BurstStartTime_{swath_test_list[i]}{pol}.npy")
    # Saving the start sensing times of burst with swath index not defined
    # in the swath type list
    if len(undefined_sensing_times) > 0:
        burst_time_file = os.path.join(
            output_dir, f"BurstStartTime_Undefined{pol}")
        np.save(burst_time_file, undefined_sensing_times)
        logger.debug(f"Saved burst times to BurstStartTime_Undefined{pol}.npy")

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
            version="%(prog)s v" + __version__
        )
    else:
        parser = subparsers.add_parser(name, description=doc, help=synopsis)

    # Command line options
    parser.add_argument(
       "-o",
       "--output_folder",
       help="output folder where rawdata bursts will be stored "
       "(if not present, output folder will be current working directory)",
    )
    parser.add_argument(
       "-l",
       "--log_file",
       help="log file name (if not present, the name will be "
            "'Logfile.log' and located at working directory",
    )

    # Positional arguments
    parser.add_argument("filename", help="RAW data file name")

    return parser


def parse_args(args=None, namespace=None, parser=None):
    """Parse command line arguments."""
    if parser is None:
        parser = get_parser()

    args = parser.parse_args(args, namespace)

    return args


def main (*argv):
    """Main CLI interface."""
    logger = logging.getLogger("log")
    logger.level = logging.INFO
    file_handler = None
    start_time = time.time()
    args = parse_args(argv if argv else None)
    raw_data_file = args.filename
    if args.output_folder is None:
        output_dir = "."
    else:
        output_dir = args.output_folder
    if args.log_file is None:
        file_handler = logging.FileHandler(Path(__file__).parent.joinpath("logfile.log"))
    else:
        file_handler = logging.FileHandler(Path(__file__).parent.joinpath(args.log_file))
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    read_sentinel_raw_burstmode(raw_data_file, output_dir, logger)

    logger.info(f"Processed {raw_data_file} in  {time.time() - start_time:.2f} s")

    return 0


if __name__ == "__main__":
    sys.exit(main())

