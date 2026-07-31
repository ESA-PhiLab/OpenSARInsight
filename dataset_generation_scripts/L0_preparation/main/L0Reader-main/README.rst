Sentinel-1 Instrument Source Packets decoder
============================================

Description
-----------

`L0Reader` is Python tool to decode Sentinel-1 Instrument Source Packets (ISPs)
contained in the RAW data files included in the Sentinel-1 L0 products.
It generates, for each burst in the ISP file, a file with the header of each pulse
in a structure following the Header definition; a file with the rawdata decoded and
in complex form and a file with the decoding parameters (BRC and THIDX) used.

The relevant specification document used to write the `L0Reader` software is:

* S1-IF-ASD-PL-0007_, "Sentinel-1 SAR Space Packet Protocol Data Unit", Issue 13
   https://sentinels.copernicus.eu/documents/247904/2142675/Sentinel-1-SAR-Space-Packet-Protocol-Data-Unit.pdf


Requirements and Installation
-----------------------------

The package requires numpy. This dependency is
managed automatically by `setuptools` and the `pyproject.toml` file.

To install the package, use the following command:

    $ python -m pip install .

For editable mode, use:

    $ python -m pip install --editable .


Command line interface (CLI)
----------------------------

The package has a simple CLI::

    $ python -m L0Reader -h
		usage: L0Reader [-h] [--version] [-o OUTPUT_FOLDER] [-l LOG_FILE] filename

		Sentinel-1 rawdata reader and decoder

		positional arguments:
		  filename              RAW data file name

		options:
		  -h, --help            show this help message and exit
		  --version             show program's version number and exit
		  -o, --output_folder OUTPUT_FOLDER
								output folder where rawdata bursts will be stored (if not present, output folder will be
								current working directory)
		  -l, --log_file LOG_FILE
								log file name (if not present, the name will be 'Logfile.log' and located at working directory


License
-------

Copyright (c) 2025 Marcos Garcia <garciarm@inta.es>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
