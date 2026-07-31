Trace coordinates from SLC to RAW for OPENSAR-INSIGHT
=====================================================

Description
-----------

`slc2raw` is Python tool to translate the SLC coordinates of the L1B patch to the
required coordinates of the L0 data th generate a L0 patch equivalent to the L1B ones.
This L0 patch, after processing, will generates a L1B product equivalent to the original
L1B SLC Patch.
This L0 Patch will be used in the OPENSAR-INSIGHT project.
The tool receives as input a folder where the patches information (pixel coordinates,
patch ID, ....) will be stored (one file per patch) and generates an output file with the
corresponding L0 coordinates for each patch.
Output of this tool will be used as input in the L0Patcher tool


Requirements and Installation
-----------------------------

The package requires numpy and astropy. These dependencies are
managed automatically by `setuptools` and the `pyproject.toml` file.

To install the package, use the following command:

    $ python -m pip install .

For editable mode, use:

    $ python -m pip install --editable .


Command line interface (CLI)
----------------------------

The package has a simple CLI::

    $ python -m slc2raw -h
		usage: slc2raw [-h] [--version] [-o OUTPUT_FILE] [-l LOG_FILE] label_folder product_folder

		Trace coordinates from SLC to RAW

		positional arguments:
		  label_folder          Folder where the patch labels (one file per patch) are located
		  product_folder        Folder where the L1B SLC product is located

		options:
		  -h, --help            show this help message and exit
		  --version             show program's version number and exit
		  -o, --output_file OUTPUT_FILE
								output file with the L0 coordinates after traceability (if not present, output file will be
								'rawH.npy' in current folder)
		  -l, --log_file LOG_FILE
								log file name (if not present, the name will be 'traceability_log.txt' and located at working
								directory)

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
