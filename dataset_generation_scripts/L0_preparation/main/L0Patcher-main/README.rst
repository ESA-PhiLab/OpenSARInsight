l0 Patch Generator for TDS of project OPENSAR-INSIGHT
=====================================================

Description
-----------

`L0Patcher` is Python tool to to generate the raw data patches equivalent to the
L1B SSC patches used in the OPNESAR-INSIGHT project.
The tool receives as inpujt a file with the pathes information (pixel coordinates,
patch ID, ....) and generates, for each patch, an output file with the raw data block
corresponding to the coordinates provided.


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

    $ python -m L0Patcher -h
		usage: L0Patcher [-h] [--version] [-o OUTPUT_FOLDER] [-l LOG_FILE] [-b BASE_NAME] l0_folder patch_info

		l0 Patch Generator for TDS of project OPENSAR-INSIGHT

		positional arguments:
		  l0_folder             Decoded raw data folder location
		  patch_info            File with patch info parameters

		options:
		  -h, --help            show this help message and exit
		  --version             show program's version number and exit
		  -o, --output_folder OUTPUT_FOLDER
								output folder where patches will be stored (if not present, output folder will be current
								working directory)
		  -l, --log_file LOG_FILE
								log file name (if not present, the name will be 'Logfile.log' and located at working
								directory)
		  -b, --base_name BASE_NAME
								First part of the L0 patches file name.The L0 patch filename will be composed
								as:<base_name>-<Swath>-<Patch_identification>-<Pol>.dat (if not present, base_name will be
								'Patch')

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
