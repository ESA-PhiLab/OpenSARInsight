![GitHub stars](https://img.shields.io/github/stars/ESA-PhiLab/Repo-Template.svg)
![GitHub forks](https://img.shields.io/github/forks/ESA-PhiLab/Repo-Template.svg)
![GitHub issues](https://img.shields.io/github/issues/ESA-PhiLab/Repo-Template.svg)
![GitHub pull requests](https://img.shields.io/github/issues-pr/ESA-PhiLab/Repo-Template.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/ESA-PhiLab/Repo-Template.svg)
![GitHub code size](https://img.shields.io/github/languages/code-size/ESA-PhiLab/Repo-Template.svg)
![GitHub top language](https://img.shields.io/github/languages/top/ESA-PhiLab/Repo-Template.svg)
![GitHub repo size](https://img.shields.io/github/repo-size/ESA-PhiLab/Repo-Template.svg)
![GitHub contributors](https://img.shields.io/github/contributors/ESA-PhiLab/Repo-Template.svg)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://img.shields.io/badge/docs-latest-green.svg)](https://github.com/ESA-PhiLab/Repo-Template/wiki)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://makeapullrequest.com)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15332053.svg)](https://doi.org/10.5281/zenodo.15332053)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

(For these shield to work, set proper name..)


<p align="right">
<img src="./docs/images/banner.png" alt="ESA-PhiLab Logo" width="100%" style='margin-bottom: 0em;'/>
</p>

<p align="center" style="font-size:1.5em;">
    <strong>🌍 TRANSFORMING EARTH OBSERVATION IN ACTIONS FOR THE HUMAN PROSPERITY 🌍</strong>
</p>

<div align="center" style="font-size:1.2em;">

[![Website](https://img.shields.io/badge/Website-philab.esa.int-blue?style=for-the-badge)](https://philab.esa.int)
[![Twitter](https://img.shields.io/badge/Twitter-ESA_EO-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/ESA_EO)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-ESA_PhiLab-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/groups/8984375/)
[![YouTube](https://img.shields.io/badge/YouTube-ESA-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/channel/UCIBaDdAbGlFDeS33shmlD0A)

</div>


## Repository Template


Use this template to kickstart your ESA-PhiLab project with all the necessary structure and documentation. You'll get:
- A professional repository structure following best practices
- Automatic GitHub Pages website generation (example: https://esa-philab.github.io/Repo-Template/)
- Pre-configured badges and documentation templates
- Standardized license and citation formats

Getting started is easy - just follow the detailed instructions in `docs/readme.md` to customize the template for your specific project.

## 📌 Project Title
> Replace this with your project name.

## 👥 Authors
List all project contributors. Preferably include links to their professional profiles.

- Name One ([profile](https://example.com))  
- Name Two ([profile](https://example.com))

## 📖 Project Reference
Specify the type and context of the project.

- Example: ESA Φ-lab Research Fellowship — [Link to ESA page](https://philab.esa.int)

## 📝 Abstract
Provide a concise summary of the project goals, methodology, and outcomes.

## 🛠️ How to Use
Detailed instructions to set up and run the project.

### Quick Start

**1. Setup the development environment:**
```bash
./scripts/setup.sh
```

**2. Activate the virtual environment:**
```bash
source venv/bin/activate
```

**3. Download datasets:**
```bash
./scripts/download_data.sh
```

**4. Run tests and build:**
```bash
./scripts/build.sh
```

### Requirements
This repository uses a modern Python environment standard:
- Python ≥ 3.9
- Dependencies are specified in the `pyproject.toml` file.

**Manual installation:**
```bash
pip install .
```
Or using PDM:
```bash
pdm install
```

### Available Scripts

**Bash Scripts (in `scripts/`):**
- `setup.sh` - Sets up the development environment
- `build.sh` - Builds, tests, and packages the project
- `download_data.sh` - Downloads and processes datasets
- `cleanup.sh` - Removes build artifacts and cache files

**Python Scripts (in `pyscripts/`):**
- `download_hf_datasets.py` - Downloads from Hugging Face Hub
- `process_data.py` - Data processing and analysis utilities
- `example_script.py` - Example Python utility script

## 📚 Citations
Please cite the following works if you use this code:
- [Reference Paper 1]
- [Reference Paper 2]

## 📂 Repository Structure
```
├── LICENSE
├── README.md
├── pyproject.toml      # project metadata & dependencies
├── setup.cfg           # optional packaging config
├── environment.yml     # optional environment specification
├── src/                # source code & Python package
│   └── your_package/   # replace with your package name
├── notebooks/          # Jupyter notebooks for experiments or analysis
├── papers/             # manuscript sources (LaTeX, figures, assets)
├── data/               # raw/processed datasets or external links
├── scripts/            # bash/shell scripts for automation
├── pyscripts/          # Python scripts and utilities
├── tests/              # unit and integration tests
├── docs/               # documentation (Sphinx, MkDocs, GitHub Pages)
│   ├── index.html      # main documentation page
│   ├── images/         # documentation assets (images,..)
└── examples/           # usage examples and demos
```

The `docs/` directory contains the project's documentation website with comprehensive information about the project, including:
- Project overview and goals
- Installation and setup instructions
- Usage examples and tutorials
- API documentation
- Links to related resources and publications
- Team and contributor information


## 📄 License
This project is licensed under the **Apache License 2.0**. See the [LICENSE](./LICENSE) file or read more at [apache.org](https://www.apache.org/licenses/LICENSE-2.0).

## 📊 Dataset Hosting
Datasets are either included in `data/` or hosted externally:
- [Zenodo](https://zenodo.org)
- [Hugging Face Datasets](https://huggingface.co/datasets)

Ensure external links are functional and persistent.

### 📥 Data Download Utilities

This repository includes utilities for downloading datasets:

**Python Script (Hugging Face Hub):**
```bash
# Install required dependencies
pip install huggingface_hub

# Run the download script
python pyscripts/download_hf_datasets.py
```

**Bash Script (General Data Download):**
```bash
# Download and process datasets
./scripts/download_data.sh

# Force re-download and verbose output
./scripts/download_data.sh --force --verbose
```

The Python script (`pyscripts/download_hf_datasets.py`) provides a robust way to fetch datasets from Hugging Face Hub with automatic retries and error handling. The bash script (`scripts/download_data.sh`) handles general dataset downloads and preprocessing with the following features:

- Downloads datasets from various sources
- Configurable retry mechanism for handling network issues
- Progress tracking and error reporting
- Data validation and preprocessing
- Automatic directory organization

To customize which datasets to download, edit the `repo_ids` list in the script.

## 🌐 GitHub Pages
This repository supports GitHub Pages. Request support to enable and configure the page if needed.

---

*For internal use only: This repository conforms to the ESA-PhiLab development and archiving standards.*
