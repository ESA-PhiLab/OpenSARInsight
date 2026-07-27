# Data

This folder is intended to store raw and processed datasets, or links to external datasets relevant to the project. 

## Structure
- `raw/` - Raw downloaded datasets
- `processed/` - Cleaned and preprocessed data
- `external/` - Links to external datasets
- `interim/` - Intermediate processing results

## Examples
- Raw data files (e.g., CSV, JSON, images)
- Processed data files (e.g., cleaned data, feature-engineered data)
- Text files with links to datasets hosted on platforms like Zenodo or Hugging Face Datasets

## Data Download

Use the provided scripts to download datasets:

**General datasets:**
```bash
./scripts/download_data.sh
```

**Hugging Face datasets:**
```bash
python pyscripts/download_hf_datasets.py
```

Both scripts will organize downloaded data into the appropriate subdirectories.
