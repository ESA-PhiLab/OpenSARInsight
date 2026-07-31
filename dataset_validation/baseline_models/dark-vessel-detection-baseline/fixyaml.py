import yaml

# Load current YAML
yaml_path = '/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/vessels/aresys_baseline/dvd/yolo_dataset/dataset.yaml'

with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

# Add the missing directory paths
config_fixed = {
    'path': config['path'],
    'train': 'images',  # Tell YOLO to look in images/ subdirectory
    'val': 'images',    # Tell YOLO to look in images/ subdirectory
    'test': 'images',   # Tell YOLO to look in images/ subdirectory
    'nc': config['nc'],
    'names': config['names'],
    # Keep the original split data for reference
    'split': {
        'train': config['train'],
        'val': config['val'], 
        'test': config['test']
    }
}

# Save the corrected YAML
with open(yaml_path, 'w') as f:
    yaml.dump(config_fixed, f, default_flow_style=False)

print("Fixed dataset.yaml - now points to images/ subdirectory")