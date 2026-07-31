import tempfile
import os
from pathlib import Path
from utilities.create_data_yaml import create_data_yaml

def test_create_data_yaml(tmp_path):
	# Simulate a dataset path
	dataset_path = tmp_path / "yolo_dataset"
	dataset_path.mkdir()
	class_names = ["vessel", "boat"]

	# Change working directory to tmp_path to avoid polluting project root
	old_cwd = os.getcwd()
	os.chdir(tmp_path)
	try:
		create_data_yaml(dataset_path, class_names)
		data_yaml = Path("data.yaml")
		assert data_yaml.exists()
		content = data_yaml.read_text()
		assert f"path: {dataset_path}" in content
		assert f"nc: {len(class_names)}" in content
		assert f"names: {class_names}" in content
	finally:
		os.chdir(old_cwd)
