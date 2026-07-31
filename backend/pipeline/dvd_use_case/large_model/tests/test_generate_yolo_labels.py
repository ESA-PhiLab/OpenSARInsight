
import tempfile
from pathlib import Path
import shutil
import types
import sys
import builtins
import pytest

import scripts.generate_yolo_labels as gyrc

def fake_parse_vessel_xml(xml_file, bbox_tag=None):
	# Always return one ship with a bbox
	return {
		'number_of_ships': 1,
		'ships': [{'bbox': [10, 20, 30, 40], 'name': 'ship1'}],
		'skipped_ships': []
	}

def fake_filter_ships(ships, **kwargs):
	# Pass through ships unchanged
	return ships

def fake_bbox_to_pose_line(bbox, class_id, img_width, img_height):
	# Return a dummy YOLO label line
	return f"{class_id} 0.5 0.5 0.1 0.1"

def fake_create_data_yaml(yolo_dataset_path):
	# Do nothing
	return None

def test_generate_yolo_labels_creates_labels(monkeypatch):
	# Patch dependencies
	monkeypatch.setattr(gyrc, 'parse_vessel_xml', fake_parse_vessel_xml)
	monkeypatch.setattr(gyrc, 'filter_ships', fake_filter_ships)
	monkeypatch.setattr(gyrc, 'bbox_to_pose_line', fake_bbox_to_pose_line)
	monkeypatch.setattr(gyrc, 'create_data_yaml', fake_create_data_yaml)

	with tempfile.TemporaryDirectory() as tmpdir:
		tmpdir = Path(tmpdir)
		# Create minimal dataset structure matching unified layout
		# dataset_root/train/labels/*.xml
		xml_dir = tmpdir / 'dataset' / 'train' / 'labels'
		png_dir = tmpdir / 'pngs' / 'train'
		yolo_dir = tmpdir / 'yolo'
		xml_dir.mkdir(parents=True)
		png_dir.mkdir(parents=True)
		yolo_dir.mkdir(parents=True)
		# Create one XML and one PNG file
		(xml_dir / 'VD_001.xml').write_text('<xml></xml>')
		(png_dir / 'VV_VD_001.png').write_bytes(b'\x89PNG\r\n\x1a\n')

		stats = gyrc.generate_yolo_labels(
			dataset_root=tmpdir / 'dataset',
			png_rc_patches_path=tmpdir / 'pngs',
			yolo_dataset_path=yolo_dir,
			img_width=512,
			img_height=512,
			class_id=0
		)
		# Check that a label file was created
		label_file = yolo_dir / 'labels' / 'train' / 'VV_VD_001.txt'
		assert label_file.exists()
		content = label_file.read_text().strip()
		assert content == '0 0.5 0.5 0.1 0.1'
		# Check stats
		assert stats['total_images'] == 1
		assert stats['images_with_detections'] == 1
