import tempfile
import os
from pathlib import Path
from utilities.inflate_label import inflate_box, process_label

def test_inflate_box_basic():
	# Box in the center, image size 100x100, pad 10px
	x_c, y_c, w, h = 0.5, 0.5, 0.2, 0.2
	pad_x = 10 / 100
	pad_y = 10 / 100
	x_c_new, y_c_new, w_new, h_new = inflate_box(x_c, y_c, w, h, pad_x, pad_y)
	assert abs(w_new - (0.2 + 0.2)) < 1e-6  # 2*pad_x added
	assert abs(h_new - (0.2 + 0.2)) < 1e-6
	assert 0 <= x_c_new <= 1
	assert 0 <= y_c_new <= 1

def test_inflate_box_clamping():
	# Box at the edge, padding should clamp
	x_c, y_c, w, h = 0.05, 0.05, 0.1, 0.1
	pad_x = 0.1
	pad_y = 0.1
	x_c_new, y_c_new, w_new, h_new = inflate_box(x_c, y_c, w, h, pad_x, pad_y)
	assert w_new <= 1.0
	assert h_new <= 1.0
	assert 0 <= x_c_new <= 1
	assert 0 <= y_c_new <= 1

def test_process_label(tmp_path):
	# Create a dummy label file and test inflation
	label_src = tmp_path / "label.txt"
	label_dst = tmp_path / "label_out.txt"
	img_w, img_h = 100, 100
	padding = 10
	# YOLO: class x_center y_center width height
	label_src.write_text("0 0.5 0.5 0.2 0.2\n")
	process_label(label_src, label_dst, img_w, img_h, padding)
	out = label_dst.read_text().strip().split()
	# Should be class + 4 floats
	assert out[0] == "0"
	assert len(out) == 5
	# Check that width and height increased
	assert float(out[3]) > 0.2
	assert float(out[4]) > 0.2
