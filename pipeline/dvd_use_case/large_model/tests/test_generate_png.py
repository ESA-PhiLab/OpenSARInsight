
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
from scripts.generate_png import to_uint8, generate_pngs

def test_to_uint8_basic():
	arr = np.array([0.0, 0.5, 1.0, -1.0, 2.0])
	out = to_uint8(arr)
	assert out.dtype == np.uint8
	assert out[0] == 0
	assert out[2] == 255
	assert out[1] == 127 or out[1] == 128  # rounding
	assert out[3] == 0  # clipped
	assert out[4] == 255  # clipped

def test_generate_pngs_from_rc_creates_png():
	# Create dummy .npy file in the new unified structure: dataset_root/train/range_compressed_rescaled/
	with tempfile.TemporaryDirectory() as tmpdir:
		rc_root = Path(tmpdir) / "train" / "range_compressed_rescaled"
		rc_root.mkdir(parents=True)
		arr = (np.random.randn(16, 16) + 1j * np.random.randn(16, 16)).astype(np.complex64)
		# Name must contain -vv- for VV polarization
		npy_path = rc_root / "patch-vd-1-vv-001.npy"
		np.save(npy_path, arr)
		png_root = Path(tmpdir) / "pngs"
		generate_pngs(Path(tmpdir), png_root)
		# Output file should be created in png_root/train/
		out_dir = png_root / "train"
		pngs = list(out_dir.glob("*.png"))
		assert len(pngs) == 1
		img = Image.open(pngs[0])
		assert img.mode == "RGB"
		assert img.size == (16, 16)
