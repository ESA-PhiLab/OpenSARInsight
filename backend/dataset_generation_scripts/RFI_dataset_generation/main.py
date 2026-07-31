import os
import io
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

import numpy as np
import tifffile
import rasterio
from PIL import Image, ImageFile
from pyproj import Geod, Transformer
import xml.etree.ElementTree as ET
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
"""
RFI Tiling & Labeling Pipeline for Sentinel-1 SLC/GRD Products
==============================================================

Overview
--------
This module implements an end-to-end pipeline that:
  1) Parses Sentinel-1 SLC product metadata (manifest, annotations, calibration) and GRD product metadata.
  2) Reads raw SLC bursts (VV and VH) interleaved as int16 (real/imag) and extracts square complex tiles per burst.
  3) Interpolates geolocation (lon/lat) for the 4 tile corners using tiepoints and a geodesic model.
  4) Aligns each SLC tile to a binary quicklook mask to assign an RFI presence flag (1/0).
  5) Writes, for every tile:
       - Complex SLC VV and VH tiles as single-band complex64 TIFF files.
       - A mask patch (PNG) aligned to the tile footprint.
       - An XML label including SAR metadata and the tile’s corner coordinates in lon/lat.
  6) Fits polynomial (least-squares) transforms from GRD pixel space to lon/lat using GeoTIFF tiepoints,
     back-projects tile polygons to GRD pixel coordinates, and saves the corresponding GRD tiles (VV & VH).
  7) Produces a legend CSV (TXT) listing (patch_id, has_ship, swath_id) for downstream training/analysis.

Intended Use
------------
The output dataset (complex SLC tiles + GRD tiles + labels + binary mask) is designed for
RFI detection experimentation, model training, or quality control where consistent georeferencing
between SLC and GRD domains is required.

Inputs (Directory Layout)
-------------------------
The function `main(inputpath, grdproduct, slcproduct, outputpath)` expects:

- inputpath/
    ├── SLC_product/
    │   └── {slcproduct}/
    │       ├── manifest.safe
    │       ├── <raw SLC .tiff files (VV/VH)>
    │       ├── <calibration XMLs (VV/VH)>
    │       └── <annotation XMLs (VV/VH)>
    ├── GRD_product/
    │   └── {grdproduct}/
    │       ├── manifest.safe
    │       ├── <GRD image VV>
    │       └── <GRD image VH>
    └── binarymask/
        └── {slcproduct}.png     # binary quicklook mask (white/colored = 1, black = 0)

Key Assumptions
---------------
- SLC bursts are stored as interleaved int16 [real, imag] per sample in strips (not tiles).
- Tiepoints (GeoTIFF ModelTiepointTag) are available and sufficient to interpolate lon/lat at tile corners.
- Binary mask and SLC geometry are coarsely aligned; the code rescales SLC geometry to the mask resolution.
- GRD products expose tiepoints to fit a polynomial transform (pixel -> lon/lat) used to cut GRD tiles.
- File references in SAFE manifests are resolved via `href_to_path` (Windows-style by default).

Outputs
-------
The pipeline creates subfolders under `outputpath/`:

- outputpath/
    ├── slc/      # complex patches from SLC
    │   ├── DB_OPENSAR_RFI_{id}_SLC_VV.tiff  (complex64)
    │   └── DB_OPENSAR_RFI_{id}_SLC_VH.tiff  (complex64)
    ├── mask/      # mask patches
    │   └── DB_OPENSAR_RFI_{id}_MASK.png
    ├── grd/     # GRD tiles
    │   ├── DB_OPENSAR_RFI_{id}_GRD_VV.tiff
    │   └── DB_OPENSAR_RFI_{id}_GRD_VH.tiff
    └── label/   # per-tile XML label and legend
        ├── DB_OPENSAR_RFI_{id}.xml          # includes corner lon/lat and SAR metadata
        └── legend{XXXX}.txt                  # CSV (patch_id,has_ship,swath_id)

Processing Highlights
---------------------
- SLC VV/VH complex tiles:
    * The tile size equals the number of lines per burst (square tiles).
    * Corner (sample,line) indices are mapped to lon/lat using tiepoint interpolation
      with `LinearNDInterpolator` + nearest-neighbor fallback.
    * Tile longitude is normalized to [-180, 180) to avoid wrap-around issues.
- Mask-based labeling:
    * The SLC geometry is rescaled to match the mask image; if any mask pixel is > 0 within the
      mapped region, the tile gets RFIDetection=1; otherwise 0.
- GRD back-projection:
    * A 2D polynomial (total degree) geotransform is fitted from GRD pixel (I,J) to lon/lat
      using GeoTIFF tiepoints, and inverted to cut VV/VH GRD tiles matching the SLC tile footprint.


Notes
-----
- For cross-platform paths, consider replacing `href_to_path` with `os.path.normpath`.
- If a given SLC file is stored as tiles (TileOffsets), a different read strategy is required.
- Complex TIFFs are saved as complex64 (GDAL GDT_CFloat32-compatible via tifffile).
"""
# Allow very large images and truncated streams (common in large SAR products)
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Global WGS84 ellipsoid model for geodesic computations
_geod = Geod(ellps='WGS84')


# ------------------------------ Utilities ------------------------------ #

def fmt4(x):
    """
    Format a floating-point number with fixed precision (10 decimals),
    using "round half up" to avoid banker's rounding issues.

    Args:
        x (float): Input value.

    Returns:
        str: Decimal-formatted string with 10 fractional digits.
    """
    return str(Decimal(x).quantize(Decimal('0.0000000001'), rounding=ROUND_HALF_UP))


def _safe_tag_name(name: str) -> str:
    """
    Sanitize XML tag/attribute names so they can be used as dictionary keys.
    Replaces '-', ':', '.' with underscore.

    Args:
        name (str): Input tag name.

    Returns:
        str: Sanitized tag name.
    """
    return name.replace('-', '_').replace(':', '_').replace('.', '_')


def xml2struct(file):
    """
    Parse an XML file (or Element) into a nested Python dictionary with a
    Matlab-like structure style.

    The structure for each element may contain:
      - Child elements as keys
      - 'Text': element text if present
      - 'Attributes': attribute dict if present

    Args:
        file (str or xml.etree.ElementTree.Element): Path to XML file, or an already-parsed Element.

    Returns:
        dict: Nested dict representing the XML tree.
    """
    if isinstance(file, ET.Element):
        root = file
    else:
        if not os.path.exists(file):
            # If no extension, try appending '.xml'
            if not file.lower().endswith('.xml') and os.path.exists(file + '.xml'):
                file = file + '.xml'
            else:
                raise FileNotFoundError(f'The file: {file} does not exist')
        root = ET.parse(file).getroot()

    def parse_attributes(elem):
        attrs = {}
        for k, v in elem.attrib.items():
            attrs[_safe_tag_name(k)] = v
        return attrs

    def parse_children(elem):
        """
        Recursively parse children into a structure that merges:
        - children dict
        - accumulated text when an element has only text (no children)
        """
        children = {}
        ptext = {}
        for child in list(elem):
            name = _safe_tag_name(child.tag)
            text = (child.text or '').strip()
            attr = parse_attributes(child)
            childs, textdict = parse_children(child)

            if name not in ['#text', '#comment']:
                node = {}
                if childs:
                    node.update(childs)
                if text:
                    node['Text'] = text
                if attr:
                    node['Attributes'] = attr

                # Convert repeated tags into lists
                if name in children:
                    if not isinstance(children[name], list):
                        children[name] = [children[name]]
                    children[name].append(node)
                else:
                    children[name] = node
            else:
                if text:
                    ptext['Text'] = ptext.get('Text', '') + text

        txt = (elem.text or '').strip()
        if not children and txt:
            ptext['Text'] = txt
        return children, ptext

    children, ptext = parse_children(root)
    s = {}
    root_name = _safe_tag_name(root.tag)
    s[root_name] = children if children else {}
    if ptext:
        s[root_name].update(ptext)
    attrs = parse_attributes(root)
    if attrs:
        s[root_name]['Attributes'] = attrs
    return s


def struct2xml(s, file):
    """
    Serialize a Matlab-like struct (single root dict) to an XML file.

    The input must be a dict with a single root key. Inside each node:
      - 'Attributes' dict is written as XML attributes
      - 'Text' is written as element text
      - Any other key becomes a nested element. Lists create repeated tags.

    Parameters
    ----------
    s : dict
        Structure with a single root.
    file : str
        Output path ('.xml' appended if missing).
    """
    if not isinstance(s, dict) or len(s.keys()) != 1:
        raise ValueError("It shall be only one field on the input structure")

    root_name = list(s.keys())[0]
    root_elem = ET.Element(root_name)

    def build(elem, dct):
        attrs = dct.pop('Attributes', None)
        if isinstance(attrs, dict):
            for ak, av in attrs.items():
                elem.set(ak, str(av))

        txt = dct.pop('Text', None)
        if txt is not None:
            elem.text = str(txt)

        for k, v in dct.items():
            if isinstance(v, list):
                for item in v:
                    child = ET.SubElement(elem, k)
                    if isinstance(item, dict):
                        build(child, item.copy())
                    else:
                        child.text = str(item)
            elif isinstance(v, dict):
                child = ET.SubElement(elem, k)
                build(child, v.copy())
            else:
                child = ET.SubElement(elem, k)
                child.text = str(v)

    build(root_elem, s[root_name].copy())
    tree = ET.ElementTree(root_elem)

    if not str(file).lower().endswith('.xml'):
        file = str(file) + '.xml'
    tree.write(file, encoding='utf-8', xml_declaration=True)


def save_complex_patch_tiff(out_path: str, complex_patch: np.ndarray, description: Optional[str] = None):
    """
    Save a complex patch (complex64) as single-band TIFF.

    Args:
        out_path (str): Output TIFF path.
        complex_patch (np.ndarray): 2D complex array.
        description (Optional[str]): TIFF description tag content.

    Notes:
        - Uses tifffile to write a band with dtype complex64.
        - Compatible with GDAL as GDT_CFloat32.
    """
    arr = np.asarray(complex_patch, dtype=np.complex64)
    tifffile.imwrite(
        out_path,
        arr,
        dtype=np.complex64,
        metadata=None,
        description=description or ''
    )


# ------------------------------ Polynomial GeoTransform (2D) ------------------------------ #

class PolyGeoTForm2D:
    """
    Bidirectional 2D polynomial transform between (x, y) and (u, v).

    This class fits polynomial mappings (least squares) in both directions
    after scaling coordinates to [0, 1] ranges for numerical stability.

    Parameters
    ----------
    src_xy : array_like, shape (N, 2)
        Source coordinates.
    dst_uv : array_like, shape (N, 2)
        Destination coordinates.
    degree : int, default=4
        Polynomial degree (total degree, using terms x^i y^j with i+j <= degree).
    """
    def __init__(self, src_xy, dst_uv, degree=4):
        self.degree = degree

        src_xy = np.asarray(src_xy, dtype=float)
        dst_uv = np.asarray(dst_uv, dtype=float)

        # Precompute min/max for normalization
        self.xmin, self.xmax = src_xy[:, 0].min(), src_xy[:, 0].max()
        self.ymin, self.ymax = src_xy[:, 1].min(), src_xy[:, 1].max()
        self.umin, self.umax = dst_uv[:, 0].min(), dst_uv[:, 0].max()
        self.vmin, self.vmax = dst_uv[:, 1].min(), dst_uv[:, 1].max()

        # Normalize to [0, 1] to improve conditioning
        Xn = self._normalize(src_xy[:, 0], self.xmin, self.xmax)
        Yn = self._normalize(src_xy[:, 1], self.ymin, self.ymax)
        Un = self._normalize(dst_uv[:, 0], self.umin, self.umax)
        Vn = self._normalize(dst_uv[:, 1], self.vmin, self.vmax)

        # Fit forward: (Xn, Yn) -> (Un, Vn)
        A = self._design_matrix(Xn, Yn, degree)
        self.coef_u, *_ = np.linalg.lstsq(A, Un, rcond=None)
        self.coef_v, *_ = np.linalg.lstsq(A, Vn, rcond=None)

        # Fit inverse: (Un, Vn) -> (Xn, Yn)
        Ain = self._design_matrix(Un, Vn, degree)
        self.coef_x, *_ = np.linalg.lstsq(Ain, Xn, rcond=None)
        self.coef_y, *_ = np.linalg.lstsq(Ain, Yn, rcond=None)

    @staticmethod
    def _normalize(arr, amin, amax):
        """Map values linearly to [0, 1] with tiny epsilon in denominator."""
        arr = np.asarray(arr, dtype=float)
        return (arr - amin) / (amax - amin + 1e-12)

    @staticmethod
    def _denorm(arrn, amin, amax):
        """Inverse of _normalize."""
        return arrn * (amax - amin + 1e-12) + amin

    def _design_matrix(self, x, y, degree):
        """
        Build the polynomial design matrix with all terms x^i y^j where i+j <= degree.

        Returns
        -------
        M : np.ndarray, shape (N, n_terms)
        """
        terms = []
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                terms.append((i, j))
        M = np.zeros((x.size, len(terms)), dtype=float)
        k = 0
        for i, j in terms:
            M[:, k] = (x ** i) * (y ** j)
            k += 1
        return M

    def _eval(self, coef, x, y):
        """
        Evaluate a polynomial (with current degree) given coefficient vector.
        """
        terms = []
        for i in range(self.degree + 1):
            for j in range(self.degree + 1 - i):
                terms.append((i, j))
        out = np.zeros_like(x, dtype=float)
        k = 0
        for i, j in terms:
            out += coef[k] * (x ** i) * (y ** j)
            k += 1
        return out

    def transform(self, x, y):
        """
        Apply forward transform: (x, y) -> (u, v)

        Returns:
            tuple: (u, v) tuple of arrays with the same shape as inputs.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        xn = self._normalize(x, self.xmin, self.xmax)
        yn = self._normalize(y, self.ymin, self.ymax)
        un = self._eval(self.coef_u, xn, yn)
        vn = self._eval(self.coef_v, xn, yn)
        u = self._denorm(un, self.umin, self.umax)
        v = self._denorm(vn, self.vmin, self.vmax)
        return u.reshape(x.shape), v.reshape(y.shape)

    def transform_inverse(self, u, v):
        """
        Apply inverse transform: (u, v) -> (x, y)

        Returns:
            tuple: (x, y) tuple of arrays with the same shape as inputs.
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        un = self._normalize(u, self.umin, self.umax)
        vn = self._normalize(v, self.vmin, self.vmax)
        xn = self._eval(self.coef_x, un, vn)
        yn = self._eval(self.coef_y, un, vn)
        x = self._denorm(xn, self.xmin, self.xmax)
        y = self._denorm(yn, self.ymin, self.ymax)
        return x.reshape(u.shape), y.reshape(v.shape)


def fitgeotform2d(src_xy, dst_uv, model="polynomial", degree=4):
    """
    Convenience wrapper to fit a 2D geotransform.

    Currently supports only the 'polynomial' model.

    Args:
        src_xy (array_like): Shape (N, 2) source points.
        dst_uv (array_like): Shape (N, 2) destination points.
        model (str): Transform model type.
        degree (int): Polynomial degree.

    Returns:
    -------
    PolyGeoTForm2D
    """
    assert model == "polynomial", "Polynomial model only"
    return PolyGeoTForm2D(src_xy, dst_uv, degree=degree)


# ------------------------------ TIFF & GeoTIFF helpers ------------------------------ #

def imfinfo_tiff(path):
    """
    Read minimal TIFF information (width/height and strip offsets).

    Args:
        path (str): Path to TIFF file.

    Returns:
        dict: Keys: 'Width', 'Height', 'StripOffsets' (list[int])

    Notes:
        - This assumes the SLC is stored in strips (not tiles).
        - Raises if the image is tiled, because a different reading strategy is needed.
    """
    info = {}
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        info['Width'] = int(page.tags['ImageWidth'].value)
        info['Height'] = int(page.tags['ImageLength'].value)

        # StripOffsets (273) or TileOffsets (324)
        strip_tag = page.tags.get(273)      # StripOffsets
        tile_tag = page.tags.get(324)       # TileOffsets

        if strip_tag is not None:
            strip_offsets = strip_tag.value
            if isinstance(strip_offsets, (int, np.integer)):
                strip_offsets = [int(strip_offsets)]
            else:
                strip_offsets = [int(x) for x in np.ravel(strip_offsets)]
            info['StripOffsets'] = strip_offsets
        elif tile_tag is not None:
            raise RuntimeError(
                "The TIFF file of SLC is in 'tiles' (TileOffsets) format. "
                "Tile-by-tile reconstruction or the use of SAR libraries is required"
            )
        else:
            raise RuntimeError("No StripOffsets, nor TileOffsets in the TIFF file.")

    return info


def geotiffinfo_tiepoints(path):
    """
    Read GeoTIFF tiepoints and dimensions. Tiepoints are returned in lon/lat if the CRS
    is projected (reprojected to EPSG:4326).

    Parameters
    ----------
    path : str

    Returns
    -------
    dict
        {
          'GeoTIFFTags': {'ModelTiepointTag': tiepoints[N,6]},
          'Width': int,
          'Height': int
        }
        Tiepoint rows are [I, J, K, X, Y, Z] with (X, Y) being lon/lat.
    """
    # Tiepoints (in image/CRS space)
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        tp_tag = page.tags.get(33922)  # ModelTiepointTag
        if tp_tag is None:
            raise ValueError("ModelTiepointTag (33922) not found in GeoTIFF.")
        vals = np.array(tp_tag.value, dtype=float)
        if vals.size % 6 != 0:
            raise ValueError("ModelTiepointTag length isn't multiple of 6 (I,J,K,X,Y,Z).")
        tiepoints = vals.reshape((-1, 6))

    # Raster size and CRS
    with rasterio.open(path) as src:
        width = src.width
        height = src.height
        crs = src.crs

        # Convert X,Y -> lon,lat if CRS is projected
        if crs is not None and not crs.is_geographic:
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            X = tiepoints[:, 3]
            Y = tiepoints[:, 4]
            lon, lat = transformer.transform(X, Y)
            tiepoints[:, 3] = lon
            tiepoints[:, 4] = lat

    info = {
        'GeoTIFFTags': {
            'ModelTiepointTag': tiepoints
        },
        'Width': width,
        'Height': height
    }
    return info


def read_slc_interleaved_int16(path, width, height, strip_offset):
    """
    Read a burst of SLC data stored as interleaved int16 [real, imag] per sample.

    Layout assumption
    -----------------
    The file stores 2 * width * height int16 values, organized column-major (Fortran order),
    where rows are lines (azimuth) and columns are samples (range). Even indices are real
    parts and odd indices are imaginary parts.

    Returns
    -------
    realimage : np.ndarray, shape (width, height)
    imagimage : np.ndarray, shape (width, height)
    """
    with open(path, 'rb') as f:
        f.seek(int(strip_offset), io.SEEK_SET)
        count = width * 2 * height
        data = np.fromfile(f, dtype=np.int16, count=count)
    slcimage = np.reshape(data, (width * 2, height), order='F')
    realimage = slcimage[0::2, :]
    imagimage = slcimage[1::2, :]
    return realimage, imagimage


# ------------------------------ Geodesy helpers ------------------------------ #

def distance_and_azimuth(lat1, lon1, lat2, lon2):
    """
    Compute geodesic distance (meters) and forward azimuth (deg) from (lat1, lon1) to (lat2, lon2).

    Returns:
        dist (np.ndarray): Distance in meters.
        az12 (np.ndarray): Forward azimuth in degrees.
    """
    lat1 = np.asarray(lat1).astype(float)
    lon1 = np.asarray(lon1).astype(float)
    lat2 = np.asarray(lat2).astype(float)
    lon2 = np.asarray(lon2).astype(float)
    # pyproj.Geod.inv: returns (az12, az21, dist)
    az12, az21, dist = _geod.inv(lon1, lat1, lon2, lat2)
    return dist, az12


def track1_gc(lat0, lon0, az_deg, dist_m):
    """
    Move from (lat0, lon0) along a great-circle with azimuth az_deg by distance dist_m.

    Returns:
        lat1 (np.ndarray): Destination latitude.
        lon1 (np.ndarray): Destination longitude.
    """
    lat0 = np.asarray(lat0).astype(float)
    lon0 = np.asarray(lon0).astype(float)
    az_deg = np.asarray(az_deg).astype(float)
    dist_m = np.asarray(dist_m).astype(float)
    lon1, lat1, _ = _geod.fwd(lon0, lat0, az_deg, dist_m)
    return lat1, lon1


# ------------------------------ Main pipeline ------------------------------ #

def main(inputpath: str, grdproduct: str, slcproduct: str, outputpath: str):
    """
    End-to-end tiling pipeline for Sentinel-1 SLC (VV/VH) and GRD patches, with:
      - Reading SLC bursts, calibration vectors, and annotation (tiepoints).
      - Creating complex patches (VV/VH) per burst tile.
      - Generating binary mask patches and XML labels for each tile.
      - Back-projecting corner coordinates to GRD space and saving GRD tiles.
      - Saving a legend CSV with (patch_id, has_ship, swath_id) flags.

    Args:
        inputpath (str): Root folder containing 'grd', 'slc', 'binarymask', 'raw'.
        grdproduct (str): GRD product directory name under 'grd'.
        slcproduct (str): SLC product directory name under 'slc'.
        outputpath (str): Output root folder; subfolders will be created: label, cp, mp, grd.
    """
    # Input/data folders
    path_grd = os.path.join(inputpath, 'grd')
    path_slc = os.path.join(inputpath, 'slc')
    path_binarymask = os.path.join(inputpath, 'binarymask')

    # Output folders
    path_label = os.path.join(outputpath, 'label')
    path_complex_patches = os.path.join(outputpath, 'slc')
    path_mask_patches = os.path.join(outputpath, 'mask')
    path_grd_patches = os.path.join(outputpath, 'grd')

    for d in [path_label, path_complex_patches, path_mask_patches, path_grd_patches]:
        os.makedirs(d, exist_ok=True)

    # Legend initialization (start row used as dummy header that will be removed later)
    slc_id = slcproduct[-4:]
    legend = {}
    legend[f'legend_{slc_id}'] = np.array([[1, 0, 0]], dtype=int)  # initial dummy row

    # Load binary mask quicklook (white -> mask=1)
    maskpng = Image.open(os.path.join(path_binarymask, slcproduct + '.png'))
    maskpng_np = np.array(maskpng)
    maskb = (maskpng_np[:, :, 0] + maskpng_np[:, :, 1] + maskpng_np[:, :, 2]) > 0
    del maskpng, maskpng_np

    npatch_first = 1  # global running patch index

    # Read metadata for SLC/GRD
    metadata_SLC = xml2struct(os.path.join(path_slc, slcproduct + ".SAFE", 'manifest.safe'))
    metadata_GRD = xml2struct(os.path.join(path_grd, grdproduct + ".SAFE", 'manifest.safe'))

    # Indices into product dataObject lists (pre-defined scheme)
    icalvh = list(range(3, 10, 3))
    icalvv = list(range(12, 19, 3))
    ivh = list(range(21, 24))
    ivv = list(range(24, 27))
    avh = list(range(1, 8, 3))
    avv = list(range(10, 17, 3))

    # SLC dataObjectSection parsing
    root_key_slc = next(iter(metadata_SLC.keys()))
    dataObjects_SLC = metadata_SLC[root_key_slc]['dataObjectSection']['dataObject']
    if not isinstance(dataObjects_SLC, list):
        dataObjects_SLC = [dataObjects_SLC]

    # Hold per-swath file paths
    slcfile_vh, slcfile_vv = {}, {}
    calfile_vh, calfile_vv = {}, {}
    annotationfile_vh, annotationfile_vv = {}, {}

    def href_to_path(aux):
        """
        Convert SAFE internal href to relative path.
        
        Strips leading './' and normalizes path separators for the OS.
        """
        path_str = str(aux)
        if path_str.startswith('./'):
            path_str = path_str[2:]
        return os.path.normpath(path_str)

    def parse_dn_vector(dn_text):
        """
        Parse space- or comma-separated calibration vector (DN) from text.
        """
        s = str(dn_text).strip()
        arr = np.fromstring(s, sep=' ')
        if arr.size == 0:
            parts = s.replace(',', ' ').split()
            arr = np.array([float(p) for p in parts], dtype=float)
        return arr

    # Resolve file paths for VH/VV images, calibration, and annotation per swath (1..3)
    for ind in range(3):
        # VH/VV image
        aux = dataObjects_SLC[ivh[ind] - 1]['byteStream']['fileLocation']['Attributes']['href']
        slcfile_vh[ind + 1] = href_to_path(aux)
        aux = dataObjects_SLC[ivv[ind] - 1]['byteStream']['fileLocation']['Attributes']['href']
        slcfile_vv[ind + 1] = href_to_path(aux)
        # Calibration files
        aux = dataObjects_SLC[icalvh[ind] - 1]['byteStream']['fileLocation']['Attributes']['href']
        calfile_vh[ind + 1] = href_to_path(aux)
        aux = dataObjects_SLC[icalvv[ind] - 1]['byteStream']['fileLocation']['Attributes']['href']
        calfile_vv[ind + 1] = href_to_path(aux)
        # Annotation files
        aux = dataObjects_SLC[avh[ind] - 1]['byteStream']['fileLocation']['Attributes']['href']
        annotationfile_vh[ind + 1] = href_to_path(aux)
        aux = dataObjects_SLC[avv[ind] - 1]['byteStream']['fileLocation']['Attributes']['href']
        annotationfile_vv[ind + 1] = href_to_path(aux)

    # GRD file paths (single VH/VV)
    root_key_grd = next(iter(metadata_GRD.keys()))
    dataObjects_GRD = metadata_GRD[root_key_grd]['dataObjectSection']['dataObject']
    if not isinstance(dataObjects_GRD, list):
        dataObjects_GRD = [dataObjects_GRD]
    aux = dataObjects_GRD[9 - 1]['byteStream']['fileLocation']['Attributes']['href']
    grdfile_vh = href_to_path(aux)
    aux = dataObjects_GRD[10 - 1]['byteStream']['fileLocation']['Attributes']['href']
    grdfile_vv = href_to_path(aux)

    # ---------------- Deburst & geometry per swath ---------------- #
    dataswath = {}  # container per idswath (1..3)
    for idswath in range(1, 4):
        # Open SLC (VV) to get basic TIFF info + tiepoints
        slc_im_path = os.path.join(path_slc, slcproduct + ".SAFE", slcfile_vv[idswath])
        slc_iminfo = imfinfo_tiff(slc_im_path)
        slc_geoinfo = geotiffinfo_tiepoints(slc_im_path)

        # Read calibration vector (VV)
        name_calfile_vv = os.path.join(path_slc, slcproduct + ".SAFE", calfile_vv[idswath])
        metadata_cal_vv = xml2struct(name_calfile_vv)
        calvec_list = metadata_cal_vv['calibration']['calibrationVectorList']['calibrationVector']
        if isinstance(calvec_list, list):
            dn_text = calvec_list[0]['dn']['Text']
        else:
            dn_text = calvec_list['dn']['Text']
        calfactorvv = parse_dn_vector(dn_text)

        # Burst metadata (annotation)
        metaburst = xml2struct(os.path.join(path_slc, slcproduct + ".SAFE", annotationfile_vv[idswath]))

        def _num(txt):  # Helper: string -> float
            return float(str(txt))

        # Pull key timing and spacing parameters
        pi = metaburst['product']['generalAnnotation']['productInformation']
        ia = metaburst['product']['imageAnnotation']['imageInformation']
        st = metaburst['product']['swathTiming']
        dataswath[idswath] = {}
        dataswath[idswath]['azsamplet'] = _num(ia['azimuthTimeInterval']['Text'])
        dataswath[idswath]['samples'] = _num(ia['numberOfSamples']['Text'])
        dataswath[idswath]['azsamples'] = _num(ia['azimuthPixelSpacing']['Text'])
        dataswath[idswath]['pxsamplet'] = 1.0 / _num(pi['rangeSamplingRate']['Text'])
        dataswath[idswath]['pxsamples'] = _num(ia['rangePixelSpacing']['Text'])
        dataswath[idswath]['heading'] = _num(pi['platformHeading']['Text'])
        dataswath[idswath]['linesburst'] = int(float(st['linesPerBurst']['Text']))
        dataswath[idswath]['sizepatch'] = dataswath[idswath]['linesburst']
        dataswath[idswath]['nburst'] = int(float(st['burstList']['Attributes']['count']))

        # Build time axis and valid-sample windows per burst
        timeaxis = np.zeros(dataswath[idswath]['nburst'])
        firstsample_ind = np.zeros(dataswath[idswath]['nburst'], dtype=int)
        lastsample_ind = np.zeros(dataswath[idswath]['nburst'], dtype=int)

        bursts = st['burstList']['burst']
        if not isinstance(bursts, list):
            bursts = [bursts]

        for ib in range(dataswath[idswath]['nburst']):
            b = bursts[ib]
            timeburst = _num(b['azimuthAnxTime']['Text'])
            timeaxis[ib] = timeburst
            firstsample = [float(x) for x in str(b['firstValidSample']['Text']).split()]
            firstsample_ind[ib] = sum(1 for x in firstsample[:round(dataswath[idswath]['linesburst'] / 2)] if int(x) == -1)
            lastsample = [float(x) for x in str(b['lastValidSample']['Text']).split()]
            lastsample_ind[ib] = int(lastsample[firstsample_ind[ib]])

        dataswath[idswath]['timeaxis'] = timeaxis
        dataswath[idswath]['firstsample'] = firstsample_ind
        dataswath[idswath]['lastsample'] = lastsample_ind

        # Tiepoints and per-grid sampling in range (xsampling)
        tiepoints = slc_geoinfo['GeoTIFFTags']['ModelTiepointTag']  # N x 6 (I, J, K, X, Y, Z)
        xsampling = int(np.sum(tiepoints[:, 1] == 0))  # number of tiepoints for the first line J=0
        dataswath[idswath]['xsampling'] = xsampling

        # Read geolocation grid points (incidence/elevation/pixel/slant range time)
        geogrid = metaburst['product']['geolocationGrid']['geolocationGridPointList']['geolocationGridPoint']
        if not isinstance(geogrid, list):
            geogrid = [geogrid]
        incidence = np.zeros(xsampling * (dataswath[idswath]['nburst'] + 1))
        elevation = np.zeros_like(incidence)
        xpixel = np.zeros_like(incidence)
        srtime = np.zeros_like(incidence)

        for indinc in range(xsampling * (dataswath[idswath]['nburst'] + 1)):
            gg = geogrid[indinc]
            incidence[indinc] = float(gg['incidenceAngle']['Text'])
            elevation[indinc] = float(gg['elevationAngle']['Text'])
            xpixel[indinc] = float(gg['pixel']['Text'])
            srtime[indinc] = float(gg['slantRangeTime']['Text'])

        dataswath[idswath]['srtime'] = srtime
        dataswath[idswath]['tiepoints'] = tiepoints
        dataswath[idswath]['Width'] = imfinfo_tiff(slc_im_path)['Width']
        dataswath[idswath]['Height'] = imfinfo_tiff(slc_im_path)['Height']
        dataswath[idswath]['StripOffsets'] = imfinfo_tiff(slc_im_path)['StripOffsets']

    # Compute resampling factors between SLC geometry and mask quicklook
    ql_slcproduct_lines = (
        max([dataswath[1]['timeaxis'][-1], dataswath[2]['timeaxis'][-1], dataswath[3]['timeaxis'][-1]])
        - min([dataswath[1]['timeaxis'][0], dataswath[2]['timeaxis'][0], dataswath[3]['timeaxis'][0]])
    ) / dataswath[1]['azsamplet']
    ql_slcproduct_lines = ql_slcproduct_lines + dataswath[3]['linesburst']
    ql_slcproduct_samples = (max(dataswath[3]['srtime']) - min(dataswath[1]['srtime'])) / dataswath[1]['pxsamplet']
    resamplinglines = ql_slcproduct_lines / maskb.shape[0]
    resamplingsamples = ql_slcproduct_samples / maskb.shape[1]

    # ---------------- Tiling complex patches (VV) ---------------- #
    for idswath in range(1, 4):
        # Indices that bound each burst's tiepoints block along the list
        iibb = np.arange(1, len(dataswath[idswath]['tiepoints'][:, 0]) + 1, dataswath[idswath]['xsampling'])

        # Read SLC VV raw interleaved data
        slcdata_vv_path = os.path.join(path_slc, slcproduct + ".SAFE", slcfile_vv[idswath])
        realimage, imagimage = read_slc_interleaved_int16(
            slcdata_vv_path,
            dataswath[idswath]['Width'],
            dataswath[idswath]['Height'],
            dataswath[idswath]['StripOffsets'][0]
        )

        for indb in range(1, dataswath[idswath]['nburst'] + 1):
            # Build lon/lat grid by interpolating between the two tiepoint rows surrounding a burst
            idx0 = iibb[indb - 1] - 1
            idx1 = iibb[indb] - 1 if indb < len(iibb) else iibb[indb - 1] - 1

            latA = dataswath[idswath]['tiepoints'][idx0:idx0 + dataswath[idswath]['xsampling'], 4]
            lonA = dataswath[idswath]['tiepoints'][idx0:idx0 + dataswath[idswath]['xsampling'], 3]
            latB = dataswath[idswath]['tiepoints'][idx1:idx1 + dataswath[idswath]['xsampling'], 4]
            lonB = dataswath[idswath]['tiepoints'][idx1:idx1 + dataswath[idswath]['xsampling'], 3]

            dmov, amov = distance_and_azimuth(latA, lonA, latB, lonB)

            # Interpolate lats/lons across azimuth lines in the burst
            lats = np.zeros((dataswath[idswath]['xsampling'], dataswath[idswath]['linesburst']))
            lons = np.zeros_like(lats)
            if indb == dataswath[idswath]['nburst']:
                # For the last burst, distribute lines evenly between tiepoint rows
                incfinal = dmov / (dataswath[idswath]['linesburst'] - 1)
                for irej in range(1, dataswath[idswath]['linesburst'] + 1):
                    lat_irej, lon_irej = track1_gc(latA, lonA, amov, (irej - 1) * incfinal)
                    lats[:, irej - 1] = lat_irej
                    lons[:, irej - 1] = lon_irej
            else:
                # Otherwise, advance by azimuth pixel spacing per line
                for irej in range(1, dataswath[idswath]['linesburst'] + 1):
                    dist_vec = np.ones(dataswath[idswath]['xsampling']) * (irej - 1) * dataswath[idswath]['azsamples']
                    lat_irej, lon_irej = track1_gc(latA, lonA, amov, dist_vec)
                    lats[:, irej - 1] = lat_irej
                    lons[:, irej - 1] = lon_irej

            # Flatten lon/lat fields, prepare (col,line) grid for interpolation
            lons_vec = lons.reshape(-1, order='F')
            lats_vec = lats.reshape(-1, order='F')

            rowaxis = np.arange(0, dataswath[idswath]['linesburst'], dtype=float)
            rowmatrix = np.tile(rowaxis, (dataswath[idswath]['xsampling'], 1))
            rowmatrixline = rowmatrix.reshape(-1, order='F')

            i0 = idx0
            xs = dataswath[idswath]['xsampling']
            colvec = dataswath[idswath]['tiepoints'][i0:i0 + xs, 0].astype(float)  # I (column)
            colmatrix = np.tile(colvec[:, None], (1, dataswath[idswath]['linesburst']))
            colmatrixline = colmatrix.reshape(-1, order='F')

            pts = np.column_stack([colmatrixline, rowmatrixline])
            interp_lon = LinearNDInterpolator(pts, lons_vec)
            interp_lat = LinearNDInterpolator(pts, lats_vec)
            nearest_lon = NearestNDInterpolator(pts, lons_vec)
            nearest_lat = NearestNDInterpolator(pts, lats_vec)

            # Build complex image for this burst: transpose to [lines, samples]
            imageburst = (
                realimage[:, (indb - 1) * dataswath[idswath]['linesburst']: indb * dataswath[idswath]['linesburst']]
                + 1j * imagimage[:, (indb - 1) * dataswath[idswath]['linesburst']: indb * dataswath[idswath]['linesburst']]
            ).T

            # Slide square tiles of size 'sizepatch' across range dimension
            for ipatchsample in range(1, dataswath[idswath]['Width'] + 1, dataswath[idswath]['sizepatch']):
                if ipatchsample + dataswath[idswath]['sizepatch'] - 1 > dataswath[idswath]['Width']:
                    frameflag = 1  # incomplete tile at right edge: skip
                else:
                    coornerpatch_sample = [
                        ipatchsample,
                        ipatchsample + dataswath[idswath]['sizepatch'] - 1,
                        ipatchsample + dataswath[idswath]['sizepatch'] - 1,
                        ipatchsample,
                        ipatchsample
                    ]
                    coornerpatch_line = [
                        1, 1,
                        dataswath[idswath]['linesburst'],
                        dataswath[idswath]['linesburst'],
                        1
                    ]
                    complexpatch = imageburst[0:dataswath[idswath]['linesburst'],
                                              ipatchsample - 1:ipatchsample - 1 + dataswath[idswath]['sizepatch']]
                    frameflag = 0

                if frameflag == 0:
                    # Save complex VV patch
                    out_tif_vv = os.path.join(path_complex_patches, f'DB_OPENSAR_RFI_{npatch_first}_SLC_VV.tiff')
                    save_complex_patch_tiff(out_tif_vv, complexpatch, description=f'calfactorvv_len={calfactorvv}')

                    # Interpolate lon/lat for the 4 corners (+ closure vertex)
                    coornerpatch_sample_arr = np.asarray(coornerpatch_sample, dtype=float) - 1
                    coornerpatch_line_arr = np.asarray(coornerpatch_line, dtype=float) - 1

                    c_lon = interp_lon(coornerpatch_sample_arr, coornerpatch_line_arr)
                    c_lat = interp_lat(coornerpatch_sample_arr, coornerpatch_line_arr)
                    # Fallback to nearest neighbor where ND interpolation yields NaN
                    nan_mask = np.isnan(c_lon) | np.isnan(c_lat)
                    if np.any(nan_mask):
                        c_lon[nan_mask] = nearest_lon(coornerpatch_sample_arr[nan_mask], coornerpatch_line_arr[nan_mask])
                        c_lat[nan_mask] = nearest_lat(coornerpatch_sample_arr[nan_mask], coornerpatch_line_arr[nan_mask])

                    # Normalize longitudes to [-180, 180)
                    c_lon = ((np.asarray(c_lon) + 180.0) % 360.0) - 180.0
                    lat_txt = ' '.join(fmt4(x) for x in np.asarray(c_lat))
                    lon_txt = ' '.join(fmt4(x) for x in np.asarray(c_lon))

                    sar_sample_txt = ' '.join(str(int(x)) for x in (coornerpatch_sample_arr + 1))
                    sar_line_txt = ' '.join(
                        str(int(x)) for x in (coornerpatch_line_arr + 1 + (indb - 1) * dataswath[idswath]['linesburst'])
                    )

                    # Compose output XML structure (label)
                    outxml_scene_id = f'DB_OPENSAR_RFI_{npatch_first}'
                    outxml_date = datetime.today().year * 10000 + datetime.today().month * 100 + datetime.today().day
                    outxml_struct = {
                        'outxml': {
                            'Scene_Info': {'Scene_ID': outxml_scene_id, 'Date': outxml_date, 'Version': 1,
                                           'CaseStudy': 'RFIDetection'},
                            'SARData': {
                                'SAR_Mission': slcproduct[0:3],
                                'SARProduct': slcproduct,
                                'SLCSwath': idswath,
                                'Time_Interval': {'Start': f"{slcproduct[17:25]}_{slcproduct[26:32]}",
                                                  'Stop': f"{slcproduct[33:41]}_{slcproduct[42:48]}"}
                            },
                            'ProcessingData': {
                                'Corner_Coord': {
                                    'SARData_Sample': sar_sample_txt,
                                    'SARData_Line': sar_line_txt,
                                    'Scene_Sample': '1 512 512 1',
                                    'Scene_Line': '1 1 512 512',
                                    'Latitude': lat_txt,
                                    'Longitude': lon_txt,
                                },
                                'StatisticsReport': {}
                            }
                        }
                    }

                    # Update running patch id
                    npatch_first += 1

                    # Map the patch position to the binary mask (quicklook) resolution
                    linemask = (dataswath[idswath]['timeaxis'][indb - 1] - dataswath[1]['timeaxis'][0]) / \
                               dataswath[idswath]['azsamplet']
                    lineresamp = int(round(linemask / resamplinglines))
                    if lineresamp < 1:
                        lineresamp = 1
                    if lineresamp > maskb.shape[0]:
                        lineresamp = maskb.shape[0]
                    patchsizelineresamp = int(round(dataswath[idswath]['sizepatch'] / resamplinglines))

                    samplemask = (dataswath[idswath]['srtime'][(indb - 1) * dataswath[idswath]['xsampling']] -
                                  dataswath[1]['srtime'][(indb - 1) * dataswath[idswath]['xsampling']]) / \
                                 dataswath[idswath]['pxsamplet']
                    samplemask = samplemask + coornerpatch_sample[0] - 1
                    sampleresamp = int(round(samplemask / resamplingsamples))

                    # Debug checkpoint in original code
                    if npatch_first - 1 == 32:
                        a = 0  # no-op / placeholder

                    if sampleresamp < 1:
                        sampleresamp = 1
                    if sampleresamp > maskb.shape[1]:
                        sampleresamp = maskb.shape[1]
                    patchsizesampleresamp = int(round(dataswath[idswath]['sizepatch'] / resamplingsamples))

                    # Extract mask patch and set label (1/0)
                    maskb_patchresamp = maskb[
                        max(0, lineresamp - 1): max(0, lineresamp - 1) + patchsizelineresamp,
                        max(0, sampleresamp - 1): max(0, sampleresamp - 1) + patchsizesampleresamp
                    ]
                    if np.sum(maskb_patchresamp > 0) > 0:
                        outxml_struct['outxml']['ProcessingData']['StatisticsReport']['RFIDetection'] = 1
                        arr = legend[f'legend_{slc_id}']
                        newrow = np.array([[npatch_first - 1, 1, idswath]])
                        legend[f'legend_{slc_id}'] = np.vstack([arr, newrow])
                    else:
                        outxml_struct['outxml']['ProcessingData']['StatisticsReport']['RFIDetection'] = 0
                        arr = legend[f'legend_{slc_id}']
                        newrow = np.array([[npatch_first - 1, 0, idswath]])
                        legend[f'legend_{slc_id}'] = np.vstack([arr, newrow])

                    # Save mask patch resized to the SLC tile shape (nearest to keep binary)
                    mask_img_patch = Image.fromarray(maskb_patchresamp.astype(np.uint8) * 255)
                    mask_img_patch = mask_img_patch.resize(
                        (dataswath[idswath]['linesburst'], dataswath[idswath]['linesburst']), resample=Image.NEAREST)
                    mask_img_patch.save(os.path.join(path_mask_patches, f'DB_OPENSAR_RFI_{npatch_first - 1}_MASK.png'))

                    # Save label XML
                    struct2xml(outxml_struct, os.path.join(path_label, f'DB_OPENSAR_RFI_{npatch_first - 1}.xml'))

    # --- Save legend as TXT (CSV) ---
    legend_key = f'legend_{slc_id}'
    legend_arr = legend[legend_key]

    # Drop header/dummy first row if present
    if legend_arr.shape[0] >= 1 and np.array_equal(legend_arr[0], [1, 0, 0]):
        legend_arr = legend_arr[1:]
    log_path = os.path.join(path_label, f'legend_{slc_id}.txt')
    np.savetxt(
        log_path,
        legend_arr,
        fmt='%d',
        delimiter=',',
        header='patch_id,has_ship,swath_id',
        comments=''  # no '# ' in header
    )
    print(f"[INFO] Legend saved in: {log_path} (rows: {legend_arr.shape[0]})")

    # ---------------- Tiling complex patches (VH) ---------------- #
    for idswath in range(1, 4):
        # Basic info for VH
        slc_iminfo_vh = imfinfo_tiff(os.path.join(path_slc, slcproduct + ".SAFE", slcfile_vh[idswath]))
        # Calibration vector (VH)
        name_calfile_vh = os.path.join(path_slc, slcproduct + ".SAFE", calfile_vh[idswath])
        metadata_cal_vh = xml2struct(name_calfile_vh)

        calvec_list_vh = metadata_cal_vh['calibration']['calibrationVectorList']['calibrationVector']
        if isinstance(calvec_list_vh, list):
            dn_text_vh = calvec_list_vh[0]['dn']['Text']
        else:
            dn_text_vh = calvec_list_vh['dn']['Text']
        calfactorvh = parse_dn_vector(dn_text_vh)

        # Read SLC VH raw interleaved data
        slcdata_vh_path = os.path.join(path_slc, slcproduct + ".SAFE", slcfile_vh[idswath])
        realimage_vh, imagimage_vh = read_slc_interleaved_int16(
            slcdata_vh_path,
            slc_iminfo_vh['Width'],
            slc_iminfo_vh['Height'],
            slc_iminfo_vh['StripOffsets'][0]
        )

        # Select patch IDs belonging to this swath
        oldlabels_swath = legend[f'legend_{slc_id}'][:, 2] == idswath
        oldlabels_id = legend[f'legend_{slc_id}'][:, 0]
        oldlabels_id_swath = oldlabels_id[oldlabels_swath]

        # Reuse SLC corner coordinates from the previously saved XMLs
        for ipat in range(int(np.sum(oldlabels_swath))):
            label_id = int(oldlabels_id_swath[ipat])

            olddata = xml2struct(os.path.join(path_label, f'DB_OPENSAR_RFI_{label_id}.xml'))
            cc = olddata['outxml']['ProcessingData']['Corner_Coord']

            sample_txt = cc['SARData_Sample']['Text'] if isinstance(cc['SARData_Sample'], dict) else str(cc['SARData_Sample'])
            line_txt = cc['SARData_Line']['Text'] if isinstance(cc['SARData_Line'], dict) else str(cc['SARData_Line'])

            coornerpatch_sample = [int(x) for x in sample_txt.split()]
            coornerpatch_line = [int(x) for x in line_txt.split()]

            cs_min, cs_max = min(coornerpatch_sample), max(coornerpatch_sample)
            cl_min, cl_max = min(coornerpatch_line), max(coornerpatch_line)

            complexpatch_vh = (
                realimage_vh[cs_min - 1:cs_max, cl_min - 1:cl_max]
                + 1j * imagimage_vh[cs_min - 1:cs_max, cl_min - 1:cl_max]
            ).T

            save_complex_patch_tiff(
                os.path.join(path_complex_patches, f'DB_OPENSAR_RFI_{label_id}_SLC_VH.tiff'),
                complexpatch_vh,
                description=f'calfactorvh_len={calfactorvh}'
            )

    # ---------------- Tiling GRD patches (VV & VH) ---------------- #
    oldlabels_id_all = legend[f'legend_{slc_id}'][:, 0]

    grd_vh_path = os.path.join(path_grd, grdproduct + ".SAFE", grdfile_vh)
    grd_vv_path = os.path.join(path_grd, grdproduct + ".SAFE", grdfile_vv)
    grdgeoinfo_vh = geotiffinfo_tiepoints(grd_vh_path)
    grdgeoinfo_vv = geotiffinfo_tiepoints(grd_vv_path)

    # Fit polynomial geotransforms from GRD pixel (I,J) to lon/lat
    tp_vh = grdgeoinfo_vh['GeoTIFFTags']['ModelTiepointTag']
    src_vh = tp_vh[:, 0:2]  # (I, J)
    dst_vh = tp_vh[:, 3:5]  # (lon, lat)
    tformgrd_vh = fitgeotform2d(src_vh, dst_vh, model="polynomial", degree=4)

    tp_vv = grdgeoinfo_vv['GeoTIFFTags']['ModelTiepointTag']
    src_vv = tp_vv[:, 0:2]
    dst_vv = tp_vv[:, 3:5]
    tformgrd_vv = fitgeotform2d(src_vv, dst_vv, model="polynomial", degree=4)

    # Load GRD rasters (uint8/uint16 depending on product; PIL keeps dtype)
    grddatavv = np.array(Image.open(grd_vv_path))
    # FIXME: Likely should be grd_vh_path, not grd_vv_path
    grddatavh = np.array(Image.open(grd_vv_path))

    for ipat in range(len(oldlabels_id_all)):
        label_id = int(oldlabels_id_all[ipat])
        olddata = xml2struct(os.path.join(path_label, f'DB_OPENSAR_RFI_{label_id}.xml'))

        cc = olddata['outxml']['ProcessingData']['Corner_Coord']

        lat_txt = cc['Latitude']['Text'] if isinstance(cc['Latitude'], dict) else str(cc['Latitude'])
        lon_txt = cc['Longitude']['Text'] if isinstance(cc['Longitude'], dict) else str(cc['Longitude'])

        coord_lat = [float(x) for x in lat_txt.split()]
        coord_lon = [float(x) for x in lon_txt.split()]

        # Back-project polygon from lon/lat to GRD pixel space
        xxxgvh, yyygvh = tformgrd_vh.transform_inverse(coord_lon, coord_lat)
        xxxgvv, yyygvv = tformgrd_vv.transform_inverse(coord_lon, coord_lat)

        # Clip to raster bounds
        xxxgvh = np.round(xxxgvh).astype(int)
        yyygvh = np.round(yyygvh).astype(int)
        xxxgvh[xxxgvh <= 0] = 1
        yyygvh[yyygvh <= 0] = 1
        xxxgvh[xxxgvh >= grdgeoinfo_vh['Width']] = grdgeoinfo_vh['Width']
        yyygvh[yyygvh >= grdgeoinfo_vh['Height']] = grdgeoinfo_vh['Height']

        xxxgvv = np.round(xxxgvv).astype(int)
        yyygvv = np.round(yyygvv).astype(int)
        xxxgvv[xxxgvv <= 0] = 1
        yyygvv[yyygvv <= 0] = 1
        xxxgvv[xxxgvv >= grdgeoinfo_vv['Width']] = grdgeoinfo_vv['Width']
        yyygvv[yyygvv >= grdgeoinfo_vv['Height']] = grdgeoinfo_vv['Height']

        # Extract rectangular tiles spanning the polygon's min/max bounds
        tilesgrdvv = grddatavv[min(yyygvv):max(yyygvv), min(xxxgvv):max(xxxgvv)]
        tilesgrdvh = grddatavv[min(yyygvh):max(yyygvh), min(xxxgvh):max(xxxgvh)]

        Image.fromarray(tilesgrdvv).save(os.path.join(path_grd_patches, f'DB_OPENSAR_RFI_{label_id}_GRD_VV.tiff'))
        Image.fromarray(tilesgrdvh).save(os.path.join(path_grd_patches, f'DB_OPENSAR_RFI_{label_id}_GRD_VH.tiff'))

    print("Pipeline completed.")


if __name__ == '__main__':
    main()