# -*- coding: utf-8 -*-

"""
Sentinel-1 Dark Vessel Detection (DVD) – SLC/GRD/OCN tiling and labeling pipeline

Overview
--------
This module builds a training/validation dataset from Sentinel-1 SAFE products by tiling
complex SLC data, generating per-patch metadata/labels, and extracting matching GRD tiles.
It also associates candidate vessels from a provided CSV (SLCvalidation) and optionally
injects wind information from OCN products.

Processing pipeline (high level)
--------------------------------
1) SAFE discovery
   - Parse SLC/GRD/OCN SAFE manifests to resolve internal file paths (measurement, calibration,
     annotations, and OCN NetCDF).
   - Paths are derived from the product names and the script directory layout.

2) Geometry and per-burst mapping
   - Read SLC VV tie points and build a degree-4 2D polynomial mapping from (sample,line) to (lon,lat)
     for each burst, preserving Fortran-like vectorization and index conventions used downstream.
   - Compute the deburst mapping table (tstrans) to translate bursting to debursting line indices.

3) Candidate vessel association
   - Load candidates from SLCvalidation CSV (required columns documented in the loader).
   - Filter by scene, swath, and is_vessel == true.
   - For each VV patch, test whether vessels fall inside the patch polygon in (sample,line) space
     (after debursting). Optionally compute a simple intensity filter threshold on the SLC burst
     amplitude at the candidate location.

4) Tiling and outputs (SLC → VV/VH)
   - Tile SLC VV into 512×512 (configurable) complex patches (single-band complex64 GeoTIFF).
   - For each patch, write a companion XML with:
       * SAR corner coordinates in both SARData indices and geodetic lon/lat (wrapped to [-180,180]).
       * Optional wind direction/speed nearest to patch center (from OCN NetCDF), if available.
       * Vessel list with centroid, bounding box in patch coordinates, shoreline distances, and confidence.
     A per-scene legend file (CSV-like) records patch_id, presence of ships, and swath_id.
   - For VH, reuse SARData corner indices from the XML to extract the corresponding complex patch.

5) GRD extraction (VV/VH)
   - Invert the per-product tie-point mapping from lon/lat to pixel coordinates and crop the GRD
     rasters to the per-patch footprint. Save plain TIFF tiles for VV and VH.

Inputs
------
- SLC SAFE product (folder name).
- GRD SAFE product (folder name).
- OCN SAFE product (folder name).
- SLCvalidation CSV with required vessel candidate columns.
- Output base directory for this scene (subfolders are created/cleaned automatically).
- Patch size (e.g., 512).

Outputs
-------
- <output>/label/
    * One XML per SLC VV patch with corner coordinates, wind (if available), and vessel list.
    * legend<id>.txt: CSV-like file with patch_id, has_ship (0/1), swath_id.
- <output>/cp/
    * DB_OPENSAR_DVD_<id>_SLC_VV.tiff   (complex64, single-band)
    * DB_OPENSAR_DVD_<id>_SLC_VH.tiff   (complex64, single-band; only for patches present in legend)
- <output>/grd/
    * DB_OPENSAR_DVD_<id>_GRD_VV.tiff   (uint raster subset from GRD VV)
    * DB_OPENSAR_DVD_<id>_GRD_VH.tiff   (uint raster subset from GRD VH)

Conventions and notes
---------------------
- Arrays are 0-based internally. XML/SAR indices that are logically 1-based are handled accordingly
  when exported or mapped back into image coordinates.
- Longitude values are wrapped to [-180, 180] when exported to XML.
- Complex patches are stored as single-band complex64 GeoTIFFs (GDAL CFloat32 compatible).
- SAFE internal hrefs are normalized to platform paths before opening.
- The wind layer from OCN (NetCDF) is optional; if missing, wind fields are omitted from XML.
- Degree-4 polynomial mapping is used for consistency; forward and inverse fits are computed.
- Very large rasters are supported by disabling PIL’s pixel limits and using tifffile/rasterio.

Dependencies
------------
- numpy, tifffile, rasterio, pyproj, Pillow (PIL), netCDF4 (optional, for OCN), pandas (for CSV loader).

Directory layout (expected)
---------------------------
- products/
   ├─ slc/<SLC_SAFE>/
   ├─ grd/<GRD_SAFE>/
   └─ ocn/<OCN_SAFE>/
- output/<scene_xxx>/
   ├─ label/
   ├─ cp/
   └─ grd/

Entry point
-----------
Use the provided `main(grdproduct, slcproduct, ocnproduct, outputpath,
slcvalidation, sizepatch)` to process one scene. A separate launcher can read
a config and a mapping CSV to run multiple scenes sequentially.
"""


import os
import io
from typing import Optional, Dict
import numpy as np

# PIL for large GeoTIFFs (trusted Sentinels)
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

import tifffile
import rasterio
import xml.etree.ElementTree as ET
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
from pyproj import Geod, Transformer
from pathlib import Path 

# ---------- Helpers: formatting, XML, path, etc. ----------

def fmt10(x: float) -> str:
    try:
        import numpy as np
        if np.isnan(x):
            return 'NaN'
    except Exception:
        pass
    return f"{float(x):.10f}".rstrip('0').rstrip('.')


def _safe_tag_name(name: str) -> str:
    return name.replace('-', '_').replace(':', '_').replace('.', '_')

def xml2struct(file_path_or_elem) -> dict:
    """Convert XML to nested dict with 'Attributes' and 'Text' fields."""
    if isinstance(file_path_or_elem, ET.Element):
        root = file_path_or_elem
    else:
        p = str(file_path_or_elem)
        if not os.path.exists(p):
            if not p.lower().endswith('.xml') and os.path.exists(p + '.xml'):
                p = p + '.xml'
            else:
                raise FileNotFoundError(f"XML not found: {file_path_or_elem}")
        root = ET.parse(p).getroot()

    def parse_attributes(elem):
        return {_safe_tag_name(k): v for k, v in elem.attrib.items()}

    def parse_children(elem):
        children, ptext = {}, {}
        for child in list(elem):
            name = _safe_tag_name(child.tag)
            text = (child.text or '').strip()
            attr = parse_attributes(child)
            cc, _ = parse_children(child)
            if name not in ['#text', '#comment']:
                node = {}
                if cc:    node.update(cc)
                if text:  node['Text'] = text
                if attr:  node['Attributes'] = attr
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

    ch, pt = parse_children(root)
    out = {_safe_tag_name(root.tag): ch if ch else {}}
    if pt:
        out[_safe_tag_name(root.tag)].update(pt)
    attrs = parse_attributes(root)
    if attrs:
        out[_safe_tag_name(root.tag)]['Attributes'] = attrs
    return out

def struct2xml(s: dict, file_path: str) -> None:
    """Convert nested dict (single root key) to XML file."""
    if not isinstance(s, dict) or len(s) != 1:
        raise ValueError("Input dict must have a single root key.")
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
    if not file_path.lower().endswith('.xml'):
        file_path = file_path + '.xml'

    tree = ET.ElementTree(root_elem)
    ET.indent(tree, space="   ", level=0)
    tree.write(file_path, encoding='utf-8', xml_declaration=True)
    #ET.ElementTree(root_elem).write(file_path, encoding='utf-8', xml_declaration=True)

def _href_to_path(href: str) -> str:
    """Convert SAFE internal href to Windows path like aux2(3:end)."""
    aux2 = str(href).replace('/', '\\')
    return aux2[2:]

def _as_text(v) -> str:
    return v.get('Text') if isinstance(v, dict) else str(v)

def _wrap_lon180(lon: np.ndarray) -> np.ndarray:
    return ((np.asarray(lon) + 180.0) % 360.0) - 180.0


# ---------- Polynomial geotransform (degree=4) ----------

class PolyGeoTForm2D:
    """2D polynomial transform with min-max normalization; forward and inverse LS fits."""
    def __init__(self, src_xy, dst_uv, degree=4):
        self.degree = degree
        src_xy = np.asarray(src_xy, dtype=float)
        dst_uv = np.asarray(dst_uv, dtype=float)
        self.xmin, self.xmax = src_xy[:, 0].min(), src_xy[:, 0].max()
        self.ymin, self.ymax = src_xy[:, 1].min(), src_xy[:, 1].max()
        self.umin, self.umax = dst_uv[:, 0].min(), dst_uv[:, 0].max()
        self.vmin, self.vmax = dst_uv[:, 1].min(), dst_uv[:, 1].max()

        Xn = self._normalize(src_xy[:, 0], self.xmin, self.xmax)
        Yn = self._normalize(src_xy[:, 1], self.ymin, self.ymax)
        Un = self._normalize(dst_uv[:, 0], self.umin, self.umax)
        Vn = self._normalize(dst_uv[:, 1], self.vmin, self.vmax)

        A = self._design_matrix(Xn, Yn, degree)
        self.coef_u, *_ = np.linalg.lstsq(A, Un, rcond=None)
        self.coef_v, *_ = np.linalg.lstsq(A, Vn, rcond=None)

        Ain = self._design_matrix(Un, Vn, degree)
        self.coef_x, *_ = np.linalg.lstsq(Ain, Xn, rcond=None)
        self.coef_y, *_ = np.linalg.lstsq(Ain, Yn, rcond=None)

    @staticmethod
    def _normalize(arr, amin, amax):
        arr = np.asarray(arr, dtype=float)
        return (arr - amin) / (amax - amin + 1e-12)

    @staticmethod
    def _denorm(arrn, amin, amax):
        return arrn * (amax - amin + 1e-12) + amin

    @staticmethod
    def _design_matrix(x, y, degree):
        terms = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]
        M = np.zeros((x.size, len(terms)), dtype=float)
        for k, (i, j) in enumerate(terms):
            M[:, k] = (x ** i) * (y ** j)
        return M

    def _eval(self, coef, x, y):
        terms = [(i, j) for i in range(self.degree + 1) for j in range(self.degree + 1 - i)]
        out = np.zeros_like(x, dtype=float)
        for k, (i, j) in enumerate(terms):
            out += coef[k] * (x ** i) * (y ** j)
        return out

    def transform(self, x, y):
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        xn = self._normalize(x, self.xmin, self.xmax)
        yn = self._normalize(y, self.ymin, self.ymax)
        un = self._eval(self.coef_u, xn, yn)
        vn = self._eval(self.coef_v, xn, yn)
        u = self._denorm(un, self.umin, self.umax)
        v = self._denorm(vn, self.vmin, self.vmax)
        return u.reshape(x.shape), v.reshape(y.shape)

    def transform_inverse(self, u, v):
        u = np.asarray(u, dtype=float); v = np.asarray(v, dtype=float)
        un = self._normalize(u, self.umin, self.umax)
        vn = self._normalize(v, self.vmin, self.vmax)
        xn = self._eval(self.coef_x, un, vn)
        yn = self._eval(self.coef_y, un, vn)
        x = self._denorm(xn, self.xmin, self.xmax)
        y = self._denorm(yn, self.ymin, self.ymax)
        return x.reshape(u.shape), y.reshape(v.shape)

def fitgeotform2d(src_xy, dst_uv, degree=4) -> PolyGeoTForm2D:
    return PolyGeoTForm2D(src_xy, dst_uv, degree=degree)


# ---------- SLC/GeoTIFF readers ----------

def imfinfo_tiff(path: str) -> dict:
    """Minimal imfinfo clone: Width, Height, StripOffsets (not tiled)."""
    info = {}
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        info['Width'] = int(page.tags['ImageWidth'].value)
        info['Height'] = int(page.tags['ImageLength'].value)
        strip_tag = page.tags.get(273)  # StripOffsets
        tile_tag  = page.tags.get(324)  # TileOffsets
        if strip_tag is not None:
            offsets = strip_tag.value
            if isinstance(offsets, (int, np.integer)):
                offsets = [int(offsets)]
            else:
                offsets = [int(x) for x in np.ravel(offsets)]
            info['StripOffsets'] = offsets
        elif tile_tag is not None:
            raise RuntimeError("Tiled TIFF not supported in this fread-style reader.")
        else:
            raise RuntimeError("No StripOffsets/TileOffsets found.")
    return info

def geotiffinfo_tiepoints(path: str) -> dict:
    """Return dict with ModelTiepointTag as Nx6 and width/height; lon/lat in EPSG:4326."""
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        tp = page.tags.get(33922)
        if tp is None:
            raise ValueError("ModelTiepointTag (33922) not found.")
        vals = np.array(tp.value, dtype=float)
        if vals.size % 6 != 0:
            raise ValueError("ModelTiepointTag length must be multiple of 6.")
        tiepoints = vals.reshape((-1, 6))
    with rasterio.open(path) as src:
        width, height, crs = src.width, src.height, src.crs
        if crs is not None and not crs.is_geographic:
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            X, Y = tiepoints[:, 3], tiepoints[:, 4]
            lon, lat = transformer.transform(X, Y)
            tiepoints[:, 3], tiepoints[:, 4] = lon, lat
    return {'GeoTIFFTags': {'ModelTiepointTag': tiepoints}, 'Width': width, 'Height': height}

def read_slc_interleaved_int16(path: str, width: int, height: int, strip_offset: int):
    """
    Read interleaved int16 SLC:
      shape (Width*2, Height) in 'F' (column-major); rows 0::2 -> real, 1::2 -> imag.
    Returns real, imag with shape (Width, Height).
    """
    with open(path, 'rb') as f:
        f.seek(int(strip_offset), io.SEEK_SET)
        count = width * 2 * height
        data = np.fromfile(f, dtype=np.int16, count=count)
    slcimage = np.reshape(data, (width * 2, height), order='F')
    realimage = slcimage[0::2, :]
    imagimage = slcimage[1::2, :]
    return realimage, imagimage


# ---------- Geodesy ----------

_GEOD = Geod(ellps='WGS84')

def distance_and_azimuth(lat1, lon1, lat2, lon2):
    """Inverse problem: distance (m) and forward azimuth (deg)."""
    lat1 = np.asarray(lat1, dtype=float); lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float); lon2 = np.asarray(lon2, dtype=float)
    az12, az21, dist = _GEOD.inv(lon1, lat1, lon2, lat2)
    return dist, az12

def track1_gc(lat0, lon0, az_deg, dist_m):
    """Direct problem: destination lat/lon after az_deg & dist_m."""
    lat0 = np.asarray(lat0, dtype=float); lon0 = np.asarray(lon0, dtype=float)
    az_deg = np.asarray(az_deg, dtype=float); dist_m = np.asarray(dist_m, dtype=float)
    lon1, lat1, _ = _GEOD.fwd(lon0, lat0, az_deg, dist_m)
    return lat1, lon1


# ---------- TIFF writer for complex patches ----------

def save_complex_patch_tiff(out_path: str, complex_patch: np.ndarray, description: Optional[str] = None):
    """
    Save a complex patch as single-band complex64 TIFF.
    Compatible with tifffile and GDAL (GDT_CFloat32).
    """
    # dtype complex64
    arr = np.asarray(complex_patch, dtype=np.complex64)
    tifffile.imwrite(
        out_path,
        arr,
        dtype=np.complex64,
        metadata=None,
        description=description or ''
    )

# ---------- SLCvalidation loader ----------

def _to_bool_isvessel(x) -> bool:
    s = str(x).strip().lower()
    return s in ('true', '1', 'yes', 'y', 't')

def load_slcvalidation_csv(csv_path: str) -> Dict[str, np.ndarray]:
    """
    Load SLCvalidation from CSV con columnas conocidas.
    Devuelve además de las claves base:
      - 'global_shoreline_vector_distance_from_shore_km'
      - 'xView3_shoreline_distance_from_shore_km'
      - 'confidence' (como string: LOW/MEDIUM/HIGH, etc.)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Not found {csv_path}")

    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required for this CSV loader (pip install pandas).") from e

    df = pd.read_csv(csv_path)

    required_base = [
        "SLC_product_identifier",
        "swath_index",
        "detect_lat",
        "detect_lon",
        "detect_scene_column",
        "detect_scene_row",
        "top",
        "left",
        "bottom",
        "right",
        "vessel_length_m",
        "is_vessel",
    ]
    missing = [c for c in required_base if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing required columns: {missing}")

    # columnas opcionales
    opt_cols = [
        "global_shoreline_vector_distance_from_shore_km",
        "xView3_shoreline_distance_from_shore_km",
        "confidence",
        "source"
    ]
    for c in opt_cols:
        if c not in df.columns:
            df[c] = np.nan

    scene_raw = df["SLC_product_identifier"].astype(str).str.strip()
    scene = scene_raw.str.replace(r"\.SAFE$", "", regex=True).to_numpy(dtype=str)

    swath_f = pd.to_numeric(df["swath_index"], errors="coerce").to_numpy(dtype=float)
    lat     = pd.to_numeric(df["detect_lat"], errors="coerce").to_numpy(dtype=float)
    lon     = pd.to_numeric(df["detect_lon"], errors="coerce").to_numpy(dtype=float)
    xpix    = pd.to_numeric(df["detect_scene_column"], errors="coerce").to_numpy(dtype=float)
    ypix    = pd.to_numeric(df["detect_scene_row"], errors="coerce").to_numpy(dtype=float)
    top     = pd.to_numeric(df["top"], errors="coerce").to_numpy(dtype=float)
    left    = pd.to_numeric(df["left"], errors="coerce").to_numpy(dtype=float)
    bottom  = pd.to_numeric(df["bottom"], errors="coerce").to_numpy(dtype=float)
    right   = pd.to_numeric(df["right"], errors="coerce").to_numpy(dtype=float)
    length  = pd.to_numeric(df["vessel_length_m"], errors="coerce").to_numpy(dtype=float)

    gshore  = pd.to_numeric(df["global_shoreline_vector_distance_from_shore_km"], errors="coerce").to_numpy(dtype=float)
    xshore  = pd.to_numeric(df["xView3_shoreline_distance_from_shore_km"], errors="coerce").to_numpy(dtype=float)

    conf    = (df["confidence"]
               .astype(str)
               .str.strip()
               .replace({'nan': '', 'NaN': '', 'None': ''})
               .str.upper()
               .to_numpy(dtype=str))
    
    source    = (df["source"]
                .astype(str)
                .str.strip()
                .replace({'nan': '', 'NaN': '', 'None': ''})
                .str.upper()
                .to_numpy(dtype=str))

    isv = df["is_vessel"]
    if isv.dtype == bool:
        isvessel = isv.to_numpy(dtype=bool)
    else:
        isvessel = (
            isv.astype(str).str.strip().str.lower()
            .isin(["true", "1", "yes", "y", "t"])
            .to_numpy(dtype=bool)
        )

    valid = (scene != "") & ~np.isnan(swath_f) & ~np.isnan(lat) & ~np.isnan(lon)

    scene   = scene[valid]
    swath_f = swath_f[valid]
    lat     = lat[valid]
    lon     = lon[valid]
    xpix    = xpix[valid]
    ypix    = ypix[valid]
    top     = top[valid]
    left    = left[valid]
    bottom  = bottom[valid]
    right   = right[valid]
    length  = length[valid]
    isvessel= isvessel[valid]
    gshore  = gshore[valid]
    xshore  = xshore[valid]
    conf    = conf[valid]  
    source  = source[valid] 

    swath = np.round(swath_f).astype(int)
    keep  = (swath >= 1) & (swath <= 3)

    return {
        "scene": scene[keep],
        "swath": swath[keep],
        "lat": lat[keep],
        "lon": lon[keep],
        "xpix": xpix[keep],
        "ypix": ypix[keep],
        "top": top[keep],
        "left": left[keep],
        "bottom": bottom[keep],
        "right": right[keep],
        "lengthpix": length[keep],  
        "isvessel": isvessel[keep],
        "global_shoreline_vector_distance_from_shore_km": gshore[keep],
        "xView3_shoreline_distance_from_shore_km": xshore[keep],
        "confidence": conf[keep], 
        "source": source[keep],    
        "keys": np.array(df.columns, dtype=str),
    }

# ---------- Main ----------


def main(
    grdproduct: str,
    slcproduct: str,
    ocnproduct: str,
    outputpath: str,
    slcvalidation: str,
    sizepatch: int
):
    """
    Process one Sentinel-1 scene (SLC/GRD/OCN triplet) using product names read from a CSV.
    Product folder paths are constructed as <script_dir>/products/{slc|grd|ocn}/<product_name>.
    Results are written to <outputpath> with subfolders label/, slc/, and grd/.

    Args:
        grdproduct (str): GRD SAFE product name (e.g., "S1A_IW_GRDH_...SAFE")
        slcproduct (str): SLC SAFE product name (e.g., "S1A_IW_SLC__...SAFE")
        ocnproduct (str): OCN SAFE product name (e.g., "S1A_IW_OCN__...SAFE")
        outputpath (str): Target output directory for this scene (e.g., ".../output/scene_001/")
        slcvalidation (str): Absolute (or valid) path to SLCvalidation.csv
    """

    import os
    import shutil
    import numpy as np
    from datetime import datetime
    from PIL import Image
    import tifffile

    # -------------------------------
    # Resolve base paths from script
    # -------------------------------
    script_dir = os.path.abspath(os.path.dirname(__file__))


    slc_path = os.path.join(script_dir, "inputs", "slc", slcproduct + ".SAFE")
    grd_path = os.path.join(script_dir, "inputs", "grd", grdproduct + ".SAFE")
    ocn_path = os.path.join(script_dir, "inputs", "ocn", ocnproduct + ".SAFE")

    # if not os.path.isdir(slc_path):
    #     raise FileNotFoundError(f"SLC product folder not found: {slc_path}")
    # if not os.path.isdir(grd_path):
    #     raise FileNotFoundError(f"GRD product folder not found: {grd_path}")
    # if not os.path.isdir(ocn_path):
    #     raise FileNotFoundError(f"OCN product folder not found: {ocn_path}")

    # ----------------------------------------
    # Prepare clean output subfolders per scene
    # ----------------------------------------
    path_label = os.path.join(outputpath, "label")
    path_slc   = os.path.join(outputpath, "slc")
    path_grd   = os.path.join(outputpath, "grd")

    for folder in (path_label, path_slc, path_grd):
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    # ---------------------------------------
    # Load SLCvalidation.csv from given path
    # ---------------------------------------
    if not os.path.isfile(slcvalidation):
        raise FileNotFoundError(f"SLCvalidation.csv not found at: {slcvalidation}")
    TT = load_slcvalidation_csv(slcvalidation)

    # ---------------------------------------
    # Processing parameters
    # ---------------------------------------
    patch_id  = 1

    # Derive a short ID from the SLC SAFE name (e.g., last 9..5 chars)
    slc_basename = os.path.basename(slcproduct)
    slc_id = slc_basename[-9:-5]  # preserves previous logic

    # Unpack SLCvalidation content
    scenes = TT['scene']
    swaths = TT['swath']
    latbbdd, lonbbdd = TT['lat'], TT['lon']
    xpixbbdd, ypixbbdd = TT['xpix'], TT['ypix']
    lengthpixbbdd = TT['lengthpix']
    toppixbbdd, leftpixbbdd = TT['top'], TT['left']
    botpixbbdd, rigthpixbbdd = TT['bottom'], TT['right']
    isvessel = TT['isvessel']
    gshore_km = TT.get('global_shoreline_vector_distance_from_shore_km', np.full_like(latbbdd, np.nan))
    xshore_km = TT.get('xView3_shoreline_distance_from_shore_km', np.full_like(latbbdd, np.nan))
    confbbdd  = TT.get('confidence', np.array(['']*len(latbbdd), dtype=str))
    sourcebbdd= TT.get('source', np.array(['']*len(latbbdd), dtype=str))



    # ---------------------------------------
    # Utility to normalize internal SAFE href
    # (remove leading "./" and Windows backslashes)
    # ---------------------------------------
    def _rel_to_platform(p: str) -> str:
        return os.path.normpath(str(p).lstrip("./").replace("\\", os.sep))

    # ---------------------------------------
    # Read SAFE manifests from each product
    # ---------------------------------------
    metadata_SLC = xml2struct(os.path.join(slc_path, 'manifest.safe'))
    metadata_GRD = xml2struct(os.path.join(grd_path, 'manifest.safe'))
    metadata_OCN = xml2struct(os.path.join(ocn_path, 'manifest.safe'))

    # Indices of the data objects (as in previous version)
    icalvh = [3, 6, 9]
    icalvv = [12, 15, 18]
    ivh    = [21, 22, 23]
    ivv    = [24, 25, 26]
    avh    = [1, 4, 7]
    avv    = [10, 13, 16]

    # Extract inner file paths for SLC
    root_slc = next(iter(metadata_SLC.keys()))
    dso_SLC = metadata_SLC[root_slc]['dataObjectSection']['dataObject']
    if not isinstance(dso_SLC, list):
        dso_SLC = [dso_SLC]

    slcfile_vh, slcfile_vv = {}, {}
    calfile_vh, calfile_vv = {}, {}
    annotation_vh, annotation_vv = {}, {}

    for ind in range(3):
        slcfile_vh[ind+1] = _rel_to_platform(
            dso_SLC[ivh[ind]-1]['byteStream']['fileLocation']['Attributes']['href']
        )
        slcfile_vv[ind+1] = _rel_to_platform(
            dso_SLC[ivv[ind]-1]['byteStream']['fileLocation']['Attributes']['href']
        )
        calfile_vh[ind+1] = _rel_to_platform(
            dso_SLC[icalvh[ind]-1]['byteStream']['fileLocation']['Attributes']['href']
        )
        calfile_vv[ind+1] = _rel_to_platform(
            dso_SLC[icalvv[ind]-1]['byteStream']['fileLocation']['Attributes']['href']
        )
        annotation_vh[ind+1] = _rel_to_platform(
            dso_SLC[avh[ind]-1]['byteStream']['fileLocation']['Attributes']['href']
        )
        annotation_vv[ind+1] = _rel_to_platform(
            dso_SLC[avv[ind]-1]['byteStream']['fileLocation']['Attributes']['href']
        )

    # Extract inner file paths for GRD
    root_grd = next(iter(metadata_GRD.keys()))
    dso_GRD = metadata_GRD[root_grd]['dataObjectSection']['dataObject']
    if not isinstance(dso_GRD, list):
        dso_GRD = [dso_GRD]
    # Keep original indexing convention
    grdfile_vh = _rel_to_platform(dso_GRD[9-1]['byteStream']['fileLocation']['Attributes']['href'])
    grdfile_vv = _rel_to_platform(dso_GRD[10-1]['byteStream']['fileLocation']['Attributes']['href'])

    # Extract inner file paths for OCN
    root_ocn = next(iter(metadata_OCN.keys()))
    dso_OCN = metadata_OCN[root_ocn]['dataObjectSection']['dataObject']
    if not isinstance(dso_OCN, list):
        dso_OCN = [dso_OCN]
    ocnfile = _rel_to_platform(dso_OCN[3-1]['byteStream']['fileLocation']['Attributes']['href'])

    # ---------------------------------------
    # Coregister XView and SAR (is_vessel == true)
    # ---------------------------------------
    scene_key = os.path.basename(slcproduct).replace(".SAFE", "")
    idx_scene_all = np.where(scenes == scene_key)[0]
    idx_scene = idx_scene_all[isvessel[idx_scene_all]]

    vessel = np.zeros((len(idx_scene), 13), dtype=float)
    vessel_conf = np.empty((len(idx_scene),), dtype=object)
    vessel_source = np.empty((len(idx_scene),), dtype=object)

    for i2, idx in enumerate(idx_scene):
        vessel[i2, :] = [
            latbbdd[idx], lonbbdd[idx],          # 0..1
            xpixbbdd[idx], ypixbbdd[idx],        # 2..3
            lengthpixbbdd[idx],                  # 4
            toppixbbdd[idx], leftpixbbdd[idx],   # 5..6
            botpixbbdd[idx], rigthpixbbdd[idx],  # 7..8
            swaths[idx],                         # 9
            1.0 if isvessel[idx] else 0.0,       # 10 numeric 0/1
            gshore_km[idx],                      # 11
            xshore_km[idx],                      # 12
        ]
        vessel_conf[i2] = confbbdd[idx]         # confidence string per vessel
        vessel_source[i2] = sourcebbdd[idx]     # source string per vessel

    # ---------------------------------------
    # Legend table (accumulated per patch)
    # ---------------------------------------
    legend = {f'legend_{slc_id}': np.array([[1, 0, 0]], dtype=int)}

    # ---------------------------------------
    # Burst info per swath (deburst mapping and poly TF)
    # ---------------------------------------
    dataswath = {}
    burstoverlapping = {}

    for swath_id in range(1, 4):
        # SLC VV geotiff + tiepoints (use slc_path)
        slc_vv_path = os.path.join(slc_path, slcfile_vv[swath_id])
        slc_info = imfinfo_tiff(slc_vv_path)
        slc_geo  = geotiffinfo_tiepoints(slc_vv_path)

        # Annotation (use slc_path)
        raf = xml2struct(os.path.join(slc_path, annotation_vv[swath_id]))
        ia = raf['product']['imageAnnotation']['imageInformation']
        pi = raf['product']['generalAnnotation']['productInformation']
        st = raf['product']['swathTiming']

        azsamplet = float(_as_text(ia['azimuthTimeInterval']))
        azsamples = float(_as_text(ia['azimuthPixelSpacing']))
        pxsamplet = 1.0 / float(_as_text(pi['rangeSamplingRate']))
        pxsamples = float(_as_text(ia['rangePixelSpacing']))
        heading   = float(_as_text(pi['platformHeading']))
        linesburst= int(float(_as_text(st['linesPerBurst'])))
        nburst    = int(float(st['burstList']['Attributes']['count']))

        timeaxis = np.zeros(nburst, dtype=float)
        bursts = st['burstList']['burst']
        if not isinstance(bursts, list):
            bursts = [bursts]
        for ib in range(nburst):
            timeaxis[ib] = float(_as_text(bursts[ib]['azimuthAnxTime']))

        xsampling = int(np.sum(slc_geo['GeoTIFFTags']['ModelTiepointTag'][:, 1] == 0))
        tiepoints = slc_geo['GeoTIFFTags']['ModelTiepointTag']  # Nx6: I,J,K,lon,lat,Z

        # Deburst map (1-based convention)
        iibb = np.arange(1, tiepoints.shape[0] + 1, xsampling)
        tstrans = np.zeros((linesburst * nburst, 2), dtype=int)
        ovlp = np.round((timeaxis - timeaxis[0]) / azsamplet).astype(int) + 1
        burstoverlapping[swath_id] = ovlp.copy()

        for indb in range(1, nburst + 1):
            debursting = np.arange(ovlp[indb - 1], ovlp[indb - 1] + linesburst, dtype=int)
            bursting   = np.arange(1 + (indb - 1) * linesburst, indb * linesburst + 1, dtype=int)
            tstrans[bursting - 1, 0] = debursting
            tstrans[bursting - 1, 1] = bursting

        # Per-burst polynomial transforms: (sample,line) -> (lon,lat)
        tform_xy2ll_list = []
        for indb in range(1, nburst + 1):
            idx0 = iibb[indb - 1] - 1
            idx1 = (iibb[indb] - 1) if indb < len(iibb) else (iibb[indb - 1] - 1)
            latA = tiepoints[idx0:idx0 + xsampling, 4]
            lonA = tiepoints[idx0:idx0 + xsampling, 3]
            latB = tiepoints[idx1:idx1 + xsampling, 4]
            lonB = tiepoints[idx1:idx1 + xsampling, 3]

            dmov, amov = distance_and_azimuth(latA, lonA, latB, lonB)
            lats = np.zeros((xsampling, linesburst), dtype=float)
            lons = np.zeros_like(lats)
            if indb == nburst:
                incfinal = dmov / (linesburst - 1 + 1e-12)
                for irej in range(1, linesburst + 1):
                    lat_j, lon_j = track1_gc(latA, lonA, amov, (irej - 1) * incfinal)
                    lats[:, irej - 1], lons[:, irej - 1] = lat_j, lon_j
            else:
                for irej in range(1, linesburst + 1):
                    dist_vec = np.ones(xsampling) * (irej - 1) * azsamples
                    lat_j, lon_j = track1_gc(latA, lonA, amov, dist_vec)
                    lats[:, irej - 1], lons[:, irej - 1] = lat_j, lon_j

            lons_vec = lons.reshape(-1, order='F')
            lats_vec = lats.reshape(-1, order='F')
            rowaxis  = np.arange(0, linesburst, dtype=float)
            rowmat   = np.tile(rowaxis, (xsampling, 1))
            rowvec   = rowmat.reshape(-1, order='F')
            colvec   = tiepoints[idx0:idx0 + xsampling, 0].astype(float)  # 1-based samples
            colmat   = np.tile(colvec[:, None], (1, linesburst))
            colvec2  = colmat.reshape(-1, order='F')

            pts_xy = np.column_stack([colvec2, rowvec])
            tform_xy2ll = fitgeotform2d(pts_xy, np.column_stack([lons_vec, lats_vec]), degree=4)
            tform_xy2ll_list.append(tform_xy2ll)

        dataswath[swath_id] = {
            'linesburst': linesburst,
            'nburst': nburst,
            'iibb': iibb,
            'tstrans': tstrans,
            'tform_xy2ll_list': tform_xy2ll_list
        }

    # ---------------------------------------
    # SLC VV tiling: iterate bursts and 512-tiles
    # ---------------------------------------
    for swath_id in range(1, 4):
        ds = dataswath[swath_id]
        linesburst = ds['linesburst']
        nburst = ds['nburst']
        tstrans = ds['tstrans']
        ovlp = burstoverlapping[swath_id]

        vessel_swath = vessel[vessel[:, 9] == swath_id, :].copy()
        vessel_conf_swath = vessel_conf[vessel[:, 9] == swath_id]
        vessel_source_swath = vessel_source[vessel[:, 9] == swath_id]
        if vessel_swath.size > 0:
            vessel_swath[:, 3] = ovlp[-1] + linesburst - vessel_swath[:, 3]

        # Attempt to read OCN wind fields
        try:
            from netCDF4 import Dataset as NetCDF
            with NetCDF(os.path.join(ocn_path, ocnfile), 'r') as ds_ocn:
                wind_dir = ds_ocn.variables['owiWindDirection'][:]
                wind_spd = ds_ocn.variables['owiWindSpeed'][:]
                wind_lat = ds_ocn.variables['owiLat'][:]
                wind_lon = ds_ocn.variables['owiLon'][:]
        except Exception:
            wind_dir = wind_spd = wind_lat = wind_lon = None

        # Read VV complex image (real/imag interleaved int16)
        slc_vv_path = os.path.join(slc_path, slcfile_vv[swath_id])
        slc_info = imfinfo_tiff(slc_vv_path)
        realimg, imagimg = read_slc_interleaved_int16(
            slc_vv_path, slc_info['Width'], slc_info['Height'], slc_info['StripOffsets'][0]
        )

        # Read calibration metadata for VV
        cal_meta_vv = xml2struct(os.path.join(slc_path, calfile_vv[swath_id]))
        cal_list = cal_meta_vv['calibration']['calibrationVectorList']['calibrationVector']
        dn_text = cal_list[0]['dn']['Text'] if isinstance(cal_list, list) else cal_list['dn']['Text']
        calvv_len = len(str(dn_text).strip().split())

        # Iterate bursts
        for indb in range(1, nburst + 1):
            imageburst = (
                realimg[:, (indb - 1) * linesburst: indb * linesburst] +
                1j * imagimg[:, (indb - 1) * linesburst: indb * linesburst]
            ).T
            tform_xy2ll = ds['tform_xy2ll_list'][indb - 1]

            # Iterate 512x512 patches
            for ipatchsample in range(1, slc_info['Width'] + 1, sizepatch):
                for ipatchline in range(1, linesburst + 1, sizepatch):
                    if ipatchsample + sizepatch - 1 > slc_info['Width']:
                        continue
                    if ipatchline + sizepatch - 1 > linesburst:
                        continue

                    patch = imageburst[
                        ipatchline - 1: ipatchline - 1 + sizepatch,
                        ipatchsample - 1: ipatchsample - 1 + sizepatch
                    ]
                    if np.sum(patch == 0) > (sizepatch * sizepatch) / 4:
                        continue

                    # Save VV complex patch
                    out_tif_vv = os.path.join(path_slc, f'DB_OPENSAR_DVD_{patch_id}_SLC_VV.tiff')
                    save_complex_patch_tiff(out_tif_vv, patch, description=f'calfactorvv_len={calvv_len}')

                    # Build metadata XML (corner lon/lat and SAR indices)
                    scene_id = f'DB_OPENSAR_DVD_{patch_id}'
                    today = datetime.today()
                    out_date = today.year * 10000 + today.month * 100 + today.day

                    corner_sample = np.array([ipatchsample,
                                              ipatchsample + sizepatch - 1,
                                              ipatchsample + sizepatch - 1,
                                              ipatchsample], dtype=float)
                    corner_line   = np.array([ipatchline,
                                              ipatchline,
                                              ipatchline + sizepatch - 1,
                                              ipatchline + sizepatch - 1,], dtype=float)

                    lon_c, lat_c = tform_xy2ll.transform(corner_sample - 1, corner_line - 1)
                    lon_c = _wrap_lon180(lon_c)

                    outxml = {
                        'outxml': {
                            'Scene_Info': {
                                'Scene_ID': scene_id,
                                'Date': out_date,
                                'Version': 1,
                                'CaseStudy': 'DarkVesselDetection'
                            },
                            'SARData': {
                                'SAR_Mission': slcproduct[0:3],
                                'SARProduct': slcproduct,
                                'SLCSwath': swath_id,
                                'Time_Interval': {
                                    'Start': f"{slcproduct[17:25]}_{slcproduct[26:32]}",
                                    'Stop':  f"{slcproduct[33:41]}_{slcproduct[42:48]}"
                                }
                            },
                            'ProcessingData': {
                                'Corner_Coord': {
                                    'SARData_Sample': ' '.join(str(int(x)) for x in corner_sample),
                                    'SARData_Line':   ' '.join(str(int(x)) for x in (corner_line + (indb - 1) * linesburst)),
                                    'Scene_Sample': f'1 {sizepatch} {sizepatch} 1',
                                    'Scene_Line':   f'1 1 {sizepatch} {sizepatch}',
                                    'Latitude':  ' '.join(fmt10(x) for x in lat_c),
                                    'Longitude': ' '.join(fmt10(x) for x in lon_c),
                                },
                                'StatisticsReport': {}
                            }
                        }
                    }

                    # Add wind data (nearest OCN grid-point) if available
                    if (wind_lat is not None) and (wind_lon is not None):
                        lon_center, lat_center = tform_xy2ll.transform(
                            ipatchsample - 1 + sizepatch // 2, ipatchline - 1 + sizepatch // 2
                        )
                        d2 = (wind_lat - lat_center) ** 2 + (wind_lon - lon_center) ** 2
                        i_flat = int(np.argmin(d2))
                        i_lat, i_lon = np.unravel_index(i_flat, d2.shape)
                        
                        # Handle masked arrays from NetCDF safely
                        if wind_dir is not None:
                            wdir_val = wind_dir[i_lat, i_lon]
                            wdir = float(wdir_val) if not np.ma.is_masked(wdir_val) else np.nan
                        else:
                            wdir = np.nan
                        
                        if wind_spd is not None:
                            wspd_val = wind_spd[i_lat, i_lon]
                            wspd = float(wspd_val) if not np.ma.is_masked(wspd_val) else np.nan
                        else:
                            wspd = np.nan
                        
                        outxml['outxml']['ProcessingData']['WindData'] = {'Direction': wdir, 'Speed': wspd}

                    # Vessel-in-polygon test (in sample/line space)
                    poly_x = corner_sample.copy()
                    corner_line_global = corner_line + (indb - 1) * linesburst  # 1-based global bursting lines
                    idx_rows = (corner_line_global.astype(int) - 1).clip(0, tstrans.shape[0] - 1)
                    poly_y = tstrans[idx_rows, 0].astype(float)  # debursting line coords (1-based)

                    if vessel_swath.size > 0:
                        vx = vessel_swath[:, 2]  # xpix
                        vy = vessel_swath[:, 3]  # adjusted y

                        def inpoly(xs, ys, px, py):
                            xs = np.asarray(xs); ys = np.asarray(ys)
                            n = len(px)
                            inside = np.zeros(xs.shape, dtype=bool)
                            j = n - 1
                            for i in range(n):
                                xi, yi = px[i], py[i]
                                xj, yj = px[j], py[j]
                                cond = ((yi > ys) != (yj > ys)) & (xs < (xj - xi) * (ys - yi) / (yj - yi + 1e-12) + xi)
                                inside ^= cond
                                j = i
                            return inside

                        vessel_mask = inpoly(vx, vy, poly_x, poly_y)
                    else:
                        vessel_mask = np.zeros((0,), dtype=bool)

                    if vessel_mask.size > 0 and np.any(vessel_mask):
                        vdef = vessel_swath[vessel_mask, :].copy()
                        conf_swath_masked = vessel_conf_swath[vessel_mask]
                        source_swath_masked = vessel_source_swath[vessel_mask]

                        zzz = np.zeros(vdef.shape[0], dtype=float)
                        zz = np.zeros(vdef.shape[0], dtype=int)
                        for ivv_idx in range(vdef.shape[0]):
                            rows = np.where(tstrans[:, 0] == int(vdef[ivv_idx, 3]))[0]
                            if rows.size == 0:
                                zz[ivv_idx] = -1
                                zzz[ivv_idx] = 0.0
                                continue
                            candidate = rows[0]
                            z_global = int(tstrans[candidate, 1])
                            local_line = z_global - (indb - 1) * linesburst
                            if local_line < 1 and rows.size > 1:
                                candidate = rows[1]
                                z_global = int(tstrans[candidate, 1])
                                local_line = z_global - (indb - 1) * linesburst
                            zz[ivv_idx] = z_global
                            if local_line < 1 or local_line > linesburst:
                                zzz[ivv_idx] = 0.0
                            else:
                                sample_1b = int(vdef[ivv_idx, 2])
                                il = local_line - 1
                                ic = sample_1b - 1
                                if 0 <= il < imageburst.shape[0] and 0 <= ic < imageburst.shape[1]:
                                    zzz[ivv_idx] = np.abs(imageburst[il, ic])
                                else:
                                    zzz[ivv_idx] = 0.0

                        # --- NEW FILTER: drop only low-intensity vessels very near burst start (first 20 local lines) ---
                        # Compute local line index for each candidate relative to this burst (1-based, to match your logic)
                        local_line_vec = zz - (indb - 1) * linesburst

                        # Build rejection mask: low intensity AND too close to top of the burst
                        reject = (zzz < 10.0) & (local_line_vec < 20)

                        # Apply rejection to all parallel arrays
                        if np.any(reject):
                            keep_mask = ~reject
                            vdef = vdef[keep_mask, :].copy()
                            zz = zz[keep_mask].copy()
                            zzz = zzz[keep_mask].copy()
                            conf_swath_masked = np.asarray(conf_swath_masked, dtype=object)[keep_mask]
                            source_swath_masked = np.asarray(source_swath_masked, dtype=object)[keep_mask]
                        else:
                            keep_mask = np.ones_like(zz, dtype=bool)

                        # Number of ships AFTER this filtering
                        nships = int(zz.shape[0])
                        outxml['outxml']['ProcessingData']['StatisticsReport']['Number_of_ships'] = nships

                        if nships >= 1:
                            # Mark this patch as positive in the legend
                            legend[f'legend_{slc_id}'] = np.vstack([legend[f'legend_{slc_id}'],
                                                                   np.array([[patch_id, 1, swath_id]])])

                            # Build ships from ALL remaining (no extra intensity threshold)
                            v_kept = vdef
                            zz_kept = zz
                            conf_kept = np.asarray(conf_swath_masked, dtype=object)
                            source_kept = np.asarray(source_swath_masked, dtype=object)

                            v_lat = v_kept[:, 0]
                            v_lon = v_kept[:, 1]
                            v_size = v_kept[:, 4]
                            v_h = (v_kept[:, 5] - v_kept[:, 7]) / 2.0
                            v_w = (v_kept[:, 6] - v_kept[:, 8]) / 2.0
                            v_sample = v_kept[:, 2].astype(int)
                            v_line_global = zz_kept.astype(int)

                            ships = []
                            for k in range(nships):
                                scene_sample = int(v_sample[k] - np.min(corner_sample) + 1)
                                scene_line = int(v_line_global[k] - np.min(corner_line) - (indb - 1) * linesburst + 1)
                                top = int(np.clip(scene_line - round(v_h[k] / 2.0), 1, sizepatch))
                                bottom = int(np.clip(scene_line + round(v_h[k] / 2.0), 1, sizepatch))
                                left = int(np.clip(scene_sample - round(v_w[k] / 2.0), 1, sizepatch))
                                right = int(np.clip(scene_sample + round(v_w[k] / 2.0), 1, sizepatch))
                                if left > right:
                                    left, right = right, left

                                isv_k = bool(v_kept[k, 10] > 0.5)
                                gshore_k = v_kept[k, 11]
                                xshore_k = v_kept[k, 12]
                                conf_k = str(conf_kept[k])
                                source_k = str(source_kept[k])

                                ships.append({
                                    'Name': f'Ship_{k + 1}',
                                    'Centroid_Position': {
                                        'Latitude': float(v_lat[k]),
                                        'Longitude': float(v_lon[k]),
                                        'SARData_Sample': int(v_sample[k]),
                                        'SARData_Line': int(v_line_global[k]),
                                        'Scene_Sample': scene_sample,
                                        'Scene_Line': scene_line,
                                    },
                                    'Size': float(v_size[k]),
                                    'BoundingBox': {'Top': top, 'Left': left, 'Bottom': bottom, 'Right': right},
                                    'is_vessel': 'true' if isv_k else 'false',
                                    'global_shoreline_vector_distance_from_shore_km': fmt10(gshore_k),
                                    'xView3_shoreline_distance_from_shore_km': fmt10(xshore_k),
                                    'confidence': conf_k,
                                    'source': source_k,
                                })
                            outxml['outxml']['ProcessingData']['List_of_ships'] = {
                                'Ship': ships if len(ships) > 1 else ships[0]
                            }
                        else:
                            # No ships left after the new filter
                            legend[f'legend_{slc_id}'] = np.vstack([legend[f'legend_{slc_id}'],
                                                                   np.array([[patch_id, 0, swath_id]])])
                    else:
                        outxml['outxml']['ProcessingData']['StatisticsReport']['Number_of_ships'] = 0
                        legend[f'legend_{slc_id}'] = np.vstack([legend[f'legend_{slc_id}'],
                                                               np.array([[patch_id, 0, swath_id]])])

                    # Write per-patch XML
                    xml_out = os.path.join(path_label, f'DB_OPENSAR_DVD_{patch_id}.xml')
                    struct2xml(outxml, xml_out)

                    patch_id += 1

    # ---------------------------------------
    # Save legend as CSV-like text
    # ---------------------------------------
    legend_key = f'legend_{slc_id}'
    legend_arr = legend[legend_key]
    if legend_arr.shape[0] >= 1 and np.array_equal(legend_arr[0], [1, 0, 0]):
        legend_arr = legend_arr[1:]
    log_path = os.path.join(path_label, f'legend_{slc_id}.txt')
    np.savetxt(
        log_path,
        legend_arr,
        fmt='%d',
        delimiter=',',
        header='patch_id,has_ship,swath_id',
        comments=''  # no '#' prefix in header
    )
    print(f"[INFO] Legend saved in: {log_path} (rows: {legend_arr.shape[0]})")

    if patch_id == 1:
        print("[WARN] No valid patches generated; skipping VH/GRD.")
        return

    # ---------------------------------------
    # SLC VH tiling using SARData indices from XML
    # ---------------------------------------
    for swath_id in range(1, 4):
        slc_vh_info = imfinfo_tiff(os.path.join(slc_path, slcfile_vh[swath_id]))
        slc_vh_path = os.path.join(slc_path, slcfile_vh[swath_id])
        real_vh, imag_vh = read_slc_interleaved_int16(
            slc_vh_path, slc_vh_info['Width'], slc_vh_info['Height'], slc_vh_info['StripOffsets'][0]
        )

        cal_meta_vh = xml2struct(os.path.join(slc_path, calfile_vh[swath_id]))
        cal_list_vh = cal_meta_vh['calibration']['calibrationVectorList']['calibrationVector']
        dn_text_vh = cal_list_vh[0]['dn']['Text'] if isinstance(cal_list_vh, list) else cal_list_vh['dn']['Text']
        calvh_len = len(str(dn_text_vh).strip().split())

        mask_sw = (legend[f'legend_{slc_id}'][:, 2] == swath_id)
        ids_sw  = legend[f'legend_{slc_id}'][mask_sw, 0].astype(int)

        for pid in ids_sw:
            xml_path = os.path.join(path_label, f'DB_OPENSAR_DVD_{pid}.xml')
            if not os.path.exists(xml_path):
                print(f"[WARN] XML not found for ID {pid}; skipping VH.")
                continue
            od = xml2struct(xml_path)
            cc = od['outxml']['ProcessingData']['Corner_Coord']
            cs = [int(x) for x in _as_text(cc['SARData_Sample']).split()]
            cl = [int(x) for x in _as_text(cc['SARData_Line']).split()]
            cs_min, cs_max = min(cs), max(cs)
            cl_min, cl_max = min(cl), max(cl)

            patch_vh = (
                real_vh[cs_min - 1:cs_max, cl_min - 1:cl_max] +
                1j * imag_vh[cs_min - 1:cs_max, cl_min - 1:cl_max]
            ).T

            out_tif_vh = os.path.join(path_slc, f'DB_OPENSAR_DVD_{pid}_SLC_VH.tiff')
            save_complex_patch_tiff(out_tif_vh, patch_vh, description=f'calfactorvh_len={calvh_len}')

    # ---------------------------------------
    # GRD tiling (VV/VH) from per-patch lon/lat
    # ---------------------------------------
    grd_vv_path = os.path.join(grd_path, grdfile_vv)
    grd_vh_path = os.path.join(grd_path, grdfile_vh)
    grd_geo_vv  = geotiffinfo_tiepoints(grd_vv_path)
    grd_geo_vh  = geotiffinfo_tiepoints(grd_vh_path)

    tp_vv = grd_geo_vv['GeoTIFFTags']['ModelTiepointTag']
    tp_vh = grd_geo_vh['GeoTIFFTags']['ModelTiepointTag']
    tform_grd_vv = fitgeotform2d(tp_vv[:, 0:2], tp_vv[:, 3:5], degree=4)
    tform_grd_vh = fitgeotform2d(tp_vh[:, 0:2], tp_vh[:, 3:5], degree=4)

    grddatavv = np.array(Image.open(grd_vv_path))
    grddatavh = np.array(Image.open(grd_vh_path))

    all_ids = legend[f'legend_{slc_id}'][:, 0].astype(int)
    for pid in all_ids:
        xml_path = os.path.join(path_label, f'DB_OPENSAR_DVD_{pid}.xml')
        if not os.path.exists(xml_path):
            print(f"[WARN] XML not found for ID {pid}; skipping GRD.")
            continue
        od = xml2struct(xml_path)
        cc = od['outxml']['ProcessingData']['Corner_Coord']
        lats = np.array([float(x) for x in _as_text(cc['Latitude']).split()])
        lons = np.array([float(x) for x in _as_text(cc['Longitude']).split()])

        x_vv, y_vv = tform_grd_vv.transform_inverse(lons, lats)
        x_vh, y_vh = tform_grd_vh.transform_inverse(lons, lats)

        def clamp_bounds(x, y, w, h):
            x = np.round(x).astype(int); y = np.round(y).astype(int)
            x[x <= 0] = 1; x[x >= w] = w
            y[y <= 0] = 1; y[y >= h] = h
            x0, x1 = int(np.min(x)) - 1, int(np.max(x))
            y0, y1 = int(np.min(y)) - 1, int(np.max(y))
            if x1 <= x0: x1 = min(x0 + 1, w)
            if y1 <= y0: y1 = min(y0 + 1, h)
            return x0, x1, y0, y1

        x0, x1, y0, y1 = clamp_bounds(x_vv, y_vv, grd_geo_vv['Width'], grd_geo_vv['Height'])
        tile_vv = grddatavv[y0:y1, x0:x1]
        if tile_vv.size > 0:
            tifffile.imwrite(
                os.path.join(path_grd, f'DB_OPENSAR_DVD_{pid}_GRD_VV.tiff'),
                tile_vv, photometric='minisblack', metadata=None
            )

        x0, x1, y0, y1 = clamp_bounds(x_vh, y_vh, grd_geo_vh['Width'], grd_geo_vh['Height'])
        tile_vh = grddatavh[y0:y1, x0:x1]
        if tile_vh.size > 0:
            tifffile.imwrite(
                os.path.join(path_grd, f'DB_OPENSAR_DVD_{pid}_GRD_VH.tiff'),
                tile_vh, photometric='minisblack', metadata=None
            )

    print("[DONE] Processing finished.")


if __name__ == '__main__':
    main()
