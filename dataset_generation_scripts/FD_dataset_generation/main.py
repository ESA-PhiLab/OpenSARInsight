"""
Flood Detection (FD) patch tiling pipeline for Sentinel-1 SLC/GRD products.

Corrections applied
-------------------
- Removed invalid direct main() call without arguments.
- Fixed missing LineString / MultiLineString imports.
- Fixed dummy legend row issue by initializing an empty legend.
- Fixed malformed FloodEvent XML nesting.
- Fixed GRD crop 1-based / 0-based indexing.
- Removed dead FloodEvent polys append block.
- Added optional USE_AOI_EXACT_POLYGON argument.
- Added safer geometry validation with buffer(0) cleanup where useful.
- Added safer TIFF tag access errors.

Expected to be called from a launcher, for example:
    from main import main
    main(inputpath, SLC_product, GRD_product, extrap, subextrap, outputpath, SIZEPATCH)
"""

# =========================
# Standard library imports
# =========================
import os
import re
from datetime import datetime
import xml.etree.ElementTree as ET
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

# =========================
# Numerical / scientific libraries
# =========================
import numpy as np
from scipy.interpolate import LinearNDInterpolator

# =========================
# Image and raster I/O
# =========================
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import tifffile
import rasterio

# =========================
# Coordinate reference systems / projections
# =========================
from pyproj import Transformer, CRS

# =========================
# Shapefile and vector geometry
# =========================
import shapefile
import geopandas as gpd
from shapely.geometry import Polygon, box, Point, LineString, MultiLineString
from shapely.ops import unary_union


# =========================
# Formatting utilities
# =========================

def fmt(x):
    return str(
        Decimal(str(float(x))).quantize(
            Decimal("0.0000000001"),
            rounding=ROUND_HALF_UP,
        )
    )


# =========================
# XML utilities
# =========================

def _safe_tag_name(name: str) -> str:
    """Normalize XML tag names to valid simple tags."""
    return name.replace("-", "_").replace(":", "_").replace(".", "_")


def xml2struct(file):
    """Parse XML file into a nested dictionary structure."""
    if isinstance(file, ET.Element):
        root = file
    else:
        if not os.path.exists(file):
            if not str(file).lower().endswith(".xml") and os.path.exists(str(file) + ".xml"):
                file = str(file) + ".xml"
            else:
                raise FileNotFoundError(f"File {file} does not exist")
        root = ET.parse(file).getroot()

    def parse_attributes(elem):
        return {_safe_tag_name(k): v for k, v in elem.attrib.items()}

    def parse_children(elem):
        children = {}
        ptext = {}
        for child in list(elem):
            name = _safe_tag_name(child.tag)
            text = (child.text or "").strip()
            attr = parse_attributes(child)
            childs, _ = parse_children(child)
            node = {}
            if childs:
                node.update(childs)
            if text:
                node["Text"] = text
            if attr:
                node["Attributes"] = attr
            if name in children:
                if not isinstance(children[name], list):
                    children[name] = [children[name]]
                children[name].append(node)
            else:
                children[name] = node

        txt = (elem.text or "").strip()
        if not children and txt:
            ptext["Text"] = txt
        return children, ptext

    children, ptext = parse_children(root)
    s = {_safe_tag_name(root.tag): children}
    if ptext:
        s[_safe_tag_name(root.tag)].update(ptext)
    attrs = parse_attributes(root)
    if attrs:
        s[_safe_tag_name(root.tag)]["Attributes"] = attrs
    return s


def struct2xml(s, file):
    """Serialize a MATLAB-like dictionary structure back to XML."""
    if not isinstance(s, dict) or len(s.keys()) != 1:
        raise ValueError("There must be a single root field in the input structure.")

    root_name = list(s.keys())[0]
    root_elem = ET.Element(root_name)

    def build(elem, dct):
        if not isinstance(dct, dict):
            elem.text = str(dct)
            return

        dct = dct.copy()

        attrs = dct.pop("Attributes", None)
        if isinstance(attrs, dict):
            for ak, av in attrs.items():
                elem.set(ak, str(av))

        txt = dct.pop("Text", None)
        if txt is not None:
            elem.text = str(txt)

        for k, v in dct.items():
            if isinstance(v, list):
                for item in v:
                    child = ET.SubElement(elem, k)
                    build(child, item)
            elif isinstance(v, dict):
                child = ET.SubElement(elem, k)
                build(child, v)
            else:
                child = ET.SubElement(elem, k)
                child.text = str(v)

    build(root_elem, s[root_name])

    if not str(file).lower().endswith(".xml"):
        file = str(file) + ".xml"

    ET.ElementTree(root_elem).write(file, encoding="utf-8", xml_declaration=True)


# =========================
# SAFE / FD helper utilities
# =========================

def _extract_times_from_safe_name(name: str):
    """
    Extract acquisition start/stop times from a SAFE name/path.

    Returns:
        dict: {'start': 'YYYYMMDD_HHMMSS', 'stop': 'YYYYMMDD_HHMMSS'}
    """
    hits = re.findall(r"(\d{8})T(\d{6})", name)
    if len(hits) >= 2:
        return {
            "start": f"{hits[0][0]}_{hits[0][1]}",
            "stop": f"{hits[1][0]}_{hits[1][1]}",
        }
    return {"start": "", "stop": ""}


def _href_to_path(href: str) -> str:
    """Normalize SAFE manifest hrefs to operating-system paths."""
    s = str(href).strip()
    if s.startswith("./") or s.startswith(".\\"):
        s = s[2:]
    return os.path.normpath(s)


def _parse_dn_vector(dn_text):
    """Parse calibration DN vectors, space- or comma-separated."""
    s = str(dn_text).strip()
    arr = np.fromstring(s, sep=" ")
    if arr.size == 0:
        arr = np.array([float(p) for p in s.replace(",", " ").split()])
    return arr


def _as_list(x):
    return x if isinstance(x, list) else [x]


# =========================
# GeoTIFF / SLC utilities
# =========================

def imfinfo_tiff(path):
    """Read basic TIFF metadata."""
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        if "ImageWidth" not in page.tags or "ImageLength" not in page.tags:
            raise RuntimeError(f"TIFF is missing ImageWidth/ImageLength tags: {path}")
        if 273 not in page.tags:
            raise RuntimeError(f"TIFF is missing StripOffsets tag: {path}")
        return {
            "Width": int(page.tags["ImageWidth"].value),
            "Height": int(page.tags["ImageLength"].value),
            "StripOffsets": [int(x) for x in np.ravel(page.tags[273].value)],
        }


def geotiffinfo_tiepoints(path):
    """Read georeferencing tiepoints from GeoTIFF."""
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        if 33922 not in page.tags:
            raise RuntimeError(f"TIFF is missing ModelTiepointTag: {path}")
        tp = np.array(page.tags[33922].value, dtype=float).reshape((-1, 6))

    with rasterio.open(path) as src:
        width, height = src.width, src.height
        if src.crs and not src.crs.is_geographic:
            tr = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            lon, lat = tr.transform(tp[:, 3], tp[:, 4])
            tp[:, 3], tp[:, 4] = lon, lat

    return {"GeoTIFFTags": {"ModelTiepointTag": tp}, "Width": width, "Height": height}


def read_slc_interleaved_int16(path, width, height, strip_offset):
    """Read interleaved INT16 SLC data."""
    with open(path, "rb") as f:
        f.seek(int(strip_offset))
        data = np.fromfile(f, dtype=np.int16, count=width * height * 2)

    expected = width * height * 2
    if data.size != expected:
        raise RuntimeError(
            f"Unexpected SLC data size in {path}: got {data.size}, expected {expected}"
        )

    slc = np.reshape(data, (width * 2, height), order="F")
    return slc[0::2, :], slc[1::2, :]


# =========================
# Polynomial geometric transforms
# =========================

class PolyGeoTForm2D:
    """2D polynomial transform between two coordinate spaces."""

    def __init__(self, src_xy, dst_uv, degree=4):
        self.degree = degree
        src_xy = np.asarray(src_xy, dtype=float)
        dst_uv = np.asarray(dst_uv, dtype=float)

        good = np.all(np.isfinite(src_xy), axis=1) & np.all(np.isfinite(dst_uv), axis=1)
        src_xy = src_xy[good]
        dst_uv = dst_uv[good]

        if src_xy.shape[0] < 6:
            raise ValueError("Not enough valid points to fit polynomial transform")

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
        return (arr - amin) / (amax - amin + 1e-12)

    @staticmethod
    def _denorm(arrn, amin, amax):
        return arrn * (amax - amin + 1e-12) + amin

    @staticmethod
    def _design_matrix(x, y, degree):
        terms = []
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                terms.append((i, j))
        M = np.zeros((x.size, len(terms)), dtype=float)
        for k, (i, j) in enumerate(terms):
            M[:, k] = (x ** i) * (y ** j)
        return M

    def _eval(self, coef, x, y):
        out = np.zeros_like(x, dtype=float)
        k = 0
        for i in range(self.degree + 1):
            for j in range(self.degree + 1 - i):
                out += coef[k] * (x ** i) * (y ** j)
                k += 1
        return out

    def transform(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        xn = self._normalize(x, self.xmin, self.xmax)
        yn = self._normalize(y, self.ymin, self.ymax)
        u = self._denorm(self._eval(self.coef_u, xn, yn), self.umin, self.umax)
        v = self._denorm(self._eval(self.coef_v, xn, yn), self.vmin, self.vmax)
        return u, v

    def transform_inverse(self, u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        un = self._normalize(u, self.umin, self.umax)
        vn = self._normalize(v, self.vmin, self.vmax)
        x = self._denorm(self._eval(self.coef_x, un, vn), self.xmin, self.xmax)
        y = self._denorm(self._eval(self.coef_y, un, vn), self.ymin, self.ymax)
        return x, y


def fitgeotform2d(src_xy, dst_uv, model="polynomial", degree=4):
    """Factory function matching MATLAB's fitgeotform2d for polynomial transforms."""
    if model != "polynomial":
        raise ValueError("Only polynomial model is supported")
    return PolyGeoTForm2D(src_xy, dst_uv, degree=degree)


# =========================
# Vector geometry utilities
# =========================

def _load_shp_polygons(shp_path, prj_path):
    """
    Load polygons from a shapefile.

    MATLAB-parity logic:
    - If .prj defines a GEOGRAPHIC CRS: coordinates are already lon/lat.
    - If .prj defines a PROJECTED CRS: coordinates are transformed to EPSG:4326.
    - If .prj is missing or invalid: no transformation is applied.
    """
    if not os.path.exists(shp_path):
        return []

    crs = None
    crs_valid = False
    is_geographic = False

    if os.path.exists(prj_path):
        try:
            with open(prj_path) as f:
                crs = CRS.from_user_input(f.read())
                crs_valid = True
                is_geographic = crs.is_geographic
        except Exception:
            crs = None
            crs_valid = False
            is_geographic = False

    transformer = None
    if crs_valid and not is_geographic:
        transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)

    sf = shapefile.Reader(shp_path)
    geoms = []

    for shape in sf.shapes():
        parts = list(shape.parts) + [len(shape.points)]
        rings = []

        for i in range(len(parts) - 1):
            pts = shape.points[parts[i]:parts[i + 1]]
            if not pts:
                continue

            x, y = zip(*pts)
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)

            if crs_valid and not is_geographic:
                lon, lat = transformer.transform(x, y)
                lon = np.asarray(lon)
                lat = np.asarray(lat)
            else:
                lon = x
                lat = y

            rings.append((lon, lat))

        if len(rings) == 1:
            geoms.append({
                "Lon": rings[0][0],
                "Lat": rings[0][1],
                "Original": list(zip(rings[0][0], rings[0][1])),
            })
        elif len(rings) > 1:
            geoms.append({
                "Multi": [
                    {
                        "Lon": r[0],
                        "Lat": r[1],
                        "Original": list(zip(r[0], r[1])),
                    }
                    for r in rings
                ]
            })

    return geoms


def _poly_from_lonlat(lon, lat):
    """Build a valid Shapely Polygon from lon/lat vectors."""
    coords = [
        (float(lon[i]), float(lat[i]))
        for i in range(len(lon))
        if not (np.isnan(lon[i]) or np.isnan(lat[i]))
    ]

    if len(coords) < 3:
        return None

    try:
        p = Polygon(coords)
        if p.is_empty:
            return None
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty or p.area <= 0:
            return None
        return p
    except Exception:
        return None


def close_event_by_patch(event_geom, patch_poly):
    """
    Close an open FloodEvent geometry using the patch borders.

    This helper is kept for compatibility, but the main pipeline preserves the
    original MATLAB-like behavior and does not clip event polygons by default.
    """
    inter = event_geom.intersection(patch_poly)

    if inter.is_empty:
        return None

    if isinstance(inter, Polygon):
        if inter.area > 0:
            return inter
        return None

    if isinstance(inter, (LineString, MultiLineString)):
        lines = [inter] if isinstance(inter, LineString) else list(inter.geoms)
        patch_boundary = patch_poly.boundary
        merged = unary_union(lines + [patch_boundary])
        try:
            poly = Polygon(merged.convex_hull)
            if poly.is_valid and poly.area > 0:
                return poly
        except Exception:
            pass

    return None


# =========================
# Output utilities
# =========================

def save_complex_patch_tiff(out_path, complex_patch, description=None):
    """Save a complex SAR patch as a single-band complex64 TIFF."""
    arr = np.asarray(complex_patch, dtype=np.complex64)
    tifffile.imwrite(
        out_path,
        arr,
        dtype=np.complex64,
        metadata=None,
        description=description or "",
    )


def load_land_polygons(ne_land_shp):
    """Load and merge Natural Earth land polygons into a single geometry."""
    if not os.path.isfile(ne_land_shp):
        raise FileNotFoundError(f"Natural Earth land file not found: {ne_land_shp}")
    land = gpd.read_file(ne_land_shp)
    return unary_union(land.geometry)


def join_safe(root1, root2, rel_or_abs):
    """Join root1/root2/rel_or_abs unless rel_or_abs is absolute."""
    if not isinstance(rel_or_abs, (str, os.PathLike)):
        raise TypeError(
            f"join_safe: expected path str, got {type(rel_or_abs).__name__} "
            f"(value={rel_or_abs})"
        )
    return str(rel_or_abs) if os.path.isabs(rel_or_abs) else os.path.join(root1, root2, rel_or_abs)


def _get_text(node):
    if isinstance(node, dict):
        return node.get("Text", "")
    return str(node)


# =========================
# MAIN
# =========================

def main(
    inputpath: str,
    SLC_product: str,
    GRD_product: str,
    extrap: str,
    subextrap,
    outputpath: str,
    SIZEPATCH: int,
    USE_AOI_EXACT_POLYGON: bool = False,
):
    """
    Main Flood Detection tiling routine.

    Args:
        inputpath (str): Root input folder.
        SLC_product (str): SLC SAFE folder name.
        GRD_product (str): GRD SAFE folder name.
        extrap (str): Scene folder identifier under inputpath.
        subextrap (str | list[str]): AOI id or list of AOI ids, usually ['01', '02'].
        outputpath (str):
        Output folder. Contains label/, slc/, grd/.
    SIZEPATCH : int
        Patch size in pixels.
    USE_AOI_EXACT_POLYGON : bool
        If False, AOI polygons are converted to bounding boxes, matching your previous behavior.
        If True, exact AOI polygons are used.
    """

    # ==========================================================
    # 1. Normalize AOI list
    # ==========================================================
    if isinstance(subextrap, str):
        aoi_list = [subextrap]
    elif isinstance(subextrap, (list, tuple)):
        aoi_list = [str(a) for a in subextrap]
    else:
        raise TypeError("subextrap must be str or list of str")

    if not aoi_list:
        raise RuntimeError("Empty AOI list")

    scenario = 1

    # ==========================================================
    # 2. Paths
    # ==========================================================
    path_data = os.path.join(inputpath, extrap)

    grdproduct = os.path.join(_href_to_path("GRD/"), GRD_product)
    slcproduct = os.path.join(_href_to_path("SLC/"), SLC_product)

    path_mask_root = os.path.join(path_data, f"{extrap}_mask")
    if not os.path.isdir(path_mask_root):
        raise RuntimeError(f"Mask folder not found: {path_mask_root}")

    path_label = os.path.join(outputpath, "label")
    path_slc = os.path.join(outputpath, "slc")
    path_grd = os.path.join(outputpath, "grd")

    for p in (path_label, path_slc, path_grd):
        os.makedirs(p, exist_ok=True)

    sizepatch = int(SIZEPATCH)
    npatch_first = 1

    # ==========================================================
    # 3. Load AOIs
    # ==========================================================
    aoi_polys = []
    aoi_ids = []

    for aoi_id in aoi_list:
        aoi_path = os.path.join(path_mask_root, aoi_id, "aoi")
        shp = os.path.join(aoi_path, "aoi.shp")
        prj = os.path.join(aoi_path, "aoi.prj")

        if not os.path.exists(shp):
            print(f"[WARN] AOI {aoi_id}: missing aoi.shp")
            continue

        aoi_parts = _load_shp_polygons(shp, prj)
        polys = []

        for part in aoi_parts:
            if "Lon" in part:
                p = _poly_from_lonlat(part["Lon"], part["Lat"])
                if p:
                    if not USE_AOI_EXACT_POLYGON:
                        minx, miny, maxx, maxy = p.bounds
                        p = box(minx, miny, maxx, maxy)
                    polys.append(p)
            else:
                for sub in part["Multi"]:
                    p = _poly_from_lonlat(sub["Lon"], sub["Lat"])
                    if p:
                        if not USE_AOI_EXACT_POLYGON:
                            minx, miny, maxx, maxy = p.bounds
                            p = box(minx, miny, maxx, maxy)
                        polys.append(p)

        if not polys:
            print(f"[WARN] AOI {aoi_id}: empty polygon")
            continue

        aoi_polys.append(unary_union(polys))
        aoi_ids.append(aoi_id)

    if not aoi_polys:
        raise RuntimeError("No valid AOIs loaded")

    # ==========================================================
    # 4. Load Flood Events
    # ==========================================================
    EPS_AREA = 1e-10
    events = []

    for aoi_id in aoi_list:
        event_path = os.path.join(path_mask_root, aoi_id, "event")
        shp = os.path.join(event_path, "event.shp")
        prj = os.path.join(event_path, "event.prj")

        if not os.path.exists(shp):
            continue

        ev_parts = _load_shp_polygons(shp, prj)

        for ev in ev_parts:
            if "Lon" in ev:
                p = _poly_from_lonlat(ev["Lon"], ev["Lat"])
                if p:
                    events.append({"AOI": aoi_id, "poly": p, "original": ev["Original"]})
            else:
                for sub in ev["Multi"]:
                    p = _poly_from_lonlat(sub["Lon"], sub["Lat"])
                    if p:
                        events.append({"AOI": aoi_id, "poly": p, "original": sub["Original"]})

    print(f"[INFO] FloodEvents loaded: {len(events)}")

    # ==========================================================
    # 5. Load Water Bodies
    # ==========================================================
    waters = []

    for aoi_id in aoi_list:
        hydro_path = os.path.join(path_mask_root, aoi_id, "hydro")
        shp = os.path.join(hydro_path, "hydroA.shp")
        prj = os.path.join(hydro_path, "hydroA.prj")

        if not os.path.exists(shp):
            continue

        wb_parts = _load_shp_polygons(shp, prj)

        for wb in wb_parts:
            if "Lon" in wb:
                p = _poly_from_lonlat(wb["Lon"], wb["Lat"])
                if p is not None:
                    waters.append({"AOI": aoi_id, "poly": p, "original": wb["Original"]})
            else:
                for sub in wb["Multi"]:
                    p = _poly_from_lonlat(sub["Lon"], sub["Lat"])
                    if p is not None:
                        waters.append({"AOI": aoi_id, "poly": p, "original": sub["Original"]})

    print(f"[INFO] WaterBodies loaded: {len(waters)}")

    # ==========================================================
    # 6. Load land / sea polygons
    # ==========================================================
    NE_LAND_SHP = os.path.join(inputpath, "shared_data", "natural_earth", "ne_110m_land.shp")
    if not os.path.isfile(NE_LAND_SHP):
        raise RuntimeError(f"Natural Earth land file not found: {NE_LAND_SHP}")

    landpoly = load_land_polygons(NE_LAND_SHP)
    world_poly = box(-180.0, -90.0, 180.0, 90.0)
    seapoly = world_poly.difference(landpoly)

    # ==========================================================
    # 7. Sentinel-1 products / manifests
    # ==========================================================
    safe_name = slcproduct.replace("\\", "/")
    m = re.search(r"_([A-Za-z0-9]{4})\.SAFE$", safe_name)
    if m is not None:
        slc_id = m.group(1)
    else:
        slc_id = os.path.basename(safe_name).replace(".SAFE", "")

    # Empty legend: scenario, patch_id, has_waterbody, has_floodevent, swath_id
    legends = {f"legend{slc_id}": np.empty((0, 5), dtype=int)}

    metadata_SLC = xml2struct(os.path.join(path_data, slcproduct, "manifest.safe"))
    #metadata_GRD = xml2struct(os.path.join(path_data, grdproduct, "manifest.