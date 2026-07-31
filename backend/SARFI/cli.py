from logging import Logger
import argparse
import logging
import os
import sys
import datetime
import tempfile
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from dateutil.parser import isoparse as _isoparse
from hashlib import md5
from rasterio import open as rio_open
from scipy.interpolate import CubicHermiteSpline, LinearNDInterpolator
from pathlib import Path
from xmltodict import parse as xml_parse
from typing import Dict, Optional, Union
from urllib.request import urlretrieve
import yaml
from time import perf_counter
from SARFI.utils import SARFILogger

# Add backend root to sys.path so pipeline.geocoding_block is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.geocoding_block.utils import range_doppler  # noqa: E402
from dem_stitcher.stitcher import stitch_dem  # noqa: E402
from pyproj import Transformer  # noqa: E402
from pyproj.sync import get_proj_endpoint  # noqa: E402
from pyproj.datadir import get_user_data_dir  # noqa: E402

class SLCMatcher:
    def __init__(self, safe_paths_dir:Union[str, Path], csv_fp:Union[str, Path], logger:Union[None, Logger]=None, dem_cache_dir:Union[str, Path, None]=None):
        """
        Initialize the SLCMatcher with the directory containing SAFE files, the input CSV file path, and an optional logger.
        Args:
            safe_paths_dir (Union[str, Path]): The directory path where the SAFE files are located
            csv_fp (Union[str, Path]): The file path to the input CSV metadata file containing target locations and time windows.
            logger (Union[None, Logger]): An optional logger instance for logging messages. If None, a default SARFILogger will be used.
            dem_cache_dir (Union[str, Path, None]): Directory for caching downloaded DEM tiles. If None, a temporary directory is created.
        """
        self.safe_paths_dir = safe_paths_dir
        self.csv_fp = csv_fp
        self.logger = logger if logger else SARFILogger().logger
        if dem_cache_dir is None:
            self.dem_cache_dir = Path(tempfile.mkdtemp(prefix="sarfi_dem_"))
        else:
            self.dem_cache_dir = Path(dem_cache_dir)
        self.dem_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_annotation_files(self) -> list[Path]:
        """
        Retrieve all annotation files from the SAFE directories.

        Returns:
            list[Path]: A list of paths to annotation XML files.
        """
        all_annotation_files = list()
        safe_dir_paths = [Path(self.safe_paths_dir, dir_name) for dir_name in os.listdir(self.safe_paths_dir) if Path(self.safe_paths_dir, dir_name).is_dir()]
        for safe_dir in safe_dir_paths:
            if safe_dir.is_dir():
                annotation_files = list(safe_dir.glob(f"**/annotation/*.xml"))
                all_annotation_files.extend(annotation_files)
        return all_annotation_files

    def _build_geoloc_arrays(self, geolocs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build numpy arrays from a geolocation grid point list for efficient processing.

        Args:
            geolocs (list[Dict]): A list of geolocation points, each represented as a dictionary
                                  with keys 'latitude', 'longitude', 'line', and 'pixel'.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Four numpy arrays containing
            latitudes, longitudes, line indices, and pixel indices, respectively.
        """
        n = len(geolocs)
        lats = np.empty(n)
        lons = np.empty(n)
        lines = np.empty(n, dtype=np.int64)
        pixels = np.empty(n, dtype=np.int64)
        for i, g in enumerate(geolocs):
            lats[i] = float(g['latitude'])
            lons[i] = float(g['longitude'])
            lines[i] = int(g['line'])
            pixels[i] = int(g['pixel'])
        return lats, lons, lines, pixels

    def _build_rough_line_interpolator(self, lats: np.ndarray, lons: np.ndarray, lines: np.ndarray) -> object:
        """
        Build a LinearNDInterpolator from (lat, lon) -> line using the geolocation
        grid.  Used only for rough burst-index determination before RDTC.

        Args:
            lats (np.ndarray): 1-D array of latitudes from the geolocation grid.
            lons (np.ndarray): 1-D array of longitudes from the geolocation grid.
            lines (np.ndarray): 1-D array of absolute line indices.

        Returns:
            LinearNDInterpolator mapping (lat, lon) -> line.
        """
        points = np.column_stack([lats, lons])
        return LinearNDInterpolator(points, lines.astype(float))

    def _extract_state_vectors_from_annotation(self, metadata: Dict) -> Dict:
        """
        Extract satellite state vectors from the annotation XML.
        Sentinel-1 annotation files store predicted/restituted state vectors in
        product/generalAnnotation/orbitList/orbit.

        Args:
            metadata (Dict): Parsed annotation XML (via xmltodict).

        Returns:
            Dict with keys 't0' (reference datetime), 't' (seconds relative to t0),
            'x', 'y', 'z' (ECEF position in metres), 'vx', 'vy', 'vz' (ECEF velocity).
        """
        orbit_list = metadata["product"]["generalAnnotation"]["orbitList"]["orbit"]
        # xmltodict returns a dict instead of a list when there is only one element
        if isinstance(orbit_list, dict):
            orbit_list = [orbit_list]
        t0 = _isoparse(orbit_list[0]["time"])
        times, xs, ys, zs, vxs, vys, vzs = [], [], [], [], [], [], []
        for sv in orbit_list:
            times.append((_isoparse(sv["time"]) - t0).total_seconds())
            xs.append(float(sv["position"]["x"]))
            ys.append(float(sv["position"]["y"]))
            zs.append(float(sv["position"]["z"]))
            vxs.append(float(sv["velocity"]["x"]))
            vys.append(float(sv["velocity"]["y"]))
            vzs.append(float(sv["velocity"]["z"]))
        return {
            "t0": t0,
            "t": np.array(times),
            "x": np.array(xs), "y": np.array(ys), "z": np.array(zs),
            "vx": np.array(vxs), "vy": np.array(vys), "vz": np.array(vzs),
        }

    def _sv_interpolator(self, state_vectors: Dict) -> tuple:
        """
        Build cubic Hermite spline interpolators for satellite position and velocity.

        Args:
            state_vectors (Dict): State vector dict with 't', 'x', 'y', 'z', 'vx', 'vy', 'vz'.

        Returns:
            tuple: (interp_pos, interp_vel) CubicHermiteSpline instances.
        """
        t = state_vectors["t"]
        pos = np.array([state_vectors["x"], state_vectors["y"], state_vectors["z"]]).T
        vel = np.array([state_vectors["vx"], state_vectors["vy"], state_vectors["vz"]]).T
        interp_pos = CubicHermiteSpline(t, pos, vel)
        interp_vel = interp_pos.derivative(1)
        return interp_pos, interp_vel

    def _fetch_dem(self, bounds: tuple) -> Path:
        """
        Download (or retrieve from cache) a cop-dem-glo-30 DEM for the given bounds.

        Args:
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) bounding box.

        Returns:
            Path: Local path to the downloaded DEM GeoTIFF.
        """
        hash_str = md5(str(bounds).encode()).hexdigest()
        dem_fp = self.dem_cache_dir / f"dem-{hash_str}.tiff"
        if dem_fp.exists():
            self.logger.info(f"Using cached DEM: {dem_fp}")
            return dem_fp
        self.logger.info(f"Downloading DEM for bounds {bounds}")
        dems_to_try = ["glo_30", "srtm_v3", "nasadem"]
        wrote_file = False
        for dem_name in dems_to_try:
            try:
                patch, patch_header = stitch_dem(
                    bounds=bounds,
                    dem_name=dem_name,
                    dst_area_or_point="Area",
                    dst_ellipsoidal_height=False,
                    fill_in_glo_30=True,
                    dst_tile_dir=None,
                    overwrite_existing_tiles=False,
                )
                with rio_open(dem_fp, "w", **patch_header) as ds:
                    ds.write(patch, 1)
                    ds.update_tags(COMPOSITE_CRS="EPSG:4326+3855")
                # Rewrite with float64 nodata=NaN for consistency
                with rio_open(dem_fp, "r") as ds:
                    data = ds.read().astype("float64")
                    profile = ds.profile.copy()
                    if profile.get("nodata") is not None:
                        data[data == profile["nodata"]] = np.nan
                    profile.update({"nodata": np.nan, "dtype": "float64"})
                with rio_open(dem_fp, "w", **profile) as ds:
                    ds.write(data)
                    ds.update_tags(COMPOSITE_CRS="EPSG:4326+3855")
                self.logger.info(f"DEM saved to {dem_fp}")
                wrote_file = True
                break
            except Exception as err:
                self.logger.warning(f"DEM download failed with source '{dem_name}': {err}")
        if not wrote_file:
            raise FileNotFoundError(f"Could not download DEM for bounds {bounds}")
        return dem_fp

    def _load_dem_coords(self, dem_fp: Path) -> tuple:
        """
        Load latitude, longitude, and altitude arrays from a DEM GeoTIFF.

        Returns:
            tuple: (lat_2d, lon_2d, alt, composite_crs) where lat_2d and lon_2d
                   have the same shape as the DEM raster and contain actual
                   latitude / longitude values respectively.
        """
        composite_crs = "EPSG:4326+3855"  # cop-dem-glo-30 uses EGM2008 heights
        with rio_open(dem_fp) as ds:
            alt = ds.read(1).astype("float64")
            dem_trans = ds.transform
            nodata = ds.nodata
        height, width = alt.shape
        # Fast-path for rectilinear (non-rotated) rasters
        ix = np.arange(width)
        iy = np.arange(height)
        lon_col = dem_trans[0] * ix + dem_trans[2]  # longitude per column
        lat_row = dem_trans[4] * iy + dem_trans[5]  # latitude per row (decreasing)
        lon_2d = lon_col[None, :] + np.zeros_like(alt)  # (height, width)
        lat_2d = lat_row[:, None] + np.zeros_like(alt)  # (height, width)
        if nodata is not None and not np.isnan(float(nodata)):
            alt[alt == float(nodata)] = np.nan
        return lat_2d, lon_2d, alt, composite_crs

    def _lla_to_ecef(self, lat_2d: np.ndarray, lon_2d: np.ndarray, alt: np.ndarray,
                     composite_crs: str = "EPSG:4326+3855") -> tuple:
        """
        Convert latitude / longitude / altitude to ECEF (x, y, z) in metres.
        Follows the same convention as Geocoder.lla_to_ecef: the Transformer is
        called with (latitude, longitude, height) since EPSG:4326 is lat-first.

        Args:
            lat_2d: 2-D latitude array (degrees).
            lon_2d: 2-D longitude array (degrees).
            alt:    2-D altitude array (metres above geoid).
            composite_crs: Compound CRS string for the input data.

        Returns:
            tuple: (dem_x, dem_y, dem_z) ECEF arrays with the same shape as inputs.
        """
        ecef_crs = "EPSG:4978"
        grid_name = "us_nga_egm08_25.tif" if composite_crs == "EPSG:4326+3855" else "us_nga_egm96_15.tif"
        proj_path = Path(get_user_data_dir())
        proj_path.mkdir(parents=True, exist_ok=True)
        grid_path = proj_path / grid_name
        if not grid_path.exists():
            grid_url = f"{get_proj_endpoint()}/{grid_name}"
            self.logger.info(f"Downloading geoid grid: {grid_url}")
            urlretrieve(grid_url, grid_path)

        def _transform_chunk(chunk):
            tf = Transformer.from_crs(composite_crs, ecef_crs)
            # EPSG:4326 is lat-first; pass (lat, lon, height)
            return tf.transform(chunk[0], chunk[1], chunk[2])

        chunk_size = 256
        rows = lat_2d.shape[0]
        wgs_pts = [
            (lat_2d[b:b + chunk_size], lon_2d[b:b + chunk_size], alt[b:b + chunk_size])
            for b in range(0, rows, chunk_size)
        ]
        with ThreadPoolExecutor() as executor:
            chunked = list(executor.map(_transform_chunk, wgs_pts))
        dem_x = np.vstack([c[0] for c in chunked])
        dem_y = np.vstack([c[1] for c in chunked])
        dem_z = np.vstack([c[2] for c in chunked])
        return dem_x, dem_y, dem_z

    def _run_rdtc_for_burst(
        self,
        metadata: Dict,
        state_vectors: Dict,
        dem_x: np.ndarray,
        dem_y: np.ndarray,
        dem_z: np.ndarray,
        dem_shape: tuple,
        burst_idx: int,
        lines_per_burst: int,
        samples_per_burst: int,
    ) -> tuple:
        """
        Run Range-Doppler Terrain Correction for a single burst and return the
        azimuth-line / range-sample lookup tables in DEM geometry.

        The DEM ECEF coordinates (dem_x, dem_y, dem_z) are pre-computed once
        for the full scene and shared across all bursts to avoid redundant I/O
        and coordinate transformations.

        Args:
            metadata: Parsed annotation XML.
            state_vectors: State vector dict produced by _extract_state_vectors_from_annotation.
            dem_x, dem_y, dem_z: ECEF coordinate arrays with shape dem_shape.
            dem_shape: (height, width) shape of the DEM grid.
            burst_idx: 1-based burst index.
            lines_per_burst: Lines per burst from the annotation.
            samples_per_burst: Samples per burst (range pixels).

        Returns:
            tuple: (az_geo_abs, rg_geo) both shaped dem_shape.
                   az_geo_abs: absolute line index in the full subswath SLC.
                   rg_geo: range pixel index (0-indexed).
                   Invalid pixels are set to NaN.
        """
        image_info = metadata["product"]["imageAnnotation"]["imageInformation"]
        dt_az = float(image_info["azimuthTimeInterval"])
        slant_range_time = float(image_info["slantRangeTime"])
        product_info = metadata["product"]["generalAnnotation"]["productInformation"]
        range_sampling_rate = float(product_info["rangeSamplingRate"])

        burst = metadata["product"]["swathTiming"]["burstList"]["burst"][burst_idx - 1]
        az_time = burst["azimuthTime"]

        t0 = state_vectors["t0"]
        t0_az = (_isoparse(az_time) - t0).total_seconds()
        naz = lines_per_burst
        nrg = samples_per_burst
        t_end_burst = t0_az + dt_az * naz

        # Crop state vectors to a window around the burst with generous padding
        t_sv = state_vectors["t"]
        t_pad = 360.0  # seconds
        cnd = (t_sv > t0_az - t_pad) & (t_sv < t_end_burst + t_pad)
        sv_cropped = {k: v[cnd] for k, v in state_vectors.items() if k != "t0"}
        interp_pos, interp_vel = self._sv_interpolator(sv_cropped)

        # One satellite position / velocity per azimuth line
        t_arr = np.linspace(t0_az, t0_az + dt_az * (naz - 1), naz)
        pos = interp_pos(t_arr)  # (naz, 3)
        vel = interp_vel(t_arr)  # (naz, 3)

        self.logger.debug(f"Running RDTC for burst {burst_idx}")
        # Shift DEM coordinates by the first satellite position for numerical precision
        az_geo_flat, dist_geo, _, _, _ = range_doppler(
            dem_x.ravel() - pos[0, 0],
            dem_y.ravel() - pos[0, 1],
            dem_z.ravel() - pos[0, 2],
            pos - pos[0],
            vel,
        )

        # Convert slant-range distance to pixel index
        c0 = 299_792_458.0
        r0 = slant_range_time * c0 / 2.0
        dr = c0 / (2.0 * range_sampling_rate)
        rg_geo_flat = (dist_geo - r0) / dr

        # Mask pixels that fall outside this burst's valid range
        valid = (
            (rg_geo_flat >= 0) & (rg_geo_flat < nrg)
            & (az_geo_flat >= 0) & (az_geo_flat < naz)
        )
        rg_geo_flat[~valid] = np.nan
        az_geo_flat[~valid] = np.nan

        rg_geo = rg_geo_flat.reshape(dem_shape)
        az_geo = az_geo_flat.reshape(dem_shape)
        # Convert in-burst line index to absolute line in the full subswath SLC
        az_geo_abs = az_geo + (burst_idx - 1) * lines_per_burst
        return az_geo_abs, rg_geo

    def check_annotation_files(self, max_matches: Optional[int] = 3) -> list[Dict]:
        """
        Check each annotation file against the input CSV metadata to find matches
        based on temporal criteria, then determine accurate SLC line / sample
        coordinates using Range-Doppler Terrain Correction (RDTC) with a
        cop-dem-glo-30 DEM.

        For each annotation file that passes the temporal filter:
          1. The DEM is downloaded once for the full scene footprint (cached).
          2. The DEM pixels are converted to ECEF coordinates once.
          3. RDTC is run lazily per burst: a burst's lookup table is only computed
             when a target point is estimated (via the sparse geolocation grid) to
             fall within that burst.  Results are cached so subsequent targets in
             the same burst reuse the already-computed LUT and interpolators.
          4. A LinearNDInterpolator built from the RDTC LUT maps (lat, lon) to
             the accurate (absolute_line, sample) pair.

        Args:
            max_matches (Optional[int]): Maximum matches per unique (lat, lon)
                coordinate. Pass None for no limit. Defaults to 3.

        Returns:
            list[Dict]: Matching entries with filename, line, sample, and metadata.
        """
        matching_list: list = []
        annotation_files = self.get_annotation_files()
        df = pd.read_csv(self.csv_fp)
        match_counts: Dict[tuple, int] = {}

        for annotation_file in sorted(annotation_files):
            with annotation_file.open() as f:
                metadata = xml_parse(f.read())

            # ── Temporal filter ──────────────────────────────────────────────
            mdst = datetime.datetime.strptime(
                metadata["product"]["adsHeader"]["startTime"], "%Y-%m-%dT%H:%M:%S.%f"
            )
            mdet = datetime.datetime.strptime(
                metadata["product"]["adsHeader"]["stopTime"], "%Y-%m-%dT%H:%M:%S.%f"
            )
            df2 = df[
                (pd.to_datetime(df["start_time"]) <= pd.to_datetime(mdst))
                & (pd.to_datetime(df["stop_time"]) >= pd.to_datetime(mdet))
            ]
            if df2.empty:
                continue

            # ── Scene geometry metadata ───────────────────────────────────────
            burst_timing = metadata["product"]["swathTiming"]
            burst_count = int(burst_timing["burstList"]["@count"])
            lines_per_burst = int(burst_timing["linesPerBurst"])
            samples_per_burst = int(burst_timing["samplesPerBurst"])

            # Geolocation grid – used for (a) scene bounds, (b) rough burst
            # determination, and (c) min_distance_deg reporting.
            geolocs = (
                metadata["product"]["geolocationGrid"]
                ["geolocationGridPointList"]["geolocationGridPoint"]
            )
            lats, lons, lines, pixels = self._build_geoloc_arrays(geolocs)

            # ── DEM download (cached per scene bounds) ────────────────────────
            bounds = (
                float(np.min(lons)), float(np.min(lats)),
                float(np.max(lons)), float(np.max(lats)),
            )
            try:
                dem_fp = self._fetch_dem(bounds)
            except Exception as err:
                self.logger.error(
                    f"Skipping {annotation_file.name}: DEM download failed – {err}"
                )
                continue

            # ── Convert full DEM to ECEF once for all bursts ──────────────────
            try:
                lat_2d, lon_2d, alt, composite_crs = self._load_dem_coords(dem_fp)
                dem_x, dem_y, dem_z = self._lla_to_ecef(lat_2d, lon_2d, alt, composite_crs)
            except Exception as err:
                self.logger.error(
                    f"Skipping {annotation_file.name}: ECEF conversion failed – {err}"
                )
                continue

            # ── Extract orbit state vectors from annotation XML ───────────────
            try:
                state_vectors = self._extract_state_vectors_from_annotation(metadata)
            except Exception as err:
                self.logger.error(
                    f"Skipping {annotation_file.name}: state vector extraction failed – {err}"
                )
                continue

            self.logger.info(
                f"Processing {annotation_file.name} ({burst_count} bursts, "
                f"{len(df2)} candidate target(s))"
            )

            # Rough (lat, lon) → line interpolator for burst-index estimation
            rough_line_interp = self._build_rough_line_interpolator(lats, lons, lines)

            # Cache: burst_idx -> (line_interp_rdtc, pixel_interp_rdtc)
            # Values of (None, None) mark a burst where RDTC produced no valid points.
            burst_interp_cache: Dict[int, tuple] = {}

            for _, row in df2.iterrows():
                target_lat = float(row["latitude"])
                target_lon = float(row["longitude"])
                key = (target_lat, target_lon)
                if max_matches is not None and match_counts.get(key, 0) >= max_matches:
                    continue

                pt = np.array([[target_lat, target_lon]])

                # ── Rough burst determination ─────────────────────────────────
                rough_line = rough_line_interp(pt)[0]
                if np.isnan(rough_line):
                    continue
                burst_idx_est = max(1, min(burst_count, int(rough_line / lines_per_burst) + 1))

                # Try the estimated burst, then its immediate neighbours, to
                # handle points near burst boundaries.
                found = False
                for b in [burst_idx_est, burst_idx_est - 1, burst_idx_est + 1]:
                    if b < 1 or b > burst_count:
                        continue

                    # ── RDTC (lazy, cached per burst) ─────────────────────────
                    if b not in burst_interp_cache:
                        try:
                            az_abs, rg = self._run_rdtc_for_burst(
                                metadata, state_vectors,
                                dem_x, dem_y, dem_z, alt.shape,
                                b, lines_per_burst, samples_per_burst,
                            )
                            valid = ~(np.isnan(az_abs) | np.isnan(rg))
                            if valid.sum() < 3:
                                burst_interp_cache[b] = (None, None)
                            else:
                                rdtc_pts = np.column_stack([
                                    lat_2d.ravel()[valid.ravel()],
                                    lon_2d.ravel()[valid.ravel()],
                                ])
                                burst_interp_cache[b] = (
                                    LinearNDInterpolator(
                                        rdtc_pts, az_abs.ravel()[valid.ravel()]
                                    ),
                                    LinearNDInterpolator(
                                        rdtc_pts, rg.ravel()[valid.ravel()]
                                    ),
                                )
                        except Exception as err:
                            self.logger.warning(
                                f"RDTC failed for burst {b} of "
                                f"{annotation_file.name}: {err}"
                            )
                            burst_interp_cache[b] = (None, None)

                    line_interp_rdtc, pixel_interp_rdtc = burst_interp_cache[b]
                    if line_interp_rdtc is None:
                        continue

                    interp_line = line_interp_rdtc(pt)[0]
                    interp_pixel = pixel_interp_rdtc(pt)[0]
                    if np.isnan(interp_line) or np.isnan(interp_pixel):
                        continue

                    dist_deg = np.sqrt(
                        (lats - target_lat) ** 2 + (lons - target_lon) ** 2
                    )
                    matching_list.append({
                        "filename": str(annotation_file),
                        "line": int(round(float(interp_line))),
                        "sample": int(round(float(interp_pixel))),
                        "target_latitude": target_lat,
                        "target_longitude": target_lon,
                        "target_start_time": row["start_time"],
                        "target_stop_time": row["stop_time"],
                        "min_distance_deg": float(np.min(dist_deg)),
                    })
                    match_counts[key] = match_counts.get(key, 0) + 1
                    found = True
                    break  # stop trying adjacent bursts once a valid result is found

        return matching_list
    
    def write_matching_results_to_csv(self, matching_list:list, output_csv_fp:Union[str, Path]):
        matching_df = pd.DataFrame(matching_list)
        matching_df = matching_df.drop(columns=['min_distance_deg'], errors='ignore')
        matching_df.to_csv(output_csv_fp, index=False)
        self.logger.info(f"Matching results saved to {output_csv_fp}")

def apply_filter(matching_list: list, filter_mode: str) -> list:
    """
    Filter the matching list based on the specified filter mode.

    Args:
        matching_list (list): The list of match dictionaries produced by check_annotation_files.
        filter_mode (str): 'all' returns all matches as-is; 'best' retains only the match with
                           the smallest min_distance_deg for each unique (latitude, longitude) coordinate.

    Returns:
        list: The filtered list of match dictionaries.
    """
    if filter_mode == 'all':
        return matching_list
    # filter_mode == 'best'
    # Key on (target_lat, target_lon, filename) so that each annotation file is
    # evaluated independently per target coordinate. This ensures all distinct
    # (target, annotation_file) pairs are retained, with the spatially closest
    # match kept when the same combination appears more than once.
    best_matches: Dict[tuple, Dict] = {}
    for match in matching_list:
        key = (match['target_latitude'], match['target_longitude'], match['filename'])
        if key not in best_matches or match['min_distance_deg'] < best_matches[key]['min_distance_deg']:
            best_matches[key] = match
    return list(best_matches.values())


def parse_yaml_config(yaml_fp:Union[str, Path]) -> Dict:
    """
    Parse the YAML configuration file and validate required keys and file paths.
    Args:
        yaml_fp (Union[str, Path]): The file path to the YAML configuration file.
    Returns:
        cfg (Dict): A dictionary containing the configuration parameters.
    """
    with open(yaml_fp, "r") as f:
        cfg = yaml.safe_load(f)
    required_keys = ['safe_dir_fp', 'input_csv_fp', 'output_csv_fp']
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing required key '{key}' in YAML configuration.")
        if key != "output_csv_fp" and 'fp' in key and not Path(cfg[key]).exists():
            raise ValueError(f"File path specified for '{key}' does not exist: {cfg[key]}")
    # dem_cache_dir is optional; validate only if provided
    if 'dem_cache_dir' in cfg and cfg['dem_cache_dir'] is not None:
        dem_dir = Path(cfg['dem_cache_dir'])
        dem_dir.mkdir(parents=True, exist_ok=True)
    return cfg

def run_slc_matcher(logger: Union[None, SARFILogger], max_matches: int = 3, filter_mode: str = 'all') -> int:
    """
    Run the SLC matching process using the configuration specified in the YAML file.
    Args:
        logger (Union[None, SARFILogger]): An optional logger instance for logging messages. If None, a default SARFILogger will be used.
        max_matches (int): Maximum number of matches per unique (latitude, longitude) coordinate. Defaults to 3.
        filter_mode (str): 'all' to include all matches up to max_matches; 'best' to return only the best match per coordinate. Defaults to 'all'.
    Returns:
        int: The number of matching entries found and written to the output CSV file.
    """
    matcher_logger = logger if logger else SARFILogger().logger

    yaml_cfg_fp = Path(__file__).parent.joinpath("configuration", "slc_matcher_config.yaml")
    if not yaml_cfg_fp.exists():
        yaml_cfg_fp = Path(os.environ.get("SARFI_CONFIG_PATH"))
    cfg = parse_yaml_config(yaml_cfg_fp)
    csv_fp = cfg['input_csv_fp']
    safe_paths_dir = cfg['safe_dir_fp']
    dem_cache_dir = cfg.get('dem_cache_dir', None)
    slc_matcher = SLCMatcher(
        safe_paths_dir=safe_paths_dir,
        csv_fp=csv_fp,
        logger=matcher_logger,
        dem_cache_dir=dem_cache_dir,
    )
    # For 'best' mode, collect all candidate annotation file matches (no cap) so the
    # filter can compare every qualifying file and pick the spatially closest one.
    effective_max = None if filter_mode == 'best' else max_matches
    matching_list = slc_matcher.check_annotation_files(max_matches=effective_max)
    matching_list = apply_filter(matching_list, filter_mode)
    output_csv_fp = cfg['output_csv_fp']
    slc_matcher.write_matching_results_to_csv(matching_list, output_csv_fp)
    return len(matching_list)

def sarfi_main():
    parser = argparse.ArgumentParser(description="SARFI SLC Matcher")
    parser.add_argument(
        '-m', '--max_matches',
        type=int,
        default=3,
        help="Maximum number of matches per location (1-10, default: 3)"
    )
    parser.add_argument(
        '-f', '--filter',
        type=str,
        default='all',
        choices=['all', 'best'],
        help="Filter mode: 'all' includes all matches up to max_matches; 'best' returns only the best match per location (default: all)"
    )
    args = parser.parse_args()
    if not (1 <= args.max_matches <= 10):
        parser.error("--max_matches must be between 1 and 10.")
    logger_obj = SARFILogger()
    logger_obj.update_log_level(logging.DEBUG)
    logger = logger_obj.logger
    start_time = perf_counter()
    num_matches = run_slc_matcher(logger, max_matches=args.max_matches, filter_mode=args.filter)
    end_time = perf_counter()
    elapsed_time = end_time - start_time
    logger.info(f"SLC matching process completed in {elapsed_time:.2f} seconds.")
    logger.info(f"Number of matching entries found: {num_matches}")


def sarfi_main_wrapper(mode='best', max_matches=10):
    logger_obj = SARFILogger()
    logger_obj.update_log_level(logging.DEBUG)
    logger = logger_obj.logger
    logger.debug(f"Running SLC matcher with filter mode '{mode}' and max_matches={max_matches}")
    start_time = perf_counter()
    num_matches = run_slc_matcher(logger, max_matches=max_matches, filter_mode=mode)
    end_time = perf_counter()
    elapsed_time = end_time - start_time
    logger.info(f"SLC matching process completed in {elapsed_time:.2f} seconds.")
    logger.info(f"Number of matching entries found: {num_matches}")
    

if __name__ == "__main__":
    sarfi_main()
