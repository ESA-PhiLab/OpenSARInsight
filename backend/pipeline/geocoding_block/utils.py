from ctypes import Union

import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from shapely.geometry import box
from numba import njit, prange, cfunc
import numpy as np 
from zipfile import ZipFile
import re
from glob import glob 
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from rasterio.windows import Window
from scipy.ndimage import uniform_filter
from scipy.ndimage import variance

@njit(parallel=True)
def _shadow_mask(theta, rg0, az):

    az_min, az_max = int(np.ceil(np.nanmin(az))), int(np.floor(np.nanmax(az)))
    rg0_min, rg0_max = int(np.ceil(np.nanmin(rg0))), int(np.floor(np.nanmax(rg0)))
    naz = az_max - az_min + 1
    nrg0 = rg0_max - rg0_min + 1

    # coarse warping into zero altitude (ground) geometry
    theta0 = np.full((naz, nrg0), fill_value=np.nan)
    for i in prange(theta.shape[0]):
        for j in range(theta.shape[1]):
            if np.isfinite(az[i, j]) and az[i, j] > 0 and rg0[i, j] > 0:
                theta0[int(az[i, j]) - az_min, int(rg0[i, j]) - rg0_min] = theta[i, j]

    # scanning lines in ground geometry
    mask0 = np.full_like(theta0, fill_value=np.nan)
    for i in prange(theta0.shape[0]):
        max_elev = 0.0
        for j in range(theta0.shape[1]):
            if not np.isnan(theta0[i, j]):
                if theta0[i, j] > max_elev:
                    max_elev = theta0[i, j]
                else:
                    mask0[i, j] = 1.0

    # back to DEM geometry
    mask = np.full_like(theta, fill_value=np.nan)
    for i in prange(mask.shape[0]):
        for j in range(mask.shape[1]):
            if not np.isfinite(az[i, j]) and az[i, j] > 0 and rg0[i, j] > 0:
                mask[i, j] = mask0[int(az[i, j] - az_min), int(rg0[i, j] - rg0_min)]

    return mask


@njit(nogil=True, cache=True, parallel=True)
def range_doppler(xx, yy, zz, positions, velocities, tol=1e-8, maxiter=10000):
    def doppler_freq(t, x, y, z, positions, velocities, t0, t1):
        factors = t - np.floor(t)

        px = positions[t0, 0] + factors * (positions[t1, 0] - positions[t0, 0])
        py = positions[t0, 1] + factors * (positions[t1, 1] - positions[t0, 1])
        pz = positions[t0, 2] + factors * (positions[t1, 2] - positions[t0, 2])
        vx = velocities[t0, 0] + factors * (velocities[t1, 0] - velocities[t0, 0])
        vy = velocities[t0, 1] + factors * (velocities[t1, 1] - velocities[t0, 1])
        vz = velocities[t0, 2] + factors * (velocities[t1, 2] - velocities[t0, 2])

        dx = x - px
        dy = y - py
        dz = z - pz
        d2 = dx**2 + dy**2 + dz**2
        fc = -(vx * dx + vy * dy + vz * dz) / np.sqrt(d2)

        return fc, dx, dy, dz

    i_zd = np.zeros_like(xx)
    r_zd = np.zeros_like(xx)
    dx = np.zeros_like(xx)
    dy = np.zeros_like(xx)
    dz = np.zeros_like(xx)
    num_orbits = len(positions)

    for i in prange(xx.shape[0]):
        x_val = xx[i]
        y_val = yy[i]
        z_val = zz[i]
        if np.isnan(x_val):
            continue
        a = 0
        b = num_orbits - 1
        fa, _, _, _ = doppler_freq(
            a, x_val, y_val, z_val, positions, velocities, int(a), int(np.ceil(a))
        )
        fb, _, _, _ = doppler_freq(
            b, x_val, y_val, z_val, positions, velocities, int(b), int(np.ceil(b))
        )

        # exit if no solution
        if np.sign(fa * fb) > 0:
            i_zd[i] = np.nan
            r_zd[i] = np.nan
            continue

        if np.abs(fa) < tol:
            i_zd[i] = a
            r_zd[i] = 0
            continue
        elif np.abs(fb) < tol:
            i_zd[i] = b
            r_zd[i] = 0
            continue

        c = (a + b) / 2.0
        fc, _, _, _ = doppler_freq(
            c, x_val, y_val, z_val, positions, velocities, int(c), int(np.ceil(c))
        )

        its = 0
        while np.abs(fc) > tol and its < maxiter:
            its += 1
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            c = (a + b) / 2.0
            fc, _, _, _ = doppler_freq(
                c, x_val, y_val, z_val, positions, velocities, int(c), int(np.ceil(c))
            )

        i_zd[i] = c
        dx[i], dy[i], dz[i] = doppler_freq(
            c, x_val, y_val, z_val, positions, velocities, int(c), int(np.ceil(c))
        )[1:]
        r_zd[i] = np.sqrt(dx[i] ** 2 + dy[i] ** 2 + dz[i] ** 2)

    return i_zd, r_zd, dx, dy, dz

@njit(nogil=True, parallel=True, cache=True)
def simulate_terrain_backscatter(
    naz, nrg, az, rg, dem_x, dem_y, dem_z, dx, dy, dz, shadow_mask
):
    """Use DEM and look vectors to simulate terrain backscatter in the SAR geometry

    Args:
        naz (int): azimuth size
        nrg (int): slant range size
        az (array): Lookup table of azimuth indices
        rg (array): Lookup table of range indices
        dem_x (array): DEM x coordinates
        dem_y (array): DEM y coordinates
        dem_z (array): DEM z coordinates
        dx (array): Look vector x coordinates
        dy (array): Look vector y coordinates
        dz (array): Look vector z coordinates
        shadow_mask(array): Shadow mask containing ones for shadow areas and NaN elsewhere

    Returns:
        array: simulated terrain gamma nought

    Note:
        This is a modified version of the algorithm described in SNAP terrain correction documentation. Two things are different:
            - Instead of the sine of the projected incidence angle,
            the tangent is computed to comply with the gamma nought convention.
            - The simulated backscatter is regridded and accumulated in the SAR geometry to account for many-to-one and one-to-many relationships.
    """

    # test if point is in triangle
    def is_in_tri(p, a, b, c):

        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        l1 = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / det
        l2 = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / det

        return (l1 >= 0) and (l2 >= 0) and (l1 + l2 < 1)

    def project_point_on_plane(p, u, v):

        uv = np.dot(u, v)
        up = np.dot(u, p)
        vp = np.dot(v, p)

        denom = 1 - uv**2

        alpha = (up - uv * vp) / denom
        beta = (vp - uv * up) / denom

        # Compute the projection
        p_proj = alpha * u + beta * v
        return p_proj

    def norm_vec(v):
        return np.sqrt((v**2).sum())

    gamma_proj = np.zeros((naz, nrg))

    nl, nc = az.shape
    # - loop on DEM
    for i in prange(0, nl - 1):
        for j in range(0, nc - 1):
            if shadow_mask[i, j] == 1:
                continue
            # - for each 4 neighborhood
            aa = az[i : i + 2, j : j + 2].flatten()
            rr = rg[i : i + 2, j : j + 2].flatten()
            xx = dem_x[i : i + 2, j : j + 2].flatten()
            yy = dem_y[i : i + 2, j : j + 2].flatten()
            zz = dem_z[i : i + 2, j : j + 2].flatten()
            # - collect triangle vertices
            aarr = np.vstack((aa, rr)).T
            if np.isnan(aarr).any():
                continue
            # - compute bounding box in the radar grid
            amin, amax = np.floor(aa.min()), np.ceil(aa.max())
            rmin, rmax = np.floor(rr.min()), np.ceil(rr.max())
            amin = int(np.maximum(amin, 0))
            rmin = int(np.maximum(rmin, 0))
            amax = int(np.minimum(amax, naz - 1)) + 1
            rmax = int(np.minimum(rmax, nrg - 1)) + 1

            # Triangle 1
            # look vector
            lv1 = np.array([dx[i, j], dy[i, j], dz[i, j]])
            lv1 /= norm_vec(lv1)

            # normal vector
            nv1 = np.cross(
                [xx[1] - xx[0], yy[1] - yy[0], zz[1] - zz[0]],
                [xx[2] - xx[0], yy[2] - yy[0], zz[2] - zz[0]],
            )
            norm1 = norm_vec(nv1)
            nv1 /= norm1

            # compute S vector (normalized position)
            s1 = np.array([dx[i, j] - xx[0], dy[i, j] - yy[0], dz[i, j] - zz[0]])
            s1 /= norm_vec(s1)

            # project normal in the slant-range plane
            nv1p = project_point_on_plane(nv1, lv1, s1)
            nv1p /= norm_vec(nv1p)
            cos1p = (nv1p * lv1).sum()

            # gamma convention: inverse of the tangent
            gamma1 = cos1p / (1e-12 + np.sqrt(1 - cos1p**2))
            gamma1 = gamma1 if gamma1 > 0 else 0

            # Triangle 2
            # look vector
            lv2 = np.array([dx[i + 1, j + 1], dy[i + 1, j + 1], dz[i + 1, j + 1]])
            lv2 /= norm_vec(lv2)

            # normal vector
            nv2 = -np.cross(
                [xx[1] - xx[3], yy[1] - yy[3], zz[1] - zz[3]],
                [xx[2] - xx[3], yy[2] - yy[3], zz[2] - zz[3]],
            )
            norm2 = norm_vec(nv2)
            nv2 /= norm2

            s2 = np.array(
                [
                    dx[i + 1, j + 1] - xx[3],
                    dy[i + 1, j + 1] - yy[3],
                    dz[i + 1, j + 1] - zz[3],
                ]
            )
            s2 /= norm_vec(s2)

            # project normal in the slant-range plane
            nv2p = project_point_on_plane(nv2, lv2, s2)
            nv2p /= norm_vec(nv2p)
            cos2p = (nv2p * lv2).sum()

            # gamma convention: inverse of the tangent
            gamma2 = cos2p / (1e-12 + np.sqrt(1 - cos2p**2))
            gamma2 = gamma2 if gamma2 >= 0 else 0

            # project into SAR geometry
            for a in range(amin, amax):
                for r in range(rmin, rmax):
                    if is_in_tri([a, r], aarr[0], aarr[1], aarr[2]):
                        gamma_proj[a, r] += gamma1
                    if is_in_tri([a, r], aarr[3], aarr[1], aarr[2]):
                        gamma_proj[a, r] += gamma2

    for a in prange(gamma_proj.shape[0]):
        for r in range(gamma_proj.shape[1]):
            if gamma_proj[a, r] == 0.0:
                gamma_proj[a, r] = np.nan

    return gamma_proj

def load_metadata(zip_path, subswath, polarization, log=None):
    if zip_path.endswith(".zip"):
        archive = ZipFile(zip_path)
        archive_files = archive.namelist()
    else:
        archive_files = glob(f"{zip_path}/**", recursive=True)
    regex_filter = r"s1(?:a|b|c)-iw\d-slc-(?:vv|vh|hh|hv)-.*\.xml"
    metadata_file_list = []
    for item in archive_files:
        if "calibration" in item:
            continue
        match = re.search(regex_filter, item)
        if match:
            metadata_file_list.append(item)
    target_file = None
    for item in metadata_file_list:
        if subswath.lower() in item and polarization.lower() in item:
            target_file = item
    if log is not None:
        log.debug(f"Metadata file list: {metadata_file_list}")
        log.debug(f"Selected metadata file: {target_file}")
        log.debug(f"Target file: {target_file}")
    if zip_path.endswith(".zip"):
        return archive.open(target_file)
    else:
        return open(target_file)

def parse_location_grid(metadata):
    tree = ET.parse(metadata)
    root = tree.getroot()
    lines = []
    coord_list = []
    for grid_list in root.iter("geolocationGrid"):
        for point in grid_list:
            for item in point:
                lat = item.find("latitude").text
                lon = item.find("longitude").text
                line = item.find("line").text
                lines.append(line)
                coord_list.append((float(lat), float(lon)))
    total_num_bursts = len(set(lines)) - 1

    return total_num_bursts, coord_list

def parse_subswath_geometry(coord_list, total_num_bursts):
    def get_coords(index, coord_list):
        coord = coord_list[index]
        assert isinstance(coord[1], float)
        assert isinstance(coord[0], float)
        return coord[1], coord[0]

    bursts_dict = {}
    top_right_idx = 0
    top_left_idx = 20
    bottom_left_idx = 41
    bottom_right_idx = 21

    for burst_num in range(1, total_num_bursts + 1):
        burst_polygon = Polygon(
            [
                [
                    get_coords(top_right_idx, coord_list)[0],
                    get_coords(top_right_idx, coord_list)[1],
                ],  # Top right
                [
                    get_coords(top_left_idx, coord_list)[0],
                    get_coords(top_left_idx, coord_list)[1],
                ],  # Top left
                [
                    get_coords(bottom_left_idx, coord_list)[0],
                    get_coords(bottom_left_idx, coord_list)[1],
                ],  # Bottom left
                [
                    get_coords(bottom_right_idx, coord_list)[0],
                    get_coords(bottom_right_idx, coord_list)[1],
                ],  # Bottom right
            ]
        )

        top_right_idx += 21
        top_left_idx += 21
        bottom_left_idx += 21
        bottom_right_idx += 21

        bursts_dict[burst_num] = burst_polygon

    return bursts_dict

def crop_with_latlon(input_path, output_path, min_lon, min_lat, max_lon, max_lat):
    """
    Crops a raster image to a specific latitude and longitude bounding box.

    Args:
        input_path (str): The path to the input raster file.
        output_path (str): The path to save the cropped raster file.
        min_lon (float): The minimum longitude of the bounding box.
        min_lat (float): The minimum latitude of the bounding box.
        max_lon (float): The maximum longitude of the bounding box.
        max_lat (float): The maximum latitude of the bounding box.
    """
    # Create a Shapely polygon from the bounding box
    bbox_shape = box(min_lon, min_lat, max_lon, max_lat)

    with rasterio.open(input_path) as src:
        # Get the source image's CRS
        source_crs = src.crs

        # If the CRS is not EPSG:4326 (WGS84), transform the bounding box
        if source_crs.to_epsg() != 4326:
            import pyproj
            transformer = pyproj.Transformer.from_crs("EPSG:4326", source_crs, always_xy=True)
            transformed_bounds = [
                transformer.transform(min_lon, min_lat),
                transformer.transform(max_lon, max_lat)
            ]
            bbox_shape = box(
                transformed_bounds[0][0],
                transformed_bounds[0][1],
                transformed_bounds[1][0],
                transformed_bounds[1][1]
            )

        # Mask the raster with the bounding box shape
        out_image, out_transform = mask(src, [bbox_shape], crop=True)

        # Update the metadata for the output file
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Write the cropped image to the output file
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(out_image)

def merge_tiles(files_to_merge, output_path):
    # Open all datasets in a list
    src_files_to_mosaic = []
    for fp in files_to_merge:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)

    # Merge the rasters
    mosaic, out_trans = merge(src_files_to_mosaic)

    # Update the metadata for the output file
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })

    # Close the source files
    for src in src_files_to_mosaic:
        src.close()

    # Write the merged mosaic to a new file
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

def remap(img, rr, cc, kernel="bicubic"):
    """Resample an image using row, column lookup tables

    Args:
        img (array): image to resample (complex is allowed)
        rr (array): lookup table for row positions
        cc (array): lookup table for column positions
        kernel (str, optional): Kernel type ("nearest", "bilinear", "bicubic" -- 4 point, "bicubic6" -- 6 point). Defaults to "bicubic".

    Returns:
        array: Resampled image with same dimensions as rr and cc.
    """
    if np.iscomplexobj(img):
        return _remap(img.real, rr, cc, kernel) + 1j * _remap(img.imag, rr, cc, kernel)
    else:
        return _remap(img, rr, cc, kernel)

@njit(parallel=True, nogil=True, cache=True)
def _remap(img, rr, cc, kernel="bicubic"):

    if rr.shape != cc.shape:
        raise ValueError("Coordinate arrays must have the same shape.")

    arr_out = np.full_like(rr, np.nan, dtype=img.dtype)
    if kernel == "nearest":
        ker = _ker_near
        H = 0
    elif kernel == "bilinear":
        ker = _ker_lin
        H = 0
    elif kernel == "bicubic":
        ker = _ker_cub
        H = 1
    elif kernel == "bicubic6":
        ker = _ker_cub6
        H = 2
    else:
        raise ValueError("Unknown interpolation type.")

    img_shape_0 = img.shape[0]
    img_shape_1 = img.shape[1]
    i2max = img_shape_0 - 1
    j2max = img_shape_1 - 1

    for idx in prange(len(rr.flat)):
        r = rr.flat[idx]
        c = cc.flat[idx]

        if np.isnan(r) | np.isnan(c):
            continue
        is_in_image = (r >= 0) & (r < img_shape_0) & (c >= 0) & (c < img_shape_1)
        if not is_in_image:
            continue

        # change boundaries if using other kernels
        rmin = np.floor(r) - H
        rmax = np.ceil(r) + H
        cmin = np.floor(c) - H
        cmax = np.ceil(c) + H

        val = 0.0
        for i in range(int(rmin), int(rmax) + 1):
            for j in range(int(cmin), int(cmax) + 1):
                # using nearest neighbor on image border
                i2 = min(max(0, i), i2max)
                j2 = min(max(0, j), j2max)
                val += ker(r - i) * ker(c - j) * img[i2, j2]
        arr_out.flat[idx] = val
    return arr_out

@cfunc("double(double)")
def _ker_near(x):
    ax = np.abs(x)
    if ax < 0.5:
        return 1.0
    elif ax == 0.5:
        return 0.5
    else:
        return 0.0

@cfunc("double(double)")
def _ker_cub6(x):
    """6-point bicubic kernel described in Keys81"""
    a = -0.5
    b = 0.5
    ax = np.abs(x)
    ax2 = ax**2
    ax3 = ax**3
    if ax < 1:
        return 4 * ax3 / 3 - 7 * ax2 / 3 + 1
    elif (ax >= 1) & (ax < 2):
        return -7 * ax3 / 12 + 3 * ax2 - 59 * ax / 12 + 15 / 6
    elif (ax >= 2) & (ax < 3):
        return ax3 / 12 - 2 * ax2 / 3 + 21 * ax / 12 - 3 / 2
    else:
        return 0.0


@cfunc("double(double)")
def _ker_lin(x):
    ax = np.abs(x)
    if ax < 1:
        return 1.0 - ax
    else:
        return 0.0


@cfunc("double(double)")
def _ker_cub(x):
    ax = np.abs(x)
    if ax < 1:
        return 1.5 * ax**3 - 2.5 * ax**2 + 1
    elif (ax >= 1) & (ax < 2):
        return -0.5 * ax**3 + 2.5 * ax**2 - 4 * ax + 2
    else:
        return 0.0

def presum(img, m, n):
    """
    Computes the m by n presummed image.

    Args:
        img (array-like): Input image array with shape (naz, nrg,...).
        m (int): Number of lines to sum. Must be an integer >= 1.
        n (int): Number of columns to sum. Must be an integer >= 1.

    Raises:
        TypeError: If m or n are not integers.
        ValueError: If m or n are less than 1, or if m > img.shape[0] or n > img.shape[1].

    Returns:
        array: Presummed image array with shape (M, N,...), where M and N are the largest multiples of m and n that are less than or equal to img.shape[0] and img.shape[1], respectively.
    Note:
        Returns the input array if m==1 and n==1.
    """
    # Check if m and n are integers >= 1
    if not isinstance(m, int) or not isinstance(n, int):
        raise TypeError("Parameters m and n must be integers.")
    if m < 1 or n < 1:
        raise ValueError(
            "Parameters m and n must be integers greater than or equal to 1."
        )

    # Check if m and n are valid in relation to the image dimensions
    if m > img.shape[0] or n > img.shape[1]:
        raise ValueError(
            "Cannot presum with these parameters; m or n is too large for the image dimensions."
        )

    # skip if m = n = 1, avoids conditionals in calls
    if (m > 1) or (n > 1):
        M = (img.shape[0] // m) * m
        N = (img.shape[1] // n) * n

        img_trimmed = img[:M, :N]

        s = img_trimmed[::m].copy()  # Make a copy once for efficiency
        for i in range(1, m):
            s += img_trimmed[i::m]

        t = s[:, ::n].copy()
        for j in range(1, n):
            t += s[:, j::n]

        return t / float(m * n)
    else:
        return img

def read_chunk(pth_tiff, first_line=0, number_of_lines=1500):

    with rasterio.open(pth_tiff) as src:
        arr = src.read(
            1, window=Window(0, first_line, src.width, number_of_lines)
        ).astype("complex64")
    return arr

def lee_filter(img:np.ndarray, size:int=3, mode='nearest'):
    """
    Applies lee filter using fixed kernel size
    """
    iarr = np.nan_to_num(img)
    img_mean = uniform_filter(iarr, (size, size), mode=mode)
    img_sqr_mean = uniform_filter(iarr**2, (size, size), mode=mode)
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(iarr)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (iarr - img_mean)
    return img_output