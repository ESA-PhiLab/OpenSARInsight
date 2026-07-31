"""
Main script for geocoding
"""

import numpy as np 
from affine import Affine
from concurrent.futures import ThreadPoolExecutor
from dateutil.parser import isoparse 
from dem_stitcher.stitcher import stitch_dem  
from geopandas import GeoDataFrame
from hashlib import md5
from logging import Logger
from lxml import etree 
from os import path as os_path
from os import remove as os_remove
from os import listdir as os_listdir
from pandas import read_excel, concat
from pathlib import Path 
from pipeline.geocoding_block.utils import _shadow_mask, range_doppler, simulate_terrain_backscatter
from pipeline.geocoding_block.utils import load_metadata, parse_location_grid, parse_subswath_geometry
from pipeline.geocoding_block.utils import remap, presum, read_chunk, lee_filter
from planetary_computer import sign_inplace
from pyproj.sync import get_proj_endpoint 
from pyproj.datadir import get_user_data_dir
from pyproj import Transformer
from pystac_client.client import Client
from rasterio import open as rio_open
from rasterio.windows import Window
from rasterio.warp import reproject, calculate_default_transform 
from rasterio.enums import Resampling
from rasterio.transform import xy as rio_transform_xy
from rasterio.errors import NotGeoreferencedWarning 
from rioxarray import open_rasterio
from rioxarray.merge import merge_arrays
from scipy.interpolate import CubicHermiteSpline, RegularGridInterpolator
from shapely.geometry import box as shapely_box
from shapely.geometry import base
from typing import Union, Dict, Tuple, List
from urllib.request import urlretrieve
from warnings import warn, filterwarnings
from xmltodict import parse as xml_parse
from yaml import safe_load

class Geocoder:
    """
    Class to geocode patches, you should only call run() method outside of this class.
    """
    def __init__(self, log:Logger, cfg_path:Path, label_fn:str):
        """
        log (Logger): python logger
        cfg_path (Path): path to configuration yaml file
        label_fn (str): label filename for the patch to geocode
        """
        self.log = log
        self.cfg = self._load_config(cfg_path)
        self.label_fn = label_fn 
        skip_geocoding = self._skip_if_geocoded()
        if skip_geocoding:
            return
        tree = etree.parse(Path(self.cfg["labels_directory_path"],self.label_fn))
        self.pol = self.cfg["pol"]
        self.iw = int(tree.xpath("//SARData//SLCSwath")[0].text)
        self.use_case = self.label_fn.split("_")[2]
        self.lut = self._parse_lut()
        label_fp = Path(self.cfg["labels_directory_path"],self.label_fn)
        safe_name = self._get_safe_name(label_fp)
        safe_path = Path(self.cfg["product_directory_path"],safe_name)
        self.log.debug(f"Safe path: {safe_path}")
        cal_path = list(safe_path.glob(f"**/annotation/calibration/calibration*iw{self.iw}*{self.pol}*.xml"))[0]
        self.ant_fp = list(safe_path.glob(f"**/annotation/*iw{self.iw}*{self.pol}*.xml"))[0]
        self.pth_tiff = list(Path(safe_path).glob(f"**/measurement/*iw{self.iw}*{self.pol}*.tiff"))[0] 
        self.dem_fp = None
        self.meta = None
        self.burst_info = None
        self.burst_count = None 
        self.lines_per_burst = None 
        self.samples_per_burst  = None
        self._set_metadata()
        self.orbit_fp =  Path(self.cfg["orbit_directory_path"],self.lut[f"{safe_path.stem}.SAFE"])
        self.state_vectors = self._get_svs()
        self.gdf_burst_geom = self._get_burst_geometry(
            path=str(safe_path), target_subswaths=f"IW{self.iw}", polarization=self.pol.upper()
        )
        # extract calibration LUT to rescale data
        calinfo=None
        with cal_path.open() as f:
            calinfo = xml_parse(f.read())
        self.calvec = calinfo["calibration"]["calibrationVectorList"][
            "calibrationVector"
        ]
        BN_str = self.calvec[0]["betaNought"]["#text"]
        self.beta_nought = float(BN_str.split(" ")[0])

    def _skip_if_geocoded(self) -> bool:
        """
        Skips geocoding if the given patch is already present in dem directory (unless overwrite = true)
        """
        skip_geocoding = False
        dem_dir = self.cfg["dem_directory_path"]
        label = self.label_fn.split(".")[0]
        geocoded_patch_path = Path(dem_dir,f"{label}_geocoded_final.tiff")
        if geocoded_patch_path.exists() and self.cfg["overwrite"] == False:
            skip_geocoding = True 
        return skip_geocoding            

    def run(self) -> Union[Path, None]:
        """
        Geocodes the patch

        Returns:
            geocoded_img_fp (Path): path to geocoded patch
        """
        skip_geocoding = self._skip_if_geocoded()
        label_fp = Path(self.cfg["labels_directory_path"],self.label_fn)
        label = self.label_fn.split(".")[0]
        if skip_geocoding:
            self.log.info(f"Skipping geocoding for {label} as it has already been geocoded.")
            return None
        original_dem_fp, self.dem_fp = self._get_dem()
        dem_dir = self.cfg["dem_directory_path"]
        self.log.debug(f"DEM for {label} saved to {original_dem_fp}")
        complex_image_path = str(Path(dem_dir,f"{label}_SLC_VH.tiff"))
        amp_path = str(Path(dem_dir,f"{label}_amplitude.tiff"))
        cal_amp_fp = str(Path(dem_dir,f"{label}_calibrated_amp.tiff"))
        cog_fp = str(Path(dem_dir,f"{label}_cog.tiff"))
        calibrated_burst_fp, lut_file = self._get_lut_and_calibrated_burst()
        b = self._get_patch_coords(label_fp)
        img_arr=open_rasterio(calibrated_burst_fp)
        # adjust offset for burst
        bn = self._get_burst_number(label_fp)
        y0=int(b[0]) - (bn-1)*self.lines_per_burst
        x0=int(b[1])
        y1=int(b[2]) - (bn-1)*self.lines_per_burst
        x1=int(b[3])
        iarr = img_arr.isel(x=slice(x0,x1,1),y=slice(y0,y1,1))
        iarr.rio.to_raster(complex_image_path)
        self._amplitude(complex_image_path,amp_path)
        self._amplitude(calibrated_burst_fp,cal_amp_fp)
        self._sar2geo(
            sar_file=cal_amp_fp,
            lut_file=lut_file,
            out_file=cog_fp,
            kernel="bicubic",
            write_phase=True,
            magnitude_only=False,
        )
        geocoded_img_fp = self._merge_cog_with_slc(cal_amp_fp, cog_fp)
        self.log.debug(f"Geocoded patch saved to {geocoded_img_fp}")
        # clean up files
        fns_to_delete = os_listdir(geocoded_img_fp.parent)
        for fn in fns_to_delete:
            # keep dem and final geocoded patch files
            if 'geocoded_final' not in fn and 'dem' not in fn:
                os_remove(Path(geocoded_img_fp.parent, fn))
        self.log.info(f"Completed geocoding for {label}!")
        return geocoded_img_fp

    def _load_config(self, cfg_path:Path) -> Union[None, Dict]:
        """
        Loads configuration from yaml file

        Args:
            cfg_path (Path): path to configuration yaml file
        Returns:
            cfg (Dict): configuration dictionary
        """
        cfg = None
        with open(cfg_path) as cfg_file:
            cfg = safe_load(cfg_file)
        return cfg

    def _parse_lut(self) -> Dict:
        """
        Gets the matches from the LUT

        Args:
            product_lut (Dict): product to orbit file LUT
        """
        df = read_excel(self.cfg["lut_file_path"])
        lut = df.loc[df["USE_CASE"]==self.use_case].reset_index(drop=True)
        product_lut = dict()
        pids = lut["L1_PRODUCT_NAME_INTA"].apply(lambda x: x.strip())
        oids = lut["ORBIT_FILE_NAME"].apply(lambda x: x.strip())
        for i in range(len(pids)):
            product_lut[pids[i]] = oids[i]
        return product_lut
    
    def _get_safe_name(self, label_fp:Path) -> str:
        """
        Gets the SAFE name from the label file

        Args:
            label_fp (Path): path to label file
        Returns:
            safe_name (str): SAFE name
        """
        tree = etree.parse(label_fp)
        safe_name = tree.xpath("//SARProduct")[0].text
        if safe_name[-5:] != ".SAFE":
            safe_name = safe_name + ".SAFE"
        return safe_name
    
    def _get_patch_coords(self, label_fp:Path) -> List:
        """
        Gets the patch coordinates from the label file

        Args:
            label_fp (Path): path to label file
        
        Returns:
            patch_coords (List): [min_line, min_sample, max_line, max_sample]
        """
        tree = etree.parse(label_fp)
        sample_coords = list(map(int,[v.text for v in tree.xpath("//ProcessingData//Corner_Coord//SARData_Sample")][0].split(' ')))
        line_coords = list(map(int,[v.text for v in tree.xpath("//ProcessingData//Corner_Coord//SARData_Line")][0].split(' ')))
        patch_coords = list([min(line_coords), min(sample_coords), max(line_coords), max(sample_coords)])
        return patch_coords

    def _get_burst_number(self, label_fp:Path) -> int:
        """
        Gets the burst number from the label file

        Args:
            label_fp (Path): path to label file
        
        Returns:
            burst_number (int): burst number
        """
        tree = etree.parse(label_fp)
        # note this is not correct for the patches so it currently overwritten
        try:
            burst_number = int(tree.xpath("//SARData//L1_burst_id")[0].text)
        except Exception as e:
            self.log.warning(f"L1 burst ID not found in label, attempting to calculate L1 burst number using L0 burst number.")
            burst_number = None
            try:
                # L1 burst number is offset by 2 from L0 burst number
                burst_number = int(tree.xpath("//SARData//L0_burst_id")[0].text)
                self.log.debug(f"Successfully retrieved L0 burst number: {burst_number}.")
                burst_number -= 2
                self.log.info(f"Calculated L1 burst number as {burst_number} using L0 burst number.")
            except Exception as e:
                self.log.error(f"Failed to retrieve L0 burst number. Cannot determine burst number for patch. Error: {e}")
        return burst_number

    def _get_label_bounds(self, lab_path) -> List:
        """
        Gets the lat/lon bounds from the label file

        Args:
            lab_path (Path): path to label file
        
        Returns:
            bounds (List): [min_lon, min_lat, max_lon, max_lat]
        """
        tree = etree.parse(lab_path)
        lats = list(map(float,[v.text for v in tree.xpath("//ProcessingData//Corner_Coord//Latitude")][0].split(' ')))
        lons = list(map(float,[v.text for v in tree.xpath("//ProcessingData//Corner_Coord//Longitude")][0].split(' ')))
        bounds = list([min(lons), min(lats), max(lons), max(lats)])
        return bounds

    def _get_dem(self) -> Path:
        """
        Gets the DEM for the patch

        Returns:
            dem_fp (Path): path to DEM file
        """
        dem_dir = self.cfg["dem_directory_path"]
        dem_server = self.cfg["dem_server"]
        label_fp = Path(self.cfg["labels_directory_path"],self.label_fn)
        bn1 = self._get_burst_number(label_fp)
        dem_fp=self._fetch_dem(bn1, bn1, dem_dir=dem_dir, dem_name="cop-dem-glo-30", dem_server=dem_server)
        self.log.debug(f"Saved DEM for burst {bn1} to {dem_fp}")
        img_arr=open_rasterio(dem_fp)
        img_arr = np.flip(img_arr, axis=2)
        bbox = self._get_label_bounds(label_fp)
        shp = shapely_box(*bbox)
        iarr = img_arr.rio.clip([shp])
        cropped_dem_fp=Path(dem_dir,f"{label_fp.stem}_cropped_DEM.tiff")
        iarr.rio.to_raster(cropped_dem_fp)
        scaled_dem_fp = Path(dem_dir,f"{label_fp.stem}_scaled_DEM.tiff")
        b = self._get_patch_coords(label_fp)
        patch_height = b[2] - b[0]
        patch_width = b[3] - b[1]
        self._rescale_image(cropped_dem_fp, scaled_dem_fp, patch_width, patch_height)
        self.log.debug(f"Saved scaled DEM ({patch_width}x{patch_height}) patch to {scaled_dem_fp}")
        os_remove(cropped_dem_fp)
        return dem_fp, scaled_dem_fp
    
    def _rescale_image(self, input_img_path, output_img_path, new_width, new_height, resampling_method='bilinear') -> None:
        """
        Rescales the input image to the specified width and height using the specified resampling method.

        Args:
            input_img_path (str): Path to the input image file.
            output_img_path (str): Path to save the rescaled image file.
            new_width (int): Desired width of the rescaled image.
            new_height (int): Desired height of the rescaled image.
            resampling_method (str): Resampling method to use ('bilinear' or 'cubic').
        """
        profile, new_data, new_transform = None, None, None 
        with rio_open(input_img_path) as src:
            def_transform, width, height = calculate_default_transform(src.crs, src.crs, src.width, src.height, src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top, dst_width=new_width, dst_height=new_height)
            resampling_method = Resampling.bilinear if resampling_method == 'bilinear' else Resampling.cubic
            new_data, new_transform = reproject(source=src.read(),
                                destination=np.zeros((src.count, height, width)),
                                src_transform=src.transform,
                                dst_transform=def_transform,
                                src_crs=src.crs,
                                dst_crs=src.crs,
                                dst_nodata=src.nodata,
                                resampling=Resampling.bilinear)

            profile = src.profile
        profile.update(transform=new_transform, driver='GTiff',
                            height=new_data.shape[1], width=new_data.shape[2])

        with rio_open(output_img_path, 'w', **profile) as dst:
            dst.write(new_data)


    def _get_lut_and_calibrated_burst(self) -> Tuple:
        """
        Computes the LUT using the dem and the calibrated burst

        Returns:
            calibrated_burst_fp, lut_file (Tuple): paths to calibrated burst and LUT filess
        """ 
        composite_crs = "EPSG:4326+3855" # for cop glo 30 DEM
        dem_dir = self.cfg["dem_directory_path"]
        label_fp = label_fp = Path(self.cfg["labels_directory_path"],self.label_fn)
        label = self.label_fn.split(".")[0]
        lut_file = str(Path(dem_dir,f"{label}_lut.tiff"))
        nrg = self.samples_per_burst
        naz = self.lines_per_burst
        burst_idx = self._get_burst_number(label_fp)
        calibrated_burst_fp = str(Path(dem_dir, f"{label}_burst_{burst_idx}_calibrated.tiff"))

        with rio_open(self.dem_fp,"r+") as ds_dem:
            ds_dem.update_tags(COMPOSITE_CRS=composite_crs)
        prof_tmp = dict(
            width=nrg,
            height=naz,
            count=1,
            dtype="complex64",
            driver="GTiff",
            nodata=np.nan,
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )

        with rio_open(self.dem_fp,"r") as ds_dem:
            width_lut = ds_dem.width
            height_lut = ds_dem.height
            crs_lut = ds_dem.crs
            transform_lut = ds_dem.transform
            w = Window(0, 0, width_lut, height_lut)
            slices = w.toslices()
            # this implementation upsamples DEM at download, not during geocoding
            az_p2g, rg_p2g = self._geocode_burst(
                self.dem_fp,
                burst_idx=burst_idx,
                dem_upsampling=1,
            )
            arr_p = self._read_burst(burst_idx, True)
            cal_p = self._calibration_factor(burst_idx)
            self.log.info("Apply calibration factor")
            arr_p /= cal_p
            first_line = 0
            with rio_open(calibrated_burst_fp, "w", **prof_tmp) as ds_prm:
                ds_prm.write(
                    arr_p, 1, window=Window(0, first_line, nrg, self.lines_per_burst)
                )

            arr_lut = np.full((2, height_lut, width_lut), fill_value=np.nan)
            msk = ~np.isnan(az_p2g)
            arr_lut[0, slices[0], slices[1]][msk] = az_p2g[msk]
            arr_lut[1, slices[0], slices[1]][msk] = rg_p2g[msk]

            prof_lut = dict(
                width=width_lut,
                height=height_lut,
                count=2,
                dtype=np.float64,
                crs=crs_lut,
                transform=transform_lut,
                nodata=np.nan,
                tiled=True,
                blockxsize=512,
                blockysize=512,
            )

        with rio_open(lut_file, "w", **prof_lut) as ds_lut:
            ds_lut.write(arr_lut)
        
        return calibrated_burst_fp, lut_file
    
    def _merge_cog_with_slc(self, cal_amp_fp:str, cog_fp:str) -> Path:
        """
        Merges the COG image with the SLC and outputs a geocoded SLC image

        Args:
            cal_amp_fp (str): path to calibrated amplitude image
            cog_fp (str): path to COG image
        Returns:
            final_img_fp (Path): path to final geocoded image
        """
        dem_dir = self.cfg["dem_directory_path"]
        label_fn = self.label_fn
        label = label_fn.split(".")[0]
        label_fp = label_fp = Path(self.cfg["labels_directory_path"],label_fn)
        slc_fp = str(Path(dem_dir,f"{label}_slc.tiff"))
        geocoded_img_fp = str(Path(dem_dir,f"{label}_geocoded.tiff"))
        final_img_fp = Path(dem_dir,f"{label}_geocoded_final.tiff")
        img_arr=open_rasterio(cal_amp_fp)
        a = self._get_patch_coords(label_fp)
        patch_height = a[2] - a[0]
        patch_width = a[3] - a[1]
        ym = 0
        xm = 0
        # adjust offset for burst
        bn = self._get_burst_number(label_fp)
        yo=(bn-1)*self.lines_per_burst # y offset
        iarr = img_arr.isel(y=slice(a[0]-yo,a[2]-yo+ym),x=slice(a[1],a[3]+xm))
        # this should match up with slc patch in our dataset can manuall check this
        iarr.rio.to_raster(slc_fp)
        #NOTE: slc_fp is how the patch should actually look
        # transform cog image to get geocoded image
        img_profile = None
        img_data = None
        lats, lons, latsxy, lonsxy = None, None, None, None
        with rio_open(slc_fp) as src:
            img_profile = src.profile
            img_data = src.read()
        
        cog_transform = None 
        with rio_open(cog_fp) as src:
            cog_transform = src.transform
        img_profile.update({"transform": cog_transform})

        with rio_open(geocoded_img_fp,"w", **img_profile) as src:
            src.write(img_data)
        
        with rio_open(geocoded_img_fp) as src:
            data = src.read(1)
            rows=[row for row in data]
            columns=[col for col in data.T]
            coords = lambda x,y : src.xy(x,y)
            a = coords(rows,columns)
            lons = a[0]
            lats = a[1]
            latsxy = lats.reshape(data.shape[0], data.shape[1])
            lonsxy = lons.reshape(data.shape[0], data.shape[1])

        # lee filter just blurs the image
        #new_data = lee_filter(data)
        combined_data = np.stack([data, latsxy, lonsxy])
        img_profile.update(count=combined_data.shape[0])

        with rio_open(final_img_fp, 'w', **img_profile) as dst:
            dst.write(combined_data)
        return final_img_fp

    def _geocode_burst(
        self, dem_file, burst_idx=1, dem_upsampling=1, simulate_terrain=False
    ) -> Tuple[np.ndarray, np.ndarray, Dict, Union[None, np.ndarray]]:
        """Computes azimuth-range lookup tables for each pixel of the DEM by solving the Range Doppler equations.

        Args:
            dem_file (str): path to the DEM
            burst_idx (int, optional): Burst index. Defaults to 1.
            dem_upsampling (int, optional): DEM upsampling to increase the resolution of the geocoded image. Defaults to 2.
            simulate_terrain (bool): terrain backscatter simulation in the SAR geometry which can be used for terrain flattening.

        Returns:
            (array, array, dict, optional array): azimuth and slant range indices. Arrays have the shape of the DEM. Also returns the rasterio profile of the DEM as a dict. If simulate_terrain is set to True, returns gamma_t, the simulated terrain backscatter of the burst in the SAR geometry.
        """

        if burst_idx < 1 or burst_idx > self.burst_count:
            raise ValueError(
                f"Invalid burst index (must be between 1 and {self.burst_count})"
            )

        if dem_upsampling < 0:
            raise ValueError("dem_upsampling must be > 0")

        meta = self.meta

        # general info
        image_info = meta["product"]["imageAnnotation"]["imageInformation"]
        azimuth_time_interval = image_info["azimuthTimeInterval"]
        slant_range_time = image_info["slantRangeTime"]
        product_info = meta["product"]["generalAnnotation"]["productInformation"]
        range_sampling_rate = product_info["rangeSamplingRate"]

        # look for burst info
        burst_info = meta["product"]["swathTiming"]
        if burst_idx > self.burst_count or burst_idx < 1:
            raise ValueError(f"Burst index must be between 1 and {self.burst_count}")
        burst = burst_info["burstList"]["burst"][burst_idx - 1]
        az_time = burst["azimuthTime"]

        # state vectors
        # orbit_list = meta["product"]["generalAnnotation"]["orbitList"]
        # state_vectors = orbit_list["orbit"]

        if dem_upsampling != 1:
            self.log.info("Resample DEM and extract coordinates")
        else:
            self.log.info("Extract DEM coordinates")
        lat, lon, alt, dem_prof, composite_crs = self.load_dem_coords(
            dem_file, dem_upsampling
        )

        self.log.info("Convert latitude, longitude & altitude to ECEF x, y & z")
        dem_x, dem_y, dem_z = self.lla_to_ecef(lat, lon, alt, composite_crs)

        tt0 = self.state_vectors["t0"]
        t0_az = (isoparse(az_time) - tt0).total_seconds()
        dt_az = float(azimuth_time_interval)
        naz = self.lines_per_burst
        nrg = self.samples_per_burst

        t_end_burst = t0_az + dt_az * naz
        t_sv_burst = self.state_vectors["t"]

        # crop a few minutes before and after burst
        t_spacing = 10
        t_pad = t_spacing * 36
        cnd = (t_sv_burst > t0_az - t_pad) & (t_sv_burst < t_end_burst + t_pad)

        state_vectors = {k: v[cnd] for k, v in self.state_vectors.items() if k != "t0"}

        interp_pos, interp_vel = self._sv_interpolator(state_vectors)
        # interp_pos, interp_vel = sv_interpolator_poly(state_vectors)

        self.log.info("Interpolate orbit")
        t_arr = np.linspace(t0_az, t0_az + dt_az * (naz - 1), naz)
        pos = interp_pos(t_arr)
        vel = interp_vel(t_arr)

        self.log.info("Range-Doppler terrain correction (LUT computation)")
        if simulate_terrain:
            az_geo, dist_geo, dx, dy, dz = range_doppler(
                # Removing first pos to get more precision. Is this useful?
                dem_x.ravel() - pos[0, 0],
                dem_y.ravel() - pos[0, 1],
                dem_z.ravel() - pos[0, 2],
                pos - pos[0],
                vel,
                tol=1e-8,
                maxiter=10000,
            )
        else:
            az_geo, dist_geo, _, _, _ = range_doppler(
                # Removing first pos to get more precision. Is this useful?
                dem_x.ravel() - pos[0, 0],
                dem_y.ravel() - pos[0, 1],
                dem_z.ravel() - pos[0, 2],
                pos - pos[0],
                vel,
                tol=1e-8,
                maxiter=10000,
            )

        # convert range - azimuth to pixel indices
        c0 = 299792458.0
        r0 = float(slant_range_time) * c0 / 2
        dr = c0 / (2 * float(range_sampling_rate))
        rg_geo = (dist_geo - r0) / dr

        # masking points with invalid radar coordinates
        cnd1 = (rg_geo >= 0) & (rg_geo < nrg)
        cnd2 = (az_geo >= 0) & (az_geo < naz)
        valid = cnd1 & cnd2
        rg_geo[~valid] = np.nan
        az_geo[~valid] = np.nan

        # reshape to DEM dimensions
        rg_geo = rg_geo.reshape(alt.shape)
        az_geo = az_geo.reshape(alt.shape)

        if simulate_terrain:
            dx[~valid] = np.nan
            dy[~valid] = np.nan
            dz[~valid] = np.nan

            # reshape to DEM dimensions
            dx = dx.reshape(alt.shape)
            dy = dy.reshape(alt.shape)
            dz = dz.reshape(alt.shape)

            # finding occluded shadow pixels
            self.log.info("Shadow detection")
            # compute ero altitude coordinates (use DEM reference height)
            dem_xg, dem_yg, dem_zg = self.lla_to_ecef(
                lat,
                lon,
                np.zeros_like(lat),
                composite_crs,
            )

            shadow_mask = self.detect_active_shadow(
                az_geo, dem_xg, dem_yg, dem_zg, dem_x, dem_y, dem_z, dx, dy, dz
            )

            # simulating terrain backscatter
            self.log.info("Terrain simulation")
            gamma_t = simulate_terrain_backscatter(
                naz, nrg, az_geo, rg_geo, dem_x, dem_y, dem_z, dx, dy, dz, shadow_mask
            )

            return az_geo, rg_geo, gamma_t
        else:
            return az_geo, rg_geo

    def _get_svs(self) -> Union[None, Dict]:
        """
        Gets the state vectors (SVs)

        Returns:
            state_vectors (Dict): dictionary of state vectors
        """
        early_stop, orbdict, orbdata, state_vectors = False, None, None, None

        if not self.orbit_fp.exists():
            self.log.error(f"{self.orbit_fp} does not exist!")
            early_stop = True 
        if not early_stop:
            with self.orbit_fp.open() as f:
                orbdict = xml_parse(f.read())
            self.log.debug("Loaded orbit file into orbdict")
            self.log.debug("Extracting state vectors from orbdict...")
            try:
                orbdata = orbdict["Earth_Explorer_File"]["Data_Block"]["List_of_OSVs"]["OSV"]
            except KeyError as err_msg :
                self.log.error(f"{self.orbit_fp} is missing one or more keys (Earth_Explorer_File, Data_Block, List_of_OSVs, OSV)")
                self.log.error(err_msg)
                early_stop = True
        if not early_stop:
            state_vectors = dict()
            try:
                t0 = isoparse(orbdata[0]["UTC"][4:])
                state_vectors["t0"] = t0
                times = list()
                xs = list()
                ys = list()
                zs = list()
                vxs = list()
                vys = list()
                vzs = list()
                
                for it in orbdata:
                    times.append((isoparse(it["UTC"][4:]) - t0).total_seconds())
                    xs.append(float(it["X"]["#text"]))
                    ys.append(float(it["Y"]["#text"]))
                    zs.append(float(it["Z"]["#text"]))
                    vxs.append(float(it["VX"]["#text"]))
                    vys.append(float(it["VY"]["#text"]))
                    vzs.append(float(it["VZ"]["#text"]))

                state_vectors.update({
                    "t": np.array(times),
                    "x": np.array(xs),
                    "y": np.array(ys),
                    "z": np.array(zs),
                    "vx": np.array(vxs),
                    "vy": np.array(vys),
                    "vz": np.array(vzs)
                })
            except KeyError as err_msg:
                self.log.error(f"{self.orbit_fp} missing one or more state vector keys (x, y, z, vx, vy, vz)")
            except IndexError:
                self.log.error(f"{self.orbit_fp} missing UTC entries!")
            except Exception as err_msg:
                self.log.error(err_msg)
                early_stop = True 

        if not early_stop:
            self.log.debug(f"Extracted {len(state_vectors.get('t'))} state vectors from orbit file")
            self.log.info(f"Extracted state vectors from {self.orbit_fp}")
        return state_vectors

    def _set_metadata(self) -> None:
        """
        Extracts the relevant metadata from annotation file
        """
        with self.ant_fp.open() as f:
            self.meta = xml_parse(f.read())
        
        try:
            self.burst_info = self.meta["product"]["swathTiming"]
            self.burst_count = int(self.burst_info["burstList"]["@count"])
            self.lines_per_burst = int(self.burst_info["linesPerBurst"])
            self.samples_per_burst = int(self.burst_info["samplesPerBurst"])
        except KeyError as err_msg:
            self.log.error(f"{self.ant_fp} is missing keys")
            self.log.error(err_msg)

    def load_dem_coords(self, dem_file:str, upscale_factor:int=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, str]:
        """
        Loads DEM coordinates (lat, lon, alt) from a DEM file.
        
        Args:
            dem_file (str): path to the DEM file
            upscale_factor (int, optional): factor to upscale the DEM resolution. Defaults to 1
        
        Returns:
            (array, array, array, dict, str): latitude, longitude, altitude arrays, rasterio profile of the DEM as a dict, and composite CRS string.
        """
        with rio_open(dem_file) as ds:
            if upscale_factor != 1:
                # on-read resampling
                alt = ds.read(
                    out_shape=(
                        ds.count,
                        int(ds.height * upscale_factor),
                        int(ds.width * upscale_factor),
                    ),
                    resampling=Resampling.bilinear,
                    # resampling=Resampling.cubic,
                )[0]
                # scale image transform
                dem_prof = ds.profile.copy()
                dem_trans = ds.transform * ds.transform.scale(
                    (ds.width / alt.shape[-1]), (ds.height / alt.shape[-2])
                )
                if "COMPOSITE_CRS" in ds.tags():
                    composite_crs = ds.tags()["COMPOSITE_CRS"]
                else:
                    raise KeyError("DEM file needs to have a tag named 'COMPOSITE_CRS'.")
                nodata = ds.nodata
            else:
                alt = ds.read(1)
                dem_prof = ds.profile.copy()
                dem_trans = ds.transform
                if "COMPOSITE_CRS" in ds.tags():
                    composite_crs = ds.tags()["COMPOSITE_CRS"]
                else:
                    raise KeyError("DEM file needs to have a tag named 'COMPOSITE_CRS'.")
                nodata = ds.nodata

        # output lat-lon coordinates
        width, height = alt.shape[1], alt.shape[0]
        if dem_trans[1] > 1.0e-8 or dem_trans[3] > 1.0e-8:
            grid = np.meshgrid(np.arange(width), np.arange(height))
            lat, lon = rio_transform_xy(dem_trans, grid[1].ravel(), grid[0].ravel())
            lat = np.array(lat)
            lon = np.array(lon)
        else:
            # much faster
            ix, iy = np.arange(width), np.arange(height)
            lat_ = dem_trans[0] * ix + dem_trans[2]
            lon_ = dem_trans[4] * iy + dem_trans[5]
            lon = lon_[:, None] + np.zeros_like(alt)
            lat = lat_[None, :] + np.zeros_like(alt)

        # make sure nodata is nan in output
        if not np.isnan(nodata):
            msk = alt == nodata
        alt = alt.astype("float64")
        if not np.isnan(nodata):
            alt[msk] = np.nan

        dem_prof.update({"width": width, "height": height, "transform": dem_trans})
        return lat, lon, alt, dem_prof, composite_crs

    def lla_to_ecef(self, lat: np.ndarray, lon: np.ndarray, alt: np.ndarray, composite_crs: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts latitude, longitude, altitude coordinates to ECEF x, y, z coordinates.

        Args:
            lat (array): latitude array
            lon (array): longitude array
            alt (array): altitude array
            composite_crs (str): composite CRS string of the input coordinates
        
        Returns:
            (array, array, array): ECEF dem_x, dem_y, dem_z arrays
        """
        # WGS84_crs = "EPSG:4326+5773"
        ECEF_crs = "EPSG:4978"
        if composite_crs == "EPSG:4326+5773":
            grid_name = "us_nga_egm96_15.tif"
        elif composite_crs == "EPSG:4326+3855":
            grid_name = "us_nga_egm08_25.tif"
        else:
            raise ValueError("Invalid `composite_crs`. Must be either EPSG:4326+5773 or EPSG:4326+3855")

        grid_repo_url = get_proj_endpoint()
        proj_path = Path(get_user_data_dir())
        if not proj_path.is_dir():
            proj_path.mkdir(parents=True)
        grid_path = proj_path / grid_name
        if not grid_path.exists():
            grid_url = f"{grid_repo_url}/{grid_name}" 
            self.log.info(f"Download {grid_url}")
            urlretrieve(grid_url, grid_path)


        # since pyproj 3.7.0 we need to create one transformer per thread
        def transform_chunk(data_chunk):
            tf = Transformer.from_crs(composite_crs, ECEF_crs)
            return tf.transform(data_chunk[0], data_chunk[1], data_chunk[2])

        chunk = 256
        wgs_pts = [
            (lon[b : b + chunk], lat[b : b + chunk], alt[b : b + chunk])
            for b in range(0, len(lon), chunk)
        ]

        with ThreadPoolExecutor() as executor:
            chunked = executor.map(transform_chunk, wgs_pts)
        chunked = list(chunked)
        dem_x = np.vstack([c[0] for c in chunked])
        dem_y = np.vstack([c[1] for c in chunked])
        dem_z = np.vstack([c[2] for c in chunked])

        return dem_x, dem_y, dem_z

    def _sv_interpolator(self, state_vectors) -> Tuple[CubicHermiteSpline, CubicHermiteSpline]:
        """
        Interpolates state vectors using cubic Hermite splines.

        Args:
            state_vectors (Dict): dictionary of state vectors
        
        Returns:
            (CubicHermiteSpline, CubicHermiteSpline): position and velocity interpolators
        """
        t = state_vectors["t"]
        x = state_vectors["x"]
        y = state_vectors["y"]
        z = state_vectors["z"]
        vx = state_vectors["vx"]
        vy = state_vectors["vy"]
        vz = state_vectors["vz"]

        interp_pos = CubicHermiteSpline(t, np.array([x, y, z]).T, np.array([vx, vy, vz]).T)
        interp_vel = interp_pos.derivative(1)

        return interp_pos, interp_vel

    def _detect_active_shadow(self, az: np.ndarray, dem_xg: float, dem_yg: float, dem_zg: float, 
                             dem_x: float, dem_y: float, dem_z: float, dx: float, dy: float, dz: float) -> np.ndarray:
        """
        Find occluded pixels in DEM according to the sensor zero doppler positions. 
        Reproject the look angles in a monotonic ground geometry so each line represents 
        an azimuth position and each column a distinct range coordinate. 
        Then scan the azimuth lines and find where the look angle is below its stored maximum.

        Args:
            az (array): azimuth lookup table
            dem_xg (float): dem x ground coordinate
            dem_yg (float): dem y ground coordinate
            dem_zg (float): dem z ground coordinate
            dem_x (float): dem x coordinate
            dem_y (float): dem y coordinate
            dem_z (float): dem z coordinate
            dx (float): zero doppler x coordinate
            dy (float):  zero doppler y coordinate
            dz (float): zero doppler z coordinate
        
        Returns:
            array: shadow mask (1 for shadowed pixels, 0 for non-shadowed pixels
        """
        # distance between orbit zero doppler and ellipsoid or egm
        dist0 = np.sqrt(
            (dx - dem_x + dem_xg) ** 2
            + (dy - dem_y + dem_yg) ** 2
            + (dz - dem_z + dem_zg) ** 2
        )

        # look angle for DEM points
        px = dx - dem_x
        py = dy - dem_y
        pz = dz - dem_z
        pn = np.sqrt(px**2 + py**2 + pz**2)
        dn = np.sqrt(dx**2 + dy**2 + dz**2)
        cos_theta = (px * dx + py * dy + pz * dz) / (pn * dn)
        theta = np.arccos(cos_theta)

        # compute zero altitude steps
        d0_diffs = np.sqrt(
            np.diff(dist0, axis=1, append=np.nan) ** 2
            + np.diff(dist0, axis=0, append=np.nan) ** 2
        )
        # rule-of-thumb: use average difference
        delta_d0 = np.nanmean(d0_diffs)

        # convert to index
        rg0 = (dist0 - np.nanmin(dist0)) / delta_d0

        # compute mask by projecting the angle in a ground geometry
        mask = _shadow_mask(theta, rg0, az)

        return mask

    def _get_burst_geometry(self, path:str, target_subswaths:Union[str, list], polarization:str) -> GeoDataFrame:
        """
        Gets the burst geometries for the given subswaths
        
        Args:
            path (str): path to the zip file
            target_subswaths (str or list): target subswath(s) to extract geometry
            polarization (str): polarization to extract geometry
        
        Returns:
            GeoDataFrame: GeoDataFrame with burst geometries
        """
        df_all = GeoDataFrame(
            columns=["subswath", "burst", "geometry"], crs="EPSG:4326"
        )
        if not isinstance(target_subswaths, list):
            target_subswaths_ = [target_subswaths]
        else:
            target_subswaths_ = target_subswaths

        for subswath in target_subswaths_:
            if subswath not in ["IW1", "IW2", "IW3"]:
                raise ValueError("Invalid subswath name (options are: IW1, IW2 or IW3)")
            meta = load_metadata(
                zip_path=path, subswath=subswath, polarization=polarization, log=self.log
            )
            total_num_bursts, coord_list = parse_location_grid(meta)
            subswath_geom = parse_subswath_geometry(coord_list, total_num_bursts)
            df = GeoDataFrame(
                {
                    "subswath": [subswath.upper()] * len(subswath_geom),
                    "burst": [x for x in subswath_geom.keys()],
                    "geometry": [x for x in subswath_geom.values()],
                },
                crs="EPSG:4326",
            )
            df_all = GeoDataFrame(concat([df_all, df]), crs="EPSG:4326")
        return df_all

    def _compute_burst_overlap(self, burst_idx:int=2) -> int:
        """Computes the overlap between a burst and the previous one.
        Used for ESD.

        Args:
            burst_idx (int, optional): Burst index, must be >=2. Defaults to 2.

        Raises:
            ValueError: Burst index is out of bounds.

        Returns:
            int: number of overlapping lines.
        """
        if burst_idx < 2 or burst_idx > self.burst_count:
            raise ValueError(
                f"Invalid burst index (must be between 2 and {self.burst_count})"
            )
        meta = self.meta
        image_info = meta["product"]["imageAnnotation"]["imageInformation"]
        azimuth_time_interval = float(image_info["azimuthTimeInterval"])
        burst_info = meta["product"]["swathTiming"]
        burst_1 = burst_info["burstList"]["burst"][burst_idx - 1]
        az_time_1 = isoparse(burst_1["azimuthTime"])
        burst_2 = burst_info["burstList"]["burst"][burst_idx]
        az_time_2 = isoparse(burst_2["azimuthTime"])

        diff_az_time = (
            az_time_1 - az_time_2
        ).total_seconds() + self.lines_per_burst * azimuth_time_interval
        return diff_az_time / azimuth_time_interval

    def _fetch_dem(
        self,
        min_burst:int=1,
        max_burst:int=None,
        dem_dir:str="/tmp",
        force_download:bool=False,
        upscale_factor:int=1,
        dem_name:str="nasadem",
        dem_server:str="aws",
    ) -> str:
        """Downloads the DEM for a given burst range

        Args:
            min_burst (int, optional): Minimum burst index. Defaults to 1.
            max_burst (int, optional): Maximum burst index. If None, set to last burst. Defaults to None.
            dem_dir (str, optional): Directory to store DEM files. Defaults to "/tmp".
            force_download (bool, optional): Force downloading the file to even if a DEM is already present on disk. Defaults to True.
            dem_name (str, optional): Digital Elevation Model to download. Possible values are 'nasadem', 'cop-dem-glo-30', 'cop-dem-glo-90', 'alos-dem'.
            dem_server (str, optional): Server to use for DEM (aws or msft)
        Returns:
            str: path to the downloaded file
        """

        if not max_burst:
            max_burst_ = self.burst_count
        else:
            max_burst_ = max_burst

        if min_burst < 1 or min_burst > self.burst_count:
            raise ValueError(
                f"Invalid min burst index (must be between 1 and {self.burst_count})"
            )
        if max_burst_ < 1 or max_burst_ > self.burst_count:
            raise ValueError(
                f"Invalid max burst index (must be between 1 and {self.burst_count})"
            )
        if max_burst_ < min_burst:
            raise ValueError("max_burst must be >= min_burst")
        if dem_name not in ["nasadem", "cop-dem-glo-30", "cop-dem-glo-90", "alos-dem"]:
            raise ValueError(
                f"Unkown DEM. Possible values are 'nasadem', 'cop-dem-glo-30', 'cop-dem-glo-90', 'alos-dem'"
            )

        geom_all = self.gdf_burst_geom
        geom_sub = (
            geom_all[
                (geom_all["burst"] >= min_burst) & (geom_all["burst"] <= max_burst_)
            ]
            .union_all()
        )
        shp = shapely_box(*geom_sub.bounds)

        # here we define a unique string for DEM filename
        hash_input = f"{shp.wkt}_{upscale_factor}_{dem_name}".encode("utf-8")
        hash_str = md5(hash_input).hexdigest()
        dem_prefix = f"dem-{hash_str}.tiff"
        dem_file = Path(dem_dir,dem_prefix)
        wrote_to_file = False
        if not os_path.exists(dem_file) or force_download:
            if dem_server == "msft":
                self._retrieve_dem(
                    shp,
                    dem_file,
                    dem_name=dem_name,
                    upscale_factor=upscale_factor,
                )
            else:
                dems_to_try = ['glo_30','srtm_v3', 'nasadem']
                for dem_name in dems_to_try:
                    try:
                        wrote_to_file = self.fetch_chosen_dem(geom_sub.bounds, dem_file, dem_name)
                    except Exception as err_msg:
                        self.log.error(err_msg)
                    if wrote_to_file:
                        break
            if not wrote_to_file:
                raise FileNotFoundError(f"Could not fetch DEM file for given bounds!")
            # write custom tag for geocoding to use the proper vertical CRS
            data, profile = None, None
            with rio_open(dem_file, "r") as ds:
                profile=ds.profile
                data=ds.read()
            profile.update({"nodata":np.nan})
            with rio_open(dem_file, "w", **profile) as ds:
                ds.write(data)
                
        else:
            self.log.info("DEM already on disk")

        return dem_file

    def _retrieve_dem(self, shp:base.BaseGeometry, out_file: str, dem_name: str = "cop-dem-glo-30", 
                     upscale_factor: int = 1) -> None:
        """Downloads a DEM for a given geometry from Microsoft Planetary Computer

        Args:
            shp (base.BaseGeometry): Geometry of the area of interest
            out_file (str, optional): Output file.
            dem_name (str, optional): One of the available collections ('alos-dem', 'cop-dem-glo-30', 'cop-dem-glo-90', 'nasadem'). Defaults to "cop-dem-glo-30".
            tmp_dir (str, optional): Temporary directory where the tiles to be merged and cropped will be stored. Defaults to "/tmp".
            clear_tmp_files (bool, optional): Delete original tiles. Set to False if these are to be reused.
            upscale_factor (float, optional): Upsampling factor.
        """

        self.log.info(f"Retrieve DEM ({dem_name})")
        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=sign_inplace,
        )

        data_keys = {
            "nasadem": "elevation",
            "cop-dem-glo-30": "data",
            "cop-dem-glo-90": "data",
            "alos-dem": "data",
        }

        if dem_name not in data_keys.keys():
            raise ValueError(f"Unknown DEM. Values are {list(data_keys.keys())}.")

        search = catalog.search(collections=[dem_name], intersects=shp)
        items = search.item_collection()

        to_merge = []
        for item in items:
            url = item.assets[data_keys[dem_name]].href
            da = open_rasterio(url)
            to_merge.append(da)

        dem = merge_arrays(to_merge).rio.clip([shp], all_touched=True)
        if upscale_factor == 1: # default is auto scale
            dem_upsampled = dem.rio.reproject(
                dem.rio.crs,
                shape=(int(dem.rio.height), int(dem.rio.width)),
                resampling=Resampling.bilinear,
            )
            dem_upsampled.rio.to_raster(
                out_file, tiled=True, blockxsize=512, blockysize=512
            )  
            self.log.info(f"Saved DEM to {out_file}")
            self.log.info(f"DEM shape = {dem_upsampled.shape}")

        elif upscale_factor > 0:
            self.log.info("Resample DEM")
            new_width = int(dem.rio.width * upscale_factor)
            new_height = int(dem.rio.height * upscale_factor)
            dem_upsampled = dem.rio.reproject(
                dem.rio.crs,
                shape=(new_height, new_width),
                resampling=Resampling.bilinear,
            )
            dem_upsampled.rio.to_raster(
                out_file, tiled=True, blockxsize=512, blockysize=512
            )
        else:
            raise ValueError("Upsampling factor must be positive.")    

    def fetch_chosen_dem(self, bounds, dem_file, dem_name) -> bool:
        """
        Fetches DEM tile for bounds and writes result to dem_file,
        dem_name must be one of: 'srtm_v3', 'nasadem', 'glo_90_missing', 'glo_30', '3dep', 'glo_90'
        """
        wrote_to_dem_file = False
        patch, patch_header = stitch_dem(bounds=bounds,
                            dem_name=dem_name,
                            dst_area_or_point="Area",
                            dst_ellipsoidal_height=False,
                            fill_in_glo_30=True,
                            dst_tile_dir=None,
                            overwrite_existing_tiles=False)
        with rio_open(dem_file, "w", **patch_header) as src:
            src.write(patch, 1)
            src.update_tags(AREA_OR_POINT="Area")
            wrote_to_dem_file = True 
        return wrote_to_dem_file

    def _sar2geo(
        self,
        sar_file: Path | str,
        lut_file: Path | str,
        out_file: Path | str,
        kernel: str = "bicubic",
        write_phase: bool = False,
        magnitude_only: bool = False,
    ) -> None:
        """Reproject slc file to a geographic grid using a lookup table with optional multilooking.

        Args:
            sar_file (Path | str): file in the SAR geometry
            lut_file (Path | str): file containing a lookup table (output of the `preprocess_insar_iw` function)
            out_file (Path | str): output file
            kernel (str): kernel used to align secondary SLC. Possible values are "nearest", "bilinear", "bicubic" and "bicubic6".Defaults to "bilinear".
            write_phase (bool): writes the array's phase . Defaults to False.
            magnitude_only (bool): writes the array's magnitude instead of its complex values. Has no effect it `write_phase` is True. Defaults to False.
        Note:
            Multilooking is recommended as it reduces the spatial resolution and mitigates speckle effects.
        """
        self.log.info("Project image with the lookup table.")

        with rio_open(sar_file) as ds_sar:
            arr = ds_sar.read(1)
            prof_src = ds_sar.profile.copy()
            trans_src = ds_sar.transform

        with rio_open(lut_file) as ds_lut:
            lut = ds_lut.read()
            prof_dst = ds_lut.profile.copy()

        if not trans_src.is_rectilinear:
            raise ValueError("The input dataset is not in the SAR geometry")
        if prof_src["count"] != 1:
            raise ValueError("Only single band rasters are supported.")

        if write_phase and not np.iscomplexobj(arr):
            warn(
                "write_phase: Trying to write phase of a real-valued array. This option will have no effect."
            )
        if magnitude_only and not np.iscomplexobj(arr):
            warn(
                "magnitude_only: Writing magnitude (absolute value) of a real-valued array."
            )

        # check if input was rescaled (multilook, etc.)
        sx = trans_src.a
        sy = trans_src.e
        arr_out = remap(arr, lut[0] / sy, lut[1] / sx, kernel)

        prof_dst.update({k: prof_src[k] for k in ["count", "dtype", "nodata"]})

        cog_dict = dict(
            driver="COG",
            compress="zstd",
            num_threads="all_cpus",
            resampling="nearest",
            overview_resampling="nearest",
        )
        # incompatible with COG, not needed (?) elsewhere
        prof_dst.pop("blockxsize", None)
        prof_dst.pop("blockysize", None)
        prof_dst.pop("tiled", None)
        prof_dst.pop("interleave", None)
        if write_phase and np.iscomplexobj(arr_out):
            phi = np.angle(arr_out)
            nodata = -9999
            phi[np.isnan(phi)] = nodata
            prof_dst.update({"dtype": phi.dtype.name, "nodata": nodata, **cog_dict})
            with rio_open(out_file, "w", **prof_dst) as dst:
                dst.write(phi, 1)
        else:
            if magnitude_only:
                mag = np.abs(arr_out)
                nodata = 0
                mag[np.isnan(mag)] = nodata
                prof_dst.update({"dtype": mag.dtype.name, "nodata": nodata, **cog_dict})
                with rio_open(out_file, "w", **prof_dst) as dst:
                    dst.write(mag, 1)
            else:
                # Using COG only if real-valued
                if not np.iscomplexobj(arr_out):
                    nodata = 0
                    prof_dst.update({"driver": "COG", "nodata": nodata, **cog_dict})
                    arr_out[np.isnan(arr_out)] = nodata
                else:
                    prof_dst.update({"compress": "zstd", "num_threads": "all_cpus"})
                with rio_open(out_file, "w", **prof_dst) as dst:
                    dst.write(arr_out, 1)

    def _amplitude(self, in_file: str, out_file: str, multilook: list = [1, 1]) -> None:
        """Compute the amplitude of a complex-valued image.

        Args:
            in_file (str): GeoTiff file of the primary SLC image
            out_file (str): output file
            multilook (list): number of looks in azimuth and range. Defaults to [1, 1]
        """

        if not isinstance(multilook, list):
            raise ValueError("Multilook must be a list like [mlt_az, mlt_rg]")
        else:
            mlt_az, mlt_rg = multilook

        self.log.info("Compute amplitude")
        with rio_open(in_file) as ds_slc:
            slc = ds_slc.read(1)
            prof = ds_slc.profile.copy()
            trans = ds_slc.transform

        amp = np.abs(slc)

        amp = presum(amp, mlt_az, mlt_rg)
        prof.update(
            {
                "width": amp.shape[1],
                "height": amp.shape[0],
                "transform": trans * Affine.scale(mlt_rg, mlt_az),
                "crs":"EPSG:4326",
            }
        )

        filterwarnings("ignore", category=NotGeoreferencedWarning)
        prof.update({"dtype": amp.dtype.name})
        with rio_open(out_file, "w", **prof) as dst:
            dst.write(amp, 1)

    def _read_burst(self, burst_idx:int=1, remove_invalid:bool=True) -> np.ndarray:
        """
        Reads raster SLC burst.

        Args:
            burst_idx (int, optional): burst index. Defaults to 1.
            remove_invalid (bool, optional): Sets non-valid pixels to NaN. Defaults to True.

        Returns:
            array: Complex raster
        """

        if burst_idx < 1 or burst_idx > self.burst_count:
            raise ValueError(
                f"Invalid burst index (must be between 1 and {self.burst_count})"
            )

        meta = self.meta
        burst_info = meta["product"]["swathTiming"]
        burst_data = burst_info["burstList"]["burst"][burst_idx - 1]

        first_line = (burst_idx - 1) * self.lines_per_burst

        nodataval = np.nan + 1j * np.nan
        arr = read_chunk(self.pth_tiff, first_line, self.lines_per_burst).astype(
            np.complex64
        )

        # not sure about that, should we consider these holes as NaN?
        # leaving as is for now, to avoid propagating NaN in filtering / resampling
        # arr[arr == 0 + 1j * 0] = nodataval

        if remove_invalid:
            fs_str = burst_data["firstValidSample"]["#text"]
            ls_str = burst_data["lastValidSample"]["#text"]
            first_sample_arr = np.array((fs_str).split(" "), dtype="int")
            last_sample_arr = np.array((ls_str).split(" "), dtype="int")
            for i in range(self.lines_per_burst):
                if first_sample_arr[i] > -1:
                    arr[i, : first_sample_arr[i]] = nodataval
                    arr[i, last_sample_arr[i] + 1 :] = nodataval
                else:
                    arr[i] = nodataval
        return arr

    def _calibration_factor(self, burst_idx:int=1, cal_type:str="beta") -> np.ndarray | float:
        """
        Computes calibration factor from the metadata.

        Args:
            burst_idx (int, optional): Burst index. Defaults to 1.
            cal_type (str, optional): Type of calibration. "beta" or "sigma" nought. Defaults to "beta".

        Returns:
            cal_fac: Calibration factor to apply to the raster burst. Array for sigma nought, float for beta nought.
        """
        naz = self.lines_per_burst
        nrg = self.samples_per_burst
        first_line = (burst_idx - 1) * self.lines_per_burst

        str_cols = self.calvec[0]["pixel"]["#text"]
        cols = np.array(list(map(int, str_cols.split(" "))), dtype=int)
        grid_sigma = np.zeros((len(self.calvec), len(cols)), dtype="float64")
        list_lines = []

        self.log.info(f"Compute {cal_type} nought calibration factor.")
        # interpolate values on image grid
        if cal_type == "sigma":
            for i, it in enumerate(self.calvec):
                list_lines.append(int(it["line"]))
                str_sigma = it["sigmaNought"]["#text"]
                line_sigma = list(map(float, str_sigma.split(" ")))
                grid_sigma[i] = line_sigma
            rows = np.array(list_lines, dtype=int)
            grid_arr_rg, grid_arr_az = np.meshgrid(
                np.arange(nrg), np.arange(first_line, first_line + naz)
            )
            interp = RegularGridInterpolator((rows, cols), grid_sigma, method="linear")

            cal_fac = interp((grid_arr_az, grid_arr_rg))
        # for beta, it is a just constant
        elif cal_type == "beta":
            cal_fac = self.beta_nought
        else:
            raise ValueError(
                "Calibration type not recognized (use 'beta' or 'sigma' nought)"
            )
        return cal_fac    
