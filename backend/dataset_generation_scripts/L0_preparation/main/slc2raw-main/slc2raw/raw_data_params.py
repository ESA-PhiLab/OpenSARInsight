import numpy as np

from math import radians, sqrt
from astropy.time import Time
import xml.etree.ElementTree as ET

def raw_data_line(line, annotation_file):
    """



        :param:
            line:
            annotation_file:

        :return:
            raw_data_line_val:
            ra_ref_func_length:
            az_ref_func_length:
            raw_line_within_burst:
            burst_sensing_time_gps:
            azimuth_time_mid_burst_gps:
            sensing_time_mid_burst_gps:
            burst_number:

    """

    #--------------------------------------------------------------------------
    # Constants and workspace
    #--------------------------------------------------------------------------
    C = 299792458  # speed of light [m/s]

    #--------------------------------------------------------------------------
    # Get Annotation file Parameters
    #--------------------------------------------------------------------------
    (k_psi, fc, frs, azimuth_fm_rate_tbl, dc_estimate_tbl, orbit_tbl,
     tau0, dt, ns, nl, _, prf) = _read_s1_annotation_file(annotation_file)

    pri = 1 / prf

    #--------------------------------------------------------------------------
    # Identify Burst
    #--------------------------------------------------------------------------
    burst_number = line // nl  # zero-indexed
    xdoc = ET.parse(annotation_file)
    root = xdoc.getroot()

    bursts = root.findall('.//burst')
    burst_node = bursts[burst_number]
    burst_azimuth_time_utc = burst_node.find('azimuthTime').text
    burst_sensing_time_utc = burst_node.find('sensingTime').text

    burst_azimuth_time_gps = _utc_to_gps_seconds(burst_azimuth_time_utc)
    burst_sensing_time_gps = _utc_to_gps_seconds(burst_sensing_time_utc)

    #--------------------------------------------------------------------------
    # Identify Azimuth time for input line
    #--------------------------------------------------------------------------
    line_within_burst = line - burst_number * nl  # zero-indexed

    # MidBurst Line within burst
    mid_burst_line = (nl - 1) / 2
    # Line number of complete image
    mid_burst_image_line = mid_burst_line + burst_number * nl

    azimuth_time_mid_burst_gps = burst_azimuth_time_gps + mid_burst_line * dt
    sensing_time_mid_burst_gps = burst_sensing_time_gps + pri * mid_burst_line

    #--------------------------------------------------------------------------
    # Calculate Orbit Velocity at mid-burst
    #--------------------------------------------------------------------------
    orbit_gps_time = np.array([_utc_to_gps_seconds(t) for t in orbit_tbl[:, 0]])
    v_sat_tbl = orbit_tbl[:, 1].astype(float)
    vsat = np.interp(sensing_time_mid_burst_gps, orbit_gps_time, v_sat_tbl)

    # Calculate ks (Doppler Centroid rate caused by scanning of the antenna)
    ks = 2 * vsat / C * fc * k_psi

    #--------------------------------------------------------------------------
    # ka (Doppler FM rate)
    #--------------------------------------------------------------------------

    azimuth_fm_polynom = azimuth_fm_rate_tbl[burst_number]
    coefficients = [float(c) for c in
                    azimuth_fm_polynom[2][::-1]]  # Reverse for np.polyval
    tau = tau0 + (ns - 1) / 2 / frs
    ka = np.polyval(coefficients, tau - tau0)

    #--------------------------------------------------------------------------
    # Doppler Centroid at tMidBurst, MidSwath
    #--------------------------------------------------------------------------
    fdc_polynom = dc_estimate_tbl[burst_number]
    fdc_coefficients = [float(c) for c in fdc_polynom[2][::-1]]
    fdc =  np.polyval(fdc_coefficients, tau - tau0)

    fdc_time = -fdc / ka

    #--------------------------------------------------------------------------
    # Image-> Raw Time Conversion Factor
    #--------------------------------------------------------------------------
    dt_zd = np.arange(nl) * dt - (nl - 1) * dt / 2
    dt_s = ka / (ka - ks) * dt_zd - fdc_time
    rel_raw_time = dt_s[int(line_within_burst)]
    abs_raw_time = rel_raw_time + azimuth_time_mid_burst_gps

    raw_line_within_burst = (abs_raw_time - burst_sensing_time_gps) * prf + 1
    relative_az_line = rel_raw_time * prf
    raw_data_line_val = relative_az_line + mid_burst_image_line

    #--------------------------------------------------------------------------
    # Get Reference Function Lengths
    #--------------------------------------------------------------------------
    txpl = float(root.find('.//txPulseLength').text)
    fdec = float(root.find('.//samplingFrequencyAfterDecimation').text)
    azi_bw = float(root.findall('.//processingBandwidth')[1].text)

    ra_ref_func_length = txpl * fdec
    az_ref_func_length = (azi_bw / abs(ka) * prf)

    return (raw_data_line_val,
            ra_ref_func_length,
            az_ref_func_length,
            raw_line_within_burst,
            burst_sensing_time_gps,
            azimuth_time_mid_burst_gps,
            sensing_time_mid_burst_gps,
            burst_number
            )

def raw_data_pixel(slc_pixel, annotation_file):
    """


        :param:
            slc_pixel:
            annotation_file:

        :return:
            raw_pixel:

    """

    xdoc = ET.parse(annotation_file)
    root = xdoc.getroot()

    slc_range_sampling_rate = float(root.find('.//rangeSamplingRate').text)
    raw_range_sampling_rate = float(root.find('.//samplingFrequencyAfterDecimation').text)

    raw_pixel = slc_pixel * raw_range_sampling_rate / slc_range_sampling_rate
    return raw_pixel

def _read_s1_annotation_file(annotation_file):
    """Read Sentinel annotation file

        Function for reading a number of parameters from annotationFile.

        :param:
            annotation_file:

        :return:
            k_psi:                azimuth steering rate [rads/s]
            fc:                   carrier freq [Hz]
            frs:                  range sampling rate [Hz]
            azimuth_fm_rate_tbl:
            dc_estimate_tbl:
            orbit_tbl:
            tau0:                 slant range time [s]
            dt:                   azimuth time interval of SLC image [s]
            ns:                   number of samples within a SLC sub-swath
            nl:                   number of lines within a burst (same for all burst)
            burst_tbl:
            prf:                  instrument pulse repetition frequency [Hz]

    """


    # Read a number of parameters from annotationFile
    xdoc = ET.parse(annotation_file)
    root = xdoc.getroot()

    general_annotation = root.find('.//generalAnnotation')
    doppler_centroid = root.find('.//dopplerCentroid')
    image_annotation = root.find('.//imageAnnotation')
    swath_timing = root.find('.//swathTiming')

    # Product Information
    product_info = general_annotation.find('productInformation')
    k_psi = radians(float(product_info.find('azimuthSteeringRate').text))
    fc = float(product_info.find('radarFrequency').text)
    frs = float(product_info.find('rangeSamplingRate').text)

    # Downlink Information
    prf = float(general_annotation.find('.//prf').text)
    pri = 1 / prf

    # Product Information
    azimuth_fm_rate_list = general_annotation.findall('.//azimuthFmRate')
    azimuth_fm_rate_tbl = []
    for item in azimuth_fm_rate_list:
        az_time = item.find('azimuthTime').text
        t0 = float(item.find('t0').text)
        poly = list(map(float, item.find('azimuthFmRatePolynomial').
                        text.strip().split()))
        azimuth_fm_rate_tbl.append([az_time, t0, poly])

    # Doppler Centroid
    dc_estimate_list = doppler_centroid.findall('.//dcEstimate')
    dc_estimate_tbl = []
    for item in dc_estimate_list:
        az_time = item.find('azimuthTime').text
        t0 = float(item.find('t0').text)
        poly = list(map(float, item.find('dataDcPolynomial').
                        text.strip().split()))
        dc_estimate_tbl.append([az_time, t0, poly])

    # Orbit
    orbit_tbl = []
    for orbit in root.findall('.//orbitList/orbit'):
        time = orbit.find('time').text
        vx = float(orbit.find('velocity/x').text)
        vy = float(orbit.find('velocity/y').text)
        vz = float(orbit.find('velocity/z').text)
        v = sqrt(vx**2 + vy**2 + vz**2)
        orbit_tbl.append([time, v])
    orbit_tbl = np.array(orbit_tbl)

    # Image Annotation
    tau0 = float(image_annotation.find('.//slantRangeTime').text)
    dt = float(image_annotation.find('.//azimuthTimeInterval').text)
    ns = int(image_annotation.find('.//numberOfSamples').text)

    # Swath Timing
    nl = int(swath_timing.find('.//linesPerBurst').text)
    burst_tbl = []
    for burst in swath_timing.findall('burst'):
        az_time = burst.find('azimuthTime').text
        sens_time = burst.find('sensingTime').text
        burst_tbl.append([az_time, sens_time])

    return (k_psi,
            fc,
            frs,
            azimuth_fm_rate_tbl,
            dc_estimate_tbl,
            orbit_tbl,
            tau0,
            dt,
            ns,
            nl,
            burst_tbl,
            prf
            )

def _utc_to_gps_seconds(utc_time_str):
    """UTC time to GPS time

        Function to convert UTC time to GPS time

        :param:
            utc_time_str: UTC time in ISO format, e.g. '2025-03-18T06:19:42.000Z'

        :return:
            t.gps:        GPS time in seconds

    """

    t = Time(utc_time_str, format='isot', scale='utc')
    return t.gps  # seconds since 1980-01-06