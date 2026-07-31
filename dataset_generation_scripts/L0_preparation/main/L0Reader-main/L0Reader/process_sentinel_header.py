import numpy as np

def process_sentinel_header(header_data):
    """To extract parameters from header pulse and to generate a dictionary

    This function process the first 68 bytes of the pulse containing the pulse
    header and generates a dictionary with the decoded values. The dictionary
    follows the header structure of the Sentinel-1 documentation.

    Reference document:
        Sentinel-1 SAR Space Packet Protocol Data Unit. S1-IF-ASD-PL-0007
            issue 13. 22.06.2015

    :parameter
        header_data (np.array): array containing the 68 bytes of the header

    :return
        header (dict): Dictionary with the parameters of the header already
                       decoded
        error (int): Error flag

    Header structure:
        The header dictionary will be organised as:

        "PrimaryHeader"
            "PacketVersionNumber"
            "PacketType"
            "SecondaryHeaderFlag"
            "PID"
            "PCAT"
            "SequenceFlags"
            "PacketSequenceCount"
            "PacketDataLength"
        "TimeCode"
            "CoarseTime"
            "FineTime_Code"
            "FineTime"
        "FixedAncillary"
            "SyncMarker"
            "DataTakeID"
            "ECCNumber_Code"
            "ECCNumber"
            "TestMode_Code"
            "TestMode"
            "RxChannelID_Code"
            "RxChannelID"
            "ICID"
        "AncillaryData"
            "WordIndex_Code"
            "DataWord"
        "Counters"
            "SpacePacketCount"
            "PRICount"
        "RadarConfiguration"
            "ErrorFlag"
            "BAQMode_Code"
            "BAQMode"
            "BAQBlockLength_Code"
            "BAQBlockLength"
            "RangeDecimation_Code"
            "DecimationFilterBW"
            "DecimationRatio_L"
            "DecimationRatio_M"
            "DecimationRatio"
            "DecimationFilterLength"
            "SARSWath"
            "SamplingFreq"
            "RxGain_Code"
            "RxGain"
            "TxPulseRampRate_Code"
            "TxPulseRampRate"
            "TxPulseStartFreq_Code"
            "TxPulseStartFreq"
            "TxPulseLength_Code"
            "TxPulseLength"
            "Rank"
            "PRI_Code"
            "PRI"
            "SamplingWindowStartTime_Code"
            "SamplingWindowStartTime"
            "SamplingWindowLength_Code"
            "SamplingWindowLength"
            "TimeSupressed"
            "NumberSamplesAfterDecimation"
            "SAS_SSB"
                "SSBFlag"
                "Pol_Code"
                "Pol"
                "TempComp_Code"
                "TempCompAntennaTA"
                "TempCompAntennaFE"
                "ElevBeamAddr" (if SSBFlag == 0)
                "AzBeamAddr" (if SSBFlag == 0)
                "CalType_Code" (if SSBFlag == 1)
                "CalType" (if SSBFlag == 1)
                "CalBeamAddr" (if SSBFlag == 1)
            "SES_SSB"
                "CalMode_Code"
                "CalMode"
                "TxPulseNumber"
                "SignalType_Code"
                "SignalType"
                "SwapFlag"
                "SwathNumber"
        "RadarSamples"
            "NumberOfQuads"
            "NumberOfSamples"

    Error values:
        Each bit of the Error flag has a specific meaning:
            bit 1 : Primary header invalid
            bit 2 : SyncMarker invalid
            bit 3 : ECC Code invalid
            bit 4 : Test Mode invalid
            bit 5 : Test Mode /= 0 and ECC /= 16
            bit 6 : RxChannel invalid
            bit 7 : Ancillary Data Word Index invalid
            bit 8 : BAQ invalid
            bit 9 : BAQ Block Length invalid
            bit 10 : Range Decimation invalid
            bit 11 : RxGain invalid
            bit 12 : TxPulse Length invalid
            bit 13 : error in the calculation of D parameter (Number of samples)
            bit 14 : CalType invalid
            bit 15 : Signal Type invalid
            bit 16 : Number of Quads invalid
    """

    error = 0
    fref = 37.53472224e6  # Reference frequency [Hz]
    header = {}

    # Primary header
    primary_header_data = np.uint32(header_data[:6])
    packet_version_number = (primary_header_data[0] & 0xE0) >> 5
    packet_type = (primary_header_data[0] & 0x10) >> 4
    secondary_header_flag = (primary_header_data[0] & 0x08) >> 3
    PID = (((primary_header_data[0] & 0x07) << 4)
           + ((primary_header_data[1] & 0xF0) >> 4))
    PCAT = primary_header_data[1] & 0x0F
    sequence_flags = (primary_header_data[2] & 0xC0) >> 6
    packet_sequence_count = (((primary_header_data[2] & 0x3F) << 8)
                             + primary_header_data[3])
    packet_data_length = ((primary_header_data[4] << 8)
                          + primary_header_data[5])

    primary_header = {
        "PacketVersionNumber": packet_version_number,
        "PacketType": packet_type,
        "SecondaryHeaderFlag": secondary_header_flag,
        "PID": PID,
        "PCAT": PCAT,
        "SequenceFlags": sequence_flags,
        "PacketSequenceCount": packet_sequence_count,
        "PacketDataLength": packet_data_length
    }
    header["PrimaryHeader"] = primary_header

    if (packet_version_number != 0
            or packet_type != 0
            or secondary_header_flag != 1
            or PID != 65
            or PCAT != 12
            or sequence_flags != 3):
        error += 2**0

    # Secondary header
    secondary_header_data = np.uint32(header_data[6:68])
    time_code_field = secondary_header_data[:6]
    fixed_ancillary_data_service = secondary_header_data[6:20]
    sub_commutation_ancillary_data_service = secondary_header_data[20:23]
    counters_service = secondary_header_data[23:31]
    radar_configuration_support_service = secondary_header_data[31:59]
    radar_sample_count_service = secondary_header_data[59:62]
    # Time Code Service
    coarse_time = int((time_code_field[0] << 24) + (time_code_field[1] << 16)
                 + (time_code_field[2] << 8) + time_code_field[3])
    fine_time_code = (time_code_field[4] << 8) + time_code_field[5]
    fine_time = (fine_time_code + 0.5) * 2**-16

    time_code = {
        "CoarseTime": coarse_time,
        "FineTime_Code": fine_time_code,
        "FineTime": fine_time
    }
    header["TimeCode"] = time_code

    # Fixed Ancillary Data Service
    sync_marker = ((fixed_ancillary_data_service[0] << 24)
                   + (fixed_ancillary_data_service[1] << 16)
                   + (fixed_ancillary_data_service[2] << 8)
                   + fixed_ancillary_data_service[3])
    if sync_marker != 0x352EF853:
        error += 2**1

    datatake_id = int((fixed_ancillary_data_service[4] << 24)
                      + (fixed_ancillary_data_service[5] << 16)
                      +  (fixed_ancillary_data_service[6] << 8)
                      + fixed_ancillary_data_service[7])

    ecc_number_code = fixed_ancillary_data_service[8]
    if ecc_number_code > 47 or ecc_number_code in [0, 7, 28, 29, 30, 36, 47]:
        error += 2**2

    ecc_number_text = [
        "contingency", "Stripmap 1", "Stripmap 2", "Stripmap 3", "Stripmap 4",
        "Stripmap 5-N", "Stripmap 6", "contingency",
        "Interferometric Wide Swath", "Wave Mode", "Stripmap 5-S",
        "Stripmap 1 w/o interl. Cal", "Stripmap 2 w/o interl. Cal",
        "Stripmap 3 w/o interl. Cal", "Stripmap 4 w/o interl. Cal", "RFC mode",
        "Test node", "Elevation Notch S3", "Azimuth Notch S1",
        "Azimuth Notch S2", "Azimuth Notch S3", "Azimuth Notch S4",
        "Azimuth Notch S5-N", "Azimuth Notch S5-S", "Azimuth Notch S6",
        "Stripmap 5-N w/o interl. Cal", "Stripmap 5-S w/o interl. Cal",
        "Stripmap 6 w/o interl. Cal", "contingency", "contingency",
        "contingency", "Elevation Notch S3 w/o interl. Cal",
        "Extra Wide Swath", "Azimuth Notch S1 w/o interl. Cal",
        "Azimuth Notch S3 w/o interl. Cal", "Azimuth Notch S6 w/o interl. Cal",
        "contingency", "Noise Characterisation S1",
        "Noise Characterisation S2", "Noise Characterisation S3",
        "Noise Characterisation S4", "Noise Characterisation S5-N",
        "Noise Characterisation S5-S", "Noise Characterisation S6",
        "Noise Characterisation EWS", "Noise Characterisation IWS",
        "Noise Characterisation Wave", "contingency"
    ]
    ecc = ecc_number_text[min(ecc_number_code,47)]

    test_mode_code = (fixed_ancillary_data_service[9] & 0x70) >> 4
    test_mode = {
        0: "Default", 4: "Contingency 100", 5: "Contingency 101",
        6: "Test Mode Oper", 7: "Test Mode Bypass"
    }.get(test_mode_code, "Invalid")
    if test_mode_code not in [0, 4, 5, 6, 7]:
        error += 2**3

    if test_mode_code != 0 and ecc_number_code != 16:
        error += 2**4

    rx_channel_code = fixed_ancillary_data_service[9] & 0x0F
    rx_channel = {0: "V", 1: "H"}.get(rx_channel_code, "Invalid")
    if rx_channel_code not in [0, 1]:
        error += 2**5

    ICID = int((fixed_ancillary_data_service[10] << 24)
               + (fixed_ancillary_data_service[11] << 16)
               + (fixed_ancillary_data_service[12] << 8)
               + fixed_ancillary_data_service[13])

    fixed_ancillary = {
        "SyncMarker": sync_marker,
        "DataTakeID": datatake_id,
        "ECCNumber_Code": ecc_number_code,
        "ECCNumber": ecc,
        "TestMode_Code": test_mode_code,
        "TestMode": test_mode,
        "RxChannelID_Code": rx_channel_code,
        "RxChannelID": rx_channel,
        "ICID": ICID
    }
    header["FixedAncillary"] = fixed_ancillary

    # Sub-Commutation Ancillary Data Service
    word_index_code = sub_commutation_ancillary_data_service[0]
    if word_index_code > 64:
        error += 2**6

    data_word = int((sub_commutation_ancillary_data_service[1] << 8)
                    + sub_commutation_ancillary_data_service[2])

    ancillary_data = {
        "WordIndex_Code": word_index_code,
        "DataWord": data_word
    }
    header["AncillaryData"] = ancillary_data

    # Counters Service
    space_packet_count = int(
        (counters_service[0] << 24)
        + (counters_service[1] << 16)
        + (counters_service[2] << 8)
        + counters_service[3]
    )
    pri_count = int(
        (counters_service[4] << 24)
        + (counters_service[5] << 16)
        + (counters_service[6] << 8)
        + counters_service[7]
    )

    counters = {
        "SpacePacketCount": space_packet_count,
        "PRICount": pri_count
    }
    header["Counters"] = counters
    
     # Radar Configuration Support Service
    error_flag = (radar_configuration_support_service[0] & 0x80) >> 7
    baq_code = radar_configuration_support_service[0] & 0x1F

    baq_mode_map = {
        0: "Bypass Mode",
        3: "BAQ 3-bit Mode",
        4: "BAQ 4-bit Mode",
        5: "BAQ 5-bit Mode",
        12: "FDBAQ Mode 0",
        13: "FDBAQ Mode 1",
        14: "FDBAQ Mode 2"
    }
    baq_mod = baq_mode_map.get(baq_code, "Invalid BAQ Mode")
    if baq_mod == "Invalid BAQ Mode":
        error += 2**7

    # BAQ Block Length
    baq_block_length_code = radar_configuration_support_service[1]
    baq_block_length = int(8 * (baq_block_length_code + 1))
    if baq_block_length != 256:
        error += 2**8

    # Range Decimation
    range_decimation_code = radar_configuration_support_service[3]
    if (range_decimation_code > 11) or (range_decimation_code == 2):
        error += 2**9

    # Decimation parameters
    range_decimation_params = {
        0: (100e6, 3, 4, 3/4, 28, 'Full Bandwidth'),
        1: (87.71e6, 2, 3, 2/3, 28, 'S1,WV1'),
        3: (74.25e6, 5, 9, 5/9, 32, 'S2'),
        4: (59.44e6, 4, 9, 4/9, 40, 'S3'),
        5: (50.62e6, 3, 8, 3/8, 48, 'S4'),
        6: (44.89e6, 1, 3, 1/3, 52, 'S5'),
        7: (22.2e6, 1, 6, 1/6, 92, 'EW1'),
        8: (56.59e6, 3, 7, 3/7, 36, 'IW1'),
        9: (42.86e6, 5, 16, 5/16, 68, 'S6,IW3'),
        10: (15.1e6, 3, 26, 3/26, 120, 'EW2,EW3,EW4,EW5'),
        11: (48.35e6, 4, 11, 4/11, 44, 'IW2,WV2')
    }

    if range_decimation_code in range_decimation_params:
        (dec_filter_bw, dec_ratio_L, dec_ratio_M, dec_ratio, filter_length,
         swath) = range_decimation_params[range_decimation_code]
    else:
        dec_filter_bw = dec_ratio_L = dec_ratio_M = dec_ratio = None
        filter_length = swath = None

    sampling_freq = dec_ratio * 4 * fref

    # Rx Gain
    rx_gain_code = radar_configuration_support_service[4]
    if rx_gain_code > 63:
        error += 2**10
    rx_gain = -0.5 * rx_gain_code

    # Tx Pulse Ramp Rate
    tx_pulse_ramp_rate_code = ((radar_configuration_support_service[5] << 8)
                               | radar_configuration_support_service[6])
    tx_pulse_ramp_rate_sign = \
        not ((radar_configuration_support_service[5] >> 7) & 0x1)
    tx_pulse_ramp_rate_magnitude = (
        int(((radar_configuration_support_service[5] & 0x7F) << 8)
            + radar_configuration_support_service[6]))
    tx_pulse_ramp_rate = (((-1) ** tx_pulse_ramp_rate_sign)
                          * tx_pulse_ramp_rate_magnitude
                          * (fref**2)
                          / (2**21))

    # Tx Pulse Start Frequency
    tx_pulse_start_freq_code = ((radar_configuration_support_service[7] << 8)
                                | radar_configuration_support_service[8])
    tx_pulse_start_freq_sign = \
        not ((radar_configuration_support_service[7] >> 7) & 0x1)
    tx_pulse_start_freq_magnitude = (
        int(((radar_configuration_support_service[7] & 0x7F) << 8)
            + radar_configuration_support_service[8]))
    tx_pulse_start_freq = ((tx_pulse_ramp_rate / (4 * fref))
                           + ((-1) ** tx_pulse_start_freq_sign)
                               * tx_pulse_start_freq_magnitude
                               * fref
                               / (2**14))

    # Tx Pulse Length
    tx_pulse_length_code = ((radar_configuration_support_service[9] << 16)
                            | (radar_configuration_support_service[10] << 8)
                            | radar_configuration_support_service[11])
    if not (128 <= tx_pulse_length_code <= 4223):
        error += 2**11
    tx_pulse_length = tx_pulse_length_code / fref

    # PRI
    rank = radar_configuration_support_service[12] & 0x1F
    pri_code = ((radar_configuration_support_service[13] << 16)
                | (radar_configuration_support_service[14] << 8)
                | radar_configuration_support_service[15])
    pri = pri_code / fref

    # Sampling Window Start Time
    sampling_window_start_time_code = (
            (radar_configuration_support_service[16] << 16)
            + (radar_configuration_support_service[17] << 8)
            + radar_configuration_support_service[18])
    sampling_window_start_time = sampling_window_start_time_code / fref

    time_supressed = 320 / (8 * fref)

    # Sampling Window Length
    sampling_window_length_code = (
            (radar_configuration_support_service[19] << 16) +
            (radar_configuration_support_service[20] << 8) +
            radar_configuration_support_service[21])
    sampling_window_length = sampling_window_length_code / fref

    filter_output_offset = 80 + filter_length / 4
    B = 2 * sampling_window_length_code - filter_output_offset - 17
    C = int(B - dec_ratio_M * np.floor(B / dec_ratio_M))

    # DValues matrix
    D_values = np.array([
        [1, 1, np.nan, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, np.nan, 1, 1, 1, 0, 0, 1, 0, 0, 1],
        [2, 2, np.nan, 2, 1, 1, 1, 0, 1, 1, 0, 1],
        [3, np.nan, np.nan, 2, 2, 1, np.nan, 0, 2, 1, 0, 1],
        [np.nan, np.nan, np.nan, 3, 2, 2, np.nan, 0, 2, 1, 0, 2],
        [np.nan, np.nan, np.nan, 3, 3, 2, np.nan, 1, 3, 2, 0, 2],
        [np.nan, np.nan, np.nan, 4, 3, 3, np.nan, np.nan, 3, 2, 0, 3],
        [np.nan, np.nan, np.nan, 4, 4, 3, np.nan, np.nan, np.nan, 2, 1, 3],
        [np.nan, np.nan, np.nan, 5, 4, np.nan, np.nan, np.nan, np.nan, 2, 1, 3],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
         3, 1, 4],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
         3, 1, 4],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
         3, 1, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
         4, 1, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
         4, 1, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
         4, 1, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
         5, 1, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
         np.nan, 2, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
         np.nan, 2, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
         np.nan, 2, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
         np.nan, 2, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
         np.nan, 2, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
         np.nan, 2, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
         np.nan, 2, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
         np.nan, 3, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
         np.nan, 3, np.nan]
    ])
    
    # Calculating D value
    D = D_values[C, range_decimation_code]
    if np.isnan(D):
        error += 2**12

    number_samples_after_dec = (
            2 * (dec_ratio_L * np.floor(B / dec_ratio_M) + D + 1))

    # SAS SSB Message
    sas_ssm_message = radar_configuration_support_service[22:25]
    ssb_flag = (sas_ssm_message[0] >> 7) & 0x1

    pol_code = (sas_ssm_message[0] & 0x70) >> 4
    pol = None
    if pol_code == 0:
        pol = 'H-'
    elif pol_code == 1:
        pol = 'HH'
    elif pol_code == 2:
        pol = 'HV'
    elif pol_code == 3:
        pol = 'HV/HH'
    elif pol_code == 4:
        pol = 'V-'
    elif pol_code == 5:
        pol = 'VH'
    elif pol_code == 6:
        pol = 'VV'
    elif pol_code == 7:
        pol = 'VV/VH'

    temp_comp_code = (sas_ssm_message[0] & 0x0C) >> 2
    temp_comp_antenna_fe = None
    temp_comp_antenna_ta = None
    if temp_comp_code == 0:
        temp_comp_antenna_fe = 'OFF'
        temp_comp_antenna_ta = 'OFF'
    elif temp_comp_code == 1:
        temp_comp_antenna_fe = 'ON'
        temp_comp_antenna_ta = 'OFF'
    elif temp_comp_code == 2:
        temp_comp_antenna_fe = 'OFF'
        temp_comp_antenna_ta = 'ON'
    elif temp_comp_code == 3:
        temp_comp_antenna_fe = 'ON'
        temp_comp_antenna_ta = 'ON'

    az_cal_beam_addr = (sas_ssm_message[1] & 0x03) << 8 | sas_ssm_message[2]

    if ssb_flag == 0:  # Imaging/noise
        elev_beam_addr = (sas_ssm_message[1] & 0xF0) >> 4
        sas_ssb = {
            'SSBFlag': ssb_flag,
            'Pol_Code': pol_code,
            'Pol': pol,
            'TempComp_Code': temp_comp_code,
            'TempCompAntennaTA': temp_comp_antenna_ta,
            'TempCompAntennaFE': temp_comp_antenna_fe,
            'ElevBeamAddr': elev_beam_addr,
            'AzBeamAddr': az_cal_beam_addr
        }
    else:
        sas_test_mode = (sas_ssm_message[1] >> 7) & 0x1
        cal_type_code = (sas_ssm_message[1] & 0x70) >> 4
        cal_type = None
        if cal_type_code == 0:
            cal_type = 'TxCal'
        elif cal_type_code == 1:
            cal_type = 'RxCal'
        elif cal_type_code == 2:
            cal_type = 'EPDNCal'
        elif cal_type_code == 3:
            cal_type = 'TACal'
        elif cal_type_code == 4:
            cal_type = 'APDNCal'
        elif cal_type_code in {5, 6}:
            cal_type = 'Invalid CalType'
            error += 2**13
        elif cal_type_code == 7:
            cal_type = 'TxCal Isolation at TxPol H'

        sas_ssb = {
            'SSBFlag': ssb_flag,
            'Pol_Code': pol_code,
            'Pol': pol,
            'TempComp_Code': temp_comp_code,
            'TempCompAntennaTA': temp_comp_antenna_ta,
            'TempCompAntennaFE': temp_comp_antenna_fe,
            'SASTestMode': sas_test_mode,
            'CalType_Code': cal_type_code,
            'CalType': cal_type,
            'CalBeamAddr': az_cal_beam_addr
        }

    ses_ssb_message = radar_configuration_support_service[25:28]

    # CalMode decoding
    cal_mode_code = (ses_ssb_message[0] & 0xC0) >> 6
    cal_mode = {
        0: 'Interleaved Internal Calibration based on PCC2 sequence',
        1: 'Internal Calibration in Preamble/Postamble based on PCC2 sequence',
        2: 'Phase Coded Characterisation based on PCC32 sequence',
        3: 'Phase Coded Characterisation based on RF672 sequence'
    }.get(cal_mode_code, 'Unknown Mode')

    # TxPulseNumber and SignalType decoding
    tx_pulse_number = ses_ssb_message[0] & 0x1F
    signal_type_code = (ses_ssb_message[1] & 0xF0) >> 4
    signal_type = {
        0: 'Echo',
        1: 'Noise',
        8: 'TxCal',
        9: 'RxCal',
        10: 'EPDNCal',
        11: 'TACal',
        12: 'APDNCal',
        15: 'TxCal Isolation at TxPol H'
    }.get(signal_type_code, 'Unknown Signal Type')

    # error check for invalid signal types
    if signal_type_code in {2, 3, 4, 5, 6, 7, 13, 14}:
        error += 2**14

    # SwapFlag and SwathNumber
    swap_flag = (ses_ssb_message[1] >> 0) & 0x1
    swath_number = ses_ssb_message[2]

    # Store in SES_SSB dictionary
    ses_ssb = {
        'CalMode_Code': cal_mode_code,
        'CalMode': cal_mode,
        'TxPulseNumber': tx_pulse_number,
        'SignalType_Code': signal_type_code,
        'SignalType': signal_type,
        'SwapFlag': swap_flag,
        'SwathNumber': swath_number
    }

    # Store in RadarConfiguration dictionary
    radar_configuration = {
        'errorFlag': error_flag,
        'BAQMode_Code': baq_code,
        'BAQMode': baq_mod,
        'BAQBlockLength_Code': baq_block_length_code,
        'BAQBlockLength': baq_block_length,
        'RangeDecimation_Code': range_decimation_code,
        'DecimationFilterBW': dec_filter_bw,
        'DecimationRatio_L': dec_ratio_L,
        'DecimationRatio_M': dec_ratio_M,
        'DecimationRatio': dec_ratio,
        'DecimationFilterLength': filter_length,
        'SARSWath': swath,
        'SamplingFreq': sampling_freq,
        'RxGain_Code': rx_gain_code,
        'RxGain': rx_gain,
        'TxPulseRampRate_Code': tx_pulse_ramp_rate_code,
        'TxPulseRampRate': tx_pulse_ramp_rate,
        'TxPulseStartFreq_Code': tx_pulse_start_freq_code,
        'TxPulseStartFreq': tx_pulse_start_freq,
        'TxPulseLength_Code': tx_pulse_length_code,
        'TxPulseLength': tx_pulse_length,
        'Rank': rank,
        'PRI_Code': pri_code,
        'PRI': pri,
        'SamplingWindowStartTime_Code': sampling_window_start_time_code,
        'SamplingWindowStartTime': sampling_window_start_time,
        'SamplingWindowLength_Code': sampling_window_length_code,
        'SamplingWindowLength': sampling_window_length,
        'TimeSupressed': time_supressed,
        'NumberSamplesAfterDecimation': number_samples_after_dec,
        'SAS_SSB': sas_ssb,
        'SES_SSB': ses_ssb
    }

    header['RadarConfiguration'] =  radar_configuration

    # Radar Sample Count Service
    number_of_quads = int((radar_sample_count_service[0] << 8)
                          + radar_sample_count_service[1])
    if number_of_quads > 52378:
        error += 2**15
    number_of_samples = 2 * number_of_quads
    radar_samples = {
        'NumberOfQuads': number_of_quads,
        'NumberOfSamples': number_of_samples
    }
    header['RadarSamples'] =  radar_samples


    return header, error
