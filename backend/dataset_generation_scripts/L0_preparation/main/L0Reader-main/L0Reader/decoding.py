import numpy as np

# Sample Reconstruction Tables (Section 5.2 of S1-IF-ASD-PL-0007 Issue 13)

# Simple Reconstruction Parameter A
A_TABLE = {
    0: [3, 3, 3.12, 3.55],
    1: [7, 7, 7, 7.17, 7.4, 7.76],
    2: [15, 15, 15, 15, 15, 15, 15.44, 15.56, 16.11, 16.38, 16.65],
}

# Simple Reconstruction Parameter B
B_TABLE = {0: [3, 3, 3.16, 3.53],
           1: [4, 4, 4.08, 4.37],
           2: [6, 6, 6, 6.15, 6.5, 6.88],
           3: [9, 9, 9, 9, 9.36, 9.5, 10.1],
           4: [15, 15, 15, 15, 15, 15, 15.22, 15.5, 16.05],
           }

# Normalised Reconstruction Levels (NRL) for FDBAQ mode
NRL_FDBAQ_TABLE = {
    0: [0.3637, 1.0915, 1.8208, 2.6406],
    1: [0.3042, 0.9127, 1.5216, 2.1313, 2.8426],
    2: [0.2305, 0.6916, 1.1528, 1.6140, 2.0754, 2.5369, 3.1191],
    3: [0.1702, 0.5107, 0.8511, 1.1916, 1.5321, 1.8726, 2.2131, 2.5536, 2.8942,
        3.3744],
    4: [0.1130, 0.3389, 0.5649, 0.7908, 1.0167, 1.2428, 1.4687, 1.6947, 1.9206,
        2.1466, 2.3725, 2.5985, 2.8244, 3.0504, 3.2764, 3.6623],
}

# Normalised Reconstruction Levels (NRL) for BAQ mode
NRL_BAQ_TABLE = {
    0: [0.249, 0.7681, 1.3655, 2.1864],
    1: [0.129, 0.39, 0.6601, 0.9471, 1.2623, 1.6261, 2.0793, 2.7467],
    2: [0.066, 0.1985, 0.3320, 0.4677, 0.6061, 0.7487, 0.8964, 1.051, 1.2143,
        1.3896, 1.58, 1.7914, 2.0329, 2.3234, 2.6971, 3.2692],
}

# Sigma Factors
SF = [
    0.00, 0.63, 1.25, 1.88, 2.51, 3.13, 3.76, 4.39, 5.01, 5.64, 6.27, 6.89,
    7.52, 8.15, 8.77, 9.40,
    10.03, 10.65, 11.28, 11.91, 12.53, 13.16, 13.79, 14.41, 15.04, 15.67,
    16.29, 16.92, 17.55, 18.17,
    18.80, 19.43, 20.05, 20.68, 21.31, 21.93, 22.56, 23.19, 23.81, 24.44,
    25.07, 25.69, 26.32, 26.95,
    27.57, 28.20, 28.83, 29.45, 30.08, 30.71, 31.33, 31.96, 32.59, 33.21,
    33.84, 34.47, 35.09, 35.72,
    36.35, 36.97, 37.60, 38.23, 38.85, 39.48, 40.11, 40.73, 41.36, 41.99,
    42.61, 43.24, 43.87, 44.49,
    45.12, 45.75, 46.37, 47.00, 47.63, 48.25, 48.88, 49.51, 50.13, 50.76,
    51.39, 52.01, 52.64, 53.27,
    53.89, 54.52, 55.15, 55.77, 56.40, 57.03, 57.65, 58.28, 58.91, 59.53,
    60.16, 60.79, 61.41, 62.04,
    62.98, 64.24, 65.49, 66.74, 68.00, 69.25, 70.50, 71.76, 73.01, 74.26,
    75.52, 76.77, 78.02, 79.28,
    80.53, 81.78, 83.04, 84.29, 85.54, 86.80, 88.05, 89.30, 90.56, 91.81,
    93.06, 94.32, 95.57, 96.82,
    98.08, 99.33, 100.58, 101.84, 103.09, 104.34, 105.60, 106.85, 108.10,
    109.35, 110.61, 111.86,
    113.11, 114.37, 115.62, 116.87, 118.13, 119.38, 120.63, 121.89, 123.14,
    124.39, 125.65, 126.90,
    128.15, 129.41, 130.66, 131.91, 133.17, 134.42, 135.67, 136.93, 138.18,
    139.43, 140.69, 141.94,
    143.19, 144.45, 145.70, 146.95, 148.21, 149.46, 150.71, 151.97, 153.22,
    154.47, 155.73, 156.98,
    158.23, 159.49, 160.74, 161.99, 163.25, 164.50, 165.75, 167.01, 168.26,
    169.51, 170.77, 172.02,
    173.27, 174.53, 175.78, 177.03, 178.29, 179.54, 180.79, 182.05, 183.30,
    184.55, 185.81, 187.06,
    188.31, 189.57, 190.82, 192.07, 193.33, 194.58, 195.83, 197.09, 198.34,
    199.59, 200.85, 202.10,
    203.35, 204.61, 205.86, 207.11, 208.37, 209.62, 210.87, 212.13, 213.38,
    214.63, 215.89, 217.14,
    218.39, 219.65, 220.90, 222.15, 223.41, 224.66, 225.91, 227.17, 228.42,
    229.67, 230.93, 232.18,
    233.43, 234.69, 235.94, 237.19, 238.45, 239.70, 240.95, 242.21, 243.46,
    244.71, 245.97, 247.22,
    248.47, 249.73, 250.98, 252.23, 253.49, 254.74, 255.99, 255.99
]


def _decode_bypass_mode(data, header):
    """Decode the data applying the bypass decoding method.

    This function applies the bypass decode method as defined in the Sentinel 1
    SAR Space Packet Protocol Data Unit.
    
    :parameter
        data (np.array): Array of uint8 containing the raw data.
        header (dict):   Header containing pulse metadata information 

    :return
        error (int): Error code 
            -1 if the number of 16-bit words in the data is not sufficient.
            0 if correct
        decoded_data (np.array): Decoded complex data.
    """

    error = 0

    NQ = header['RadarSamples']['NumberOfQuads']
    NW = int(np.ceil(NQ * 10 / 16))  # Number of 16-bit words

    # Check if data size is sufficient
    if len(data) < 4 * NW * 2:
        return -1, None

    # Extract binary codes for IE, IO, QE, and QO
    ie_codes = np.unpackbits(data[:2 * NW], bitorder='big').reshape(-1, 10)
    io_codes = np.unpackbits(data[2 * NW:4 * NW],
                             bitorder='big').reshape(-1, 10)
    qe_codes = np.unpackbits(data[4 * NW:6 * NW],
                             bitorder='big').reshape(-1, 10)
    qo_codes = np.unpackbits(data[6 * NW:8 * NW],
                             bitorder='big').reshape(-1, 10)

    # Decode data
    decoded_data = np.zeros(NQ * 2, dtype=complex)
    for i in range(NQ):
        # IE and IO data
        ie_data = ((-1) ** int(ie_codes[i, 0])
                   * int(''.join(map(str, ie_codes[i, 1:])), 2))
        io_data = ((-1) ** int(io_codes[i, 0])
                   * int(''.join(map(str, io_codes[i, 1:])), 2))

        # QE and QO data
        qe_data = ((-1) ** int(qe_codes[i, 0])
                   * int(''.join(map(str, qe_codes[i, 1:])), 2))
        qo_data = ((-1) ** int(qo_codes[i, 0])
                   * int(''.join(map(str, qo_codes[i, 1:])), 2))

        # Complex decoding
        decoded_data[2 * i] = ie_data + 1j * qe_data
        decoded_data[2 * i + 1] = io_data + 1j * qo_data

    return error, decoded_data, []


def _decode_baq_mode(data, header):
    """Decode the data applying the BAQ decoding method.

    This function applies the BAQ decode method as defined in the Sentinel 1
    SAR Space Packet Protocol Data Unit.

    :parameter
        data (np.array): Array of uint8 containing the raw data.
        header (dict):   Header containing pulse metadata information 

    :return
        error (int): Error code 
            -1 if the number of 16-bit words in the data is not sufficient.
            0 if correct
        decoded_data (np.array): Decoded complex data.
        decoded_params (dict): THIDX of each block
    """

    error = 0
    decoded_params = {}

    # Extracting header parameters
    baq_mode = int(header['RadarConfiguration']['BAQMode_Code'])
    baq_block_length = header['RadarConfiguration']['BAQBlockLength']
    NQ = header['RadarSamples']['NumberOfQuads']

    NB = int(np.ceil(2 * NQ / baq_block_length))  # Number of BAQ blocks
    NW = int(np.ceil(baq_mode * NQ / 16))  # IE, IO, QO words
    NWQE = int(np.ceil((baq_mode * NQ + 8 * NB) / 16))  # QE words

    last_block_size = NQ - baq_block_length // 2 * (NB - 1)

    # Select parameters based on baq_mode
    if baq_mode == 3:
        threshold_limit, M_code_limit = 3, 3
        A, NRL = A_TABLE[0], NRL_BAQ_TABLE[0]
    elif baq_mode == 4:
        threshold_limit, M_code_limit = 5, 7
        A, NRL = A_TABLE[1], NRL_BAQ_TABLE[1]
    elif baq_mode == 5:
        threshold_limit, M_code_limit = 10, 15
        A, NRL = A_TABLE[2], NRL_BAQ_TABLE[2]
    else:
        return -2, None, None

    # Check data length
    if len(data) // 2 < (3 * NW + NWQE):
        return -1, None, None

    ie_bits = (np.unpackbits(data[:2 * NW],
                             bitorder='big'))[:baq_mode * NQ]
    io_bits = (np.unpackbits(data[2 * NW:4 * NW],
                             bitorder='big'))[:baq_mode * NQ]
    qe_bits = (np.unpackbits(data[4 * NW:4 * NW + 2 * NWQE],
                             bitorder='big'))[:baq_mode * NQ + NB * 8]
    qo_bits = (np.unpackbits(data[4 * NW + 2 * NWQE:6 * NW + 2 * NWQE],
                             bitorder='big'))[:baq_mode * NQ]

    # Initialize decoded data and parameters
    decoded_data = np.zeros(NQ * 2, dtype=complex)
    THIDX = np.zeros(NB, dtype=int)

    power_factor = 2 ** np.arange(baq_mode - 2, -1, -1)

    for i in range(NB):
        data_per_block = int(baq_block_length // 2) if i < NB - 1 \
            else int(last_block_size)

        pos_ini = i * baq_mode * baq_block_length // 2
        pos_fin = min(NQ * baq_mode, pos_ini + data_per_block * baq_mode)
        pos_ini_qe = i * (baq_mode * baq_block_length // 2 + 8)
        pos_fin_qe = min(NQ * baq_mode + NB * 8,
                         pos_ini_qe + data_per_block * baq_mode + 8)

        ie_bits_reshape = ie_bits[pos_ini:
                                  pos_fin].reshape(data_per_block, baq_mode)
        io_bits_reshape = io_bits[pos_ini:
                                  pos_fin].reshape(data_per_block, baq_mode)
        THIDX[i] = np.packbits(qe_bits[pos_ini_qe:pos_ini_qe + 8],
                               bitorder='big')
        qe_bits_reshape = qe_bits[pos_ini_qe + 8:
                                  pos_fin_qe].reshape(data_per_block, baq_mode)
        qo_bits_reshape = qo_bits[pos_ini:
                                  pos_fin].reshape(data_per_block, baq_mode)

        ie_sign = (-1) ** ie_bits_reshape[:, 0].astype(np.int8)
        io_sign = (-1) ** io_bits_reshape[:, 0].astype(np.int8)
        qe_sign = (-1) ** qe_bits_reshape[:, 0].astype(np.int8)
        qo_sign = (-1) ** qo_bits_reshape[:, 0].astype(np.int8)
        ie_codes = ie_bits_reshape[:, 1:].dot(power_factor)
        io_codes = io_bits_reshape[:, 1:].dot(power_factor)
        qe_codes = qe_bits_reshape[:, 1:].dot(power_factor)
        qo_codes = qo_bits_reshape[:, 1:].dot(power_factor)

        if THIDX[i] <= threshold_limit:
            table = np.asarray(list(range(M_code_limit)) + [A[int(THIDX[i])]])
        else:
            table = np.array(NRL) * SF[int(THIDX[i])]

        ie_data = ie_sign * table[ie_codes]
        qe_data = qe_sign * table[qe_codes]
        io_data = io_sign * table[io_codes]
        qo_data = qo_sign * table[qo_codes]

        pos_ini = i * baq_block_length
        pos_fin = pos_ini + 2 * data_per_block
        decoded_data[pos_ini:pos_fin:2] = ie_data + 1j * qe_data
        decoded_data[pos_ini + 1:pos_fin:2] = io_data + 1j * qo_data

    decoded_params['THIDX'] = THIDX
    return error, decoded_data, decoded_params


def _decode_fdbaq_mode(data, header, huffman_array):
    """Decode the data applying the FDBAQ decoding method.

    This function applies the FDBAQ decode method as defined in the Sentinel 1
    SAR Space Packet Protocol Data Unit.

    :parameter
        data (np.array): Array of uint8 containing the raw data.
        header (dict):   Header containing pulse metadata information
        huffman_array (list(HuffmanCoding)): List with the applicable huffman
                      codes defined for Sentinel FDBAQ mode

    :return
        error (int): Error code
            -1 if the number of 16-bit words in the data is not sufficient.
            0 if correct
        decoded_data (np.array): Decoded complex data.
        decoded_params (dict): THIDX and BRC values of each block
    """

    error = 0
    decode_params = {}

    baq_block_length = header['RadarConfiguration']['BAQBlockLength']
    NQ = header['RadarSamples']['NumberOfQuads']

    NB = int(np.ceil(2 * NQ / baq_block_length))  # Number of BAQ blocks

    # Constant tables for reconstruction methods
    threshold_limit = [3, 3, 5, 6, 8]
    M_code_limit = [3, 4, 6, 9, 15]

    # Intermediate variables
    ie_codes = np.zeros((NB, baq_block_length // 2), dtype=int)
    io_codes = np.zeros((NB, baq_block_length // 2), dtype=int)
    qe_codes = np.zeros((NB, baq_block_length // 2), dtype=int)
    qo_codes = np.zeros((NB, baq_block_length // 2), dtype=int)
    ie_sign = np.ones((NB, baq_block_length // 2), dtype=int)
    io_sign = np.ones((NB, baq_block_length // 2), dtype=int)
    qe_sign = np.ones((NB, baq_block_length // 2), dtype=int)
    qo_sign = np.ones((NB, baq_block_length // 2), dtype=int)
    BRC = np.zeros(NB, dtype=int)
    THIDX = np.zeros(NB, dtype=int)

    # unpack byte-data stream to bit-data stream
    bit_data_stream = np.unpackbits(data, bitorder='big')
    last_block_size = NQ - baq_block_length // 2 * (NB - 1)
    pos = 0

    # Decoding block. Repetition for IE, IO, QE and QO blocks
    for code_type, codes, sign in [("IE", ie_codes, ie_sign),
                                   ("IO", io_codes, io_sign),
                                   ("QE", qe_codes, qe_sign),
                                   ("QO", qo_codes, qo_sign)]:
        for i in range(NB):
            if code_type == "IE":
                BRC[i] = sum(bit_data_stream[pos:pos + 3] * [4, 2, 1])
                pos += 3
            if code_type == "QE":
                THIDX[i] = sum(bit_data_stream[pos:pos + 8]
                               * [128, 64, 32, 16, 8, 4, 2, 1])
                pos += 8
            if i < NB - 1:  # Complete blocks
                huffman_code = huffman_array[int(BRC[i])]
                error, count, codes[i, :], sign[i, :] = huffman_code.decode(
                    bit_data_stream[pos:], baq_block_length // 2)
                pos += count
            else:
                huffman_code = huffman_array[int(BRC[i])]
                (error, count, codes[i, :last_block_size],
                 sign[i, :last_block_size]
                 ) = huffman_code.decode(bit_data_stream[pos:],
                                         last_block_size)
                pos += count
            if error != 0:
                return error, None, None

        if code_type != "QO":
            # Adjusting to next 16-bit data
            pos = int(np.ceil(pos / 16)) * 16
            if pos >= len(
                    bit_data_stream):  # End of stream -> error decoding data
                return -1, None, None

    # Reconstruction of pulse data
    decode_data = np.zeros(NQ * 2, dtype=complex)

    for i in range(NB):
        data_per_block = int(baq_block_length // 2) if i < NB - 1 else int(
            last_block_size)

        # Creation table of decoded values taking into account
        # THIDX and BRC of each block
        if THIDX[i] <= threshold_limit[int(BRC[i])]:
            table = np.asarray(list(range(M_code_limit[int(BRC[i])]))
                               + [B_TABLE[int(BRC[i])][int(THIDX[i])]])
        else:
            table = np.array(NRL_FDBAQ_TABLE[int(BRC[i])]) * SF[int(THIDX[i])]

        # Applying the table to decode the values obtained after
        # huffman decoding
        ie_data = ie_sign[i, :data_per_block] * table[
            ie_codes[i, :data_per_block]]
        qe_data = qe_sign[i, :data_per_block] * table[
            qe_codes[i, :data_per_block]]
        io_data = io_sign[i, :data_per_block] * table[
            io_codes[i, :data_per_block]]
        qo_data = qo_sign[i, :data_per_block] * table[
            qo_codes[i, :data_per_block]]

        # Reconstruction of the complex data and storage in the correct
        # part of the pulse
        pos_inic = i * baq_block_length
        pos_fin = pos_inic + 2 * data_per_block
        decode_data[pos_inic:pos_fin:2] = ie_data + 1j * qe_data
        decode_data[pos_inic + 1:pos_fin:2] = io_data + 1j * qo_data

    decode_params['THIDX'] = THIDX
    decode_params['BRC'] = BRC

    return error, decode_data, decode_params


def decode_data(data, header, huffman_codes, fid_log):
    """Decode the data applying the decoding method specified in the header.

    This function decodes the pulse received taking into account the coded
    process executed on-board (bypass, BAQ or FDBAQ) and marked in the pulse
    header.

    :parameter
        data (np.array): Array of uint8 containing the raw data.
        header (dict):   Header containing pulse metadata information
        huffman_codes (list(HuffmanCoding)): List with the applicable huffman
                      codes defined for Sentinel FDBAQ mode
        fid_log (TextIO) (optional value): File descriptor where the log
                    messages will be written. If None, no log messages are
                    generated

    :return
        error (int): Error code
            -2 if header parameter BAQMode is not valid
            -1 if the number of 16-bit words in the data is not sufficient.
            0 if correct
        decoded_data (np.array): Decoded complex data.
        decoded_params (dict):
            None for bypass mode,
            THIDX values of each block for BAQ mode
            THIDX and BRC values of each block for FDBAQ mode
    """

    baq_mode_code = header['RadarConfiguration']['BAQMode_Code']

    if baq_mode_code == 0:
        if fid_log:
            fid_log.write("Bypass mode decoding\n")
        return _decode_bypass_mode(data, header)
    elif baq_mode_code in {3, 4, 5}:
        if fid_log:
            fid_log.write("BAQ mode decoding\n")
        return _decode_baq_mode(data, header)
    elif baq_mode_code in {12, 13, 14}:
        if fid_log:
            fid_log.write("FDBAQ mode decoding\n")
        return _decode_fdbaq_mode(data, header, huffman_codes)
    else:
        print("WARN: Unknown BAQ mode, returning empty data")
        if fid_log:
            fid_log.write("WARN: Unknown BAQ mode, returning empty data\n")
        return -2, None, None
