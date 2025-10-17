import numpy as np
import cv2
import socket
import struct
import heapq
import os
import sys

## ------------------------- Encoder ------------------------- ##

def imageEncoder(orgImageFileName, quantizationStepSize):
    # 1. Read 512x512 grayscale image
    img = cv2.imread(orgImageFileName, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unable to read")
    
    if img.shape != (512, 512):
        img = cv2.resize(img, (512, 512))

    img = img.astype(np.float64)

    # 2. Apply 5-level DWT with (5, 3) wavelet
    coeffs = dwt_53_forward(img, levels=5)

    # 3. Quantize coefficients
    quantized_coeffs = []
    for i, coeff in enumerate(coeffs):
        if i == 0:   # LL band
            quantized_coeffs.append(np.round(coeff / quantizationStepSize))
        else:        # LH, HL, HH bands
            quantized_tuple = tuple(np.round(c / quantizationStepSize) for c in coeff)
            quantized_coeffs.append(quantized_tuple)

    # 4. Apply prediction to LL subband
    ll_band = quantized_coeffs[0]
    predicted_ll = np.zeros_like(ll_band)
    predicted_ll[0, 0] = ll_band[0, 0]
    # First row
    predicted_ll[0, 1:] = ll_band[0, 1:] - ll_band[0, :-1]
    # First column
    predicted_ll[1:, 0] = ll_band[1:, 0] - ll_band[:-1, 0]
    # Rest of the image
    predictor = np.floor((ll_band[:-1, 1:] + ll_band[1:, :-1]) / 2.0)
    predicted_ll[1:, 1:] = ll_band[1:, 1:] - predictor
    
    quantized_coeffs[0] = predicted_ll

    # 5. Scan coefficients and generate symbols
    symbol_sequence = []
    # LL band - raster scan
    ll_flat = quantized_coeffs[0].flatten()
    for val in ll_flat:
        if val != 0:
            symbol_sequence.append((int(abs(val)).bit_length(), val))
        else:
            symbol_sequence.append(('IZ', ))
    # High frequency bands - EZT scan
    symbol_sequence.extend(ezt_scan(quantized_coeffs[1:]))
    # 6. Huffman encoding
    huffman_codes, encoded_data = huffman_encode(symbol_sequence)

    # Save bitstream
    with open('image.bit', 'wb') as f:
        # Write Huffman codes (codebook)
        write_huffman_codes(f, huffman_codes) 
        # Write the length of encoded_data (to identify the padding)
        f.write(struct.pack('I', len(encoded_data)))
        # Write encoded data
        write_encoded_data(f, encoded_data)
    # 7. Get bitrate
    file_size_bytes = os.path.getsize('image.bit')
    total_bits = file_size_bytes * 8
    bitrate = total_bits / (512 * 512)

    return bitrate



def dwt_53_forward(img, levels=5):
    """
    Perform 5-level 2D DWT using (5,3) biorthogonal wavelet
    Returns coefficients in format: [cA_n, (cH_n, cV_n, cD_n), ... (cH_1, cV_1, cD_1)]
    """
    coeffs = []
    current_img = img.copy()
    
    for _ in range(levels):
        # Perform 1D transform on rows
        rows_transformed = np.zeros_like(current_img)
        for i in range(current_img.shape[0]):
            rows_transformed[i,:] = dwt_53_1d(current_img[i,:])
        
        # Perform 1D transform on columns
        cols_transformed = np.zeros_like(rows_transformed)
        for j in range(rows_transformed.shape[1]):
            cols_transformed[:,j] = dwt_53_1d(rows_transformed[:,j])
        
        # Split into subbands
        h = cols_transformed.shape[0] // 2
        w = cols_transformed.shape[1] // 2
        cA = cols_transformed[:h, :w]
        cH = cols_transformed[:h, w:]
        cV = cols_transformed[h:, :w]
        cD = cols_transformed[h:, w:]
        
        # Store detail coefficients and continue with approximation
        coeffs.append((cH, cV, cD))
        current_img = cA
    
    # The final approximation is the first element
    coeffs = [current_img] + coeffs[::-1]
    return coeffs

def dwt_53_1d(signal):
    """
    1D (5,3) wavelet transform
    """
    n = len(signal)
    even = signal[::2].astype(np.float64)
    odd = signal[1::2].astype(np.float64)

    # Padding
    even = np.insert(even, 0, even[1])
    even = np.append(even, even[-1])
    odd = np.insert(odd, 0, odd[0])
    
    # Predict
    odd -= (even[:-1] + even[1:]) / 2 
    
    # Update
    even[1:-1] += (odd[:-1] + odd[1:]) / 4
    
    # Pack coefficients
    transformed = np.zeros_like(signal)
    transformed[:n//2] = even[1:-1]
    transformed[n//2:] = odd[1:]
    
    return transformed



def ezt_scan(coeffs):
    symbols = []
    max_level = len(coeffs)
    # Create a map to track which coefficients are part of a zerotree
    ztr_map = [[[[False for _ in range(band.shape[1])]
                 for _ in range(band.shape[0])]
                 for band in level]
                 for level in coeffs]
    
    # Start scaning
    for level in range(max_level):
        bands = coeffs[level]     # LH, HL, HH bands at current level

        for band_idx in range(3): # 0:LH, 1:HL, 2:HH
            band = bands[band_idx]
            h, w = band.shape

            for i in range(h):
                for j in range(w):
                    # Skip if already part of a zerotree
                    if ztr_map[level][band_idx][i][j]:
                        continue

                    val = band[i, j]

                    if val != 0:
                        symbols.append((int(abs(val)).bit_length(), val))
                    else:                    
                        if level == 0:
                            # Check for zerotree root (and mark its descendants)
                            if _is_zerotree_root(coeffs, band_idx, i, j, max_level, ztr_map):
                                symbols.append(('ZTR',))
                            else:
                                symbols.append(('IZ',))
                        else:
                            symbols.append(('IZ',))
    
    return symbols

def _is_zerotree_root(coeffs, band_idx, i, j, max_level, ztr_map):
    # Check if ztr
    start_i, start_j = i, j
    for level in range(1, max_level):
        start_i, start_j = 2 * start_i, 2 * start_j
        size = 2**level
        for di in range(size):
            for dj in range(size):
                if coeffs[level][band_idx][start_i+di, start_j+dj] != 0:
                    return False
    
    # Mark zero tree
    start_i, start_j = i, j
    for level in range(1, max_level):
        start_i, start_j = 2 * start_i, 2 * start_j
        size = 2**level
        for di in range(size):
            for dj in range(size):
                ztr_map[level][band_idx][start_i+di][start_j+dj] = True

    return True



def huffman_encode(symbol_sequence: list):
    # Collect probability distribution
    symbol_counts = {}
    total_symbols = 0

    for symbol in symbol_sequence:     # ('ZTR',), ('IZ',), (size, val)
        if symbol[0] in symbol_counts:    
            symbol_counts[symbol[0]] += 1
        else:
            symbol_counts[symbol[0]] = 1
        total_symbols += 1

    for symbol in symbol_counts.keys():
        symbol_counts[symbol] = symbol_counts[symbol] / total_symbols
    
    # Get huffman codebook
    huffman_codes = generate_huffman_codes(symbol_counts)

    # Get encoded data
    encoded_data = []
    for symbol_tuple in symbol_sequence:
        key = symbol_tuple[0]
        encoded_data += huffman_codes[key]
        if len(symbol_tuple) == 2:
            val = symbol_tuple[1]
            if val < 0:
                val += 2**key - 1
            val_bit = np.binary_repr(val.astype(int), width=key)
            encoded_data.append(val_bit)

    return huffman_codes, "".join(encoded_data)

class HuffmanNode:
    def __init__(self, symbol=None, frequency=0):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.frequency < other.frequency
    
def generate_huffman_codes(probabilities):
    """
    Generate Huffman codes for given symbol probabilities.

    args:
        probabilities (dict): A dictionary mapping symbols to their probabilities.
    
    Returns:
        dict: A dictionary mapping symbols to their Huffman codes.
    """
    priority_queue = [HuffmanNode(symbol, freq) for symbol, freq in probabilities.items()]
    heapq.heapify(priority_queue)

    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)

        combined_node = HuffmanNode(frequency=left.frequency + right.frequency)
        combined_node.left = left
        combined_node.right = right

        heapq.heappush(priority_queue, combined_node)
    
    root = priority_queue[0]

    huffman_codes = {}

    def generate_codes(node, code=""):
        if not node:
            return
        if node.symbol is not None:
            huffman_codes[node.symbol] = code
        generate_codes(node.left, code + "0")
        generate_codes(node.right, code + "1")
    
    generate_codes(root)

    return huffman_codes



def write_huffman_codes(f, huffman_codes):
    symbol_order = ['ZTR', 'IZ', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # The order of codeword is: 'ZTR', 'IZ', '(1,)', '(2,)'...
    length = len(huffman_codes)
    code_count = 0
    for i, symbol in enumerate(symbol_order):
        if symbol in huffman_codes.keys():
            code_count += 1
            f.write(struct.pack('B', i))
            f.write(huffman_codes[symbol].encode('utf-8'))
            if code_count < length:
                f.write(b',')    # Code does not end
            else:
                f.write(b'.')    # Last code
        
        

def write_encoded_data(f, encoded_data):
    # Ensure bits align to bytes
    padded_encoded_data = encoded_data + '0' * ((8-len(encoded_data) % 8) % 8)

    # Convert the binary string to bytes
    byte_array = bytearray()
    for i in range(0, len(padded_encoded_data), 8):
        byte_array.append(int(padded_encoded_data[i:i+8], 2))
    f.write(byte_array)






## ------------------------- Network Communication ------------------------- ##

def send_file(filename, host, port):
    """
    Opens filename in binary, connects to (host,port),
    sends an 8‐byte little‐endian filesize header followed by the file data.
    """
    filesize = os.path.getsize(filename)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        print(f"[Client] Connecting to {host}:{port} ...")
        s.connect((host, port))
        print(f"[Client] Connected, sending header (filesize={filesize})")
        # send filesize as unsigned long long (8 bytes, little endian)
        s.sendall(struct.pack('<Q', filesize))
        
        # send file data in chunks
        with open(filename, 'rb') as f:
            sent = 0
            while sent < filesize:
                chunk = f.read(4096)
                if not chunk:
                    break
                s.sendall(chunk)
                sent += len(chunk)
        print(f"[Client] Finished sending {sent} bytes.")
    finally:
        s.close()



def receive_file(bind_host, port, dest_filename):
    """
    Binds and listens on (bind_host,port).  Accepts one incoming
    TCP connection, reads an 8‐byte little‐endian filesize header,
    then reads exactly that many bytes and writes them into dest_filename.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((bind_host, port))
    srv.listen(1)
    print(f"[Server] Listening on {bind_host}:{port} ...")
    conn, addr = srv.accept()
    print(f"[Server] Connection from {addr}")
    try:
        # first read the 8‐byte filesize header
        raw = b''
        while len(raw) < 8:
            chunk = conn.recv(8 - len(raw))
            if not chunk:
                raise ConnectionError("Failed to receive file size header")
            raw += chunk
        filesize = struct.unpack('<Q', raw)[0]
        print(f"[Server] Expecting {filesize} bytes.")
        
        # now receive the file
        with open(dest_filename, 'wb') as f:
            received = 0
            while received < filesize:
                chunk = conn.recv(min(4096, filesize - received))
                if not chunk:
                    raise ConnectionError("Connection closed prematurely")
                f.write(chunk)
                received += len(chunk)
        print(f"[Server] Received {received} bytes, saved to {dest_filename}")
    finally:
        conn.close()
        srv.close()







## ------------------------- Decoder ------------------------- ##

def imageDecoder(compressedFileName, quantizationStepSize, orgImageFileName):
    # Read original image for PSNR calculation
    original_img = cv2.imread(orgImageFileName, cv2.IMREAD_GRAYSCALE)
    if original_img is None:
        raise ValueError("Image not found or unable to read")
    if original_img.shape != (512, 512):
        original_img = cv2.resize(original_img, (512, 512))
    original_img = original_img.astype(np.float64)

    # Read compressed data
    with open(compressedFileName, 'rb') as f:
        # Read Huffman codes
        huffman_codes = read_huffman_codes(f)
        # Read the length of encoded_data
        encoded_data_len = struct.unpack('I', f.read(4))[0]
        # Read encoded data
        encoded_data = f.read()

    # 9. Huffman decoding
    symbol_sequence = huffman_decode(huffman_codes, encoded_data, encoded_data_len)
    # 10. Inverse scanning
    coeffs = inverse_scan(symbol_sequence)

    # 11. Inverse quantization
    reconstructed_coeffs = []
    for i, coeff in enumerate(coeffs):
        if i == 0:  # LL band
            reconstructed_coeffs.append(coeff * quantizationStepSize)
        else:  # High frequency bands
            reconstructed_tuple = tuple(c * quantizationStepSize for c in coeff)
            reconstructed_coeffs.append(reconstructed_tuple)

    # 12. Inverse DWT
    reconstructed_img = dwt_53_inverse(reconstructed_coeffs)

    # Clip and convert to uint8
    reconstructed_img = np.clip(reconstructed_img, 0, 255)
    reconstructed_img = reconstructed_img.astype(np.uint8)

    # Save reconstructed image as BMP file
    reconstructed_filename = 'reconstructed_' + orgImageFileName
    cv2.imwrite(reconstructed_filename, reconstructed_img)

    # 13. Calculate PSNR
    mse = np.mean((original_img - reconstructed_img) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr



def read_huffman_codes(f):
    # Read up to the period (dot) which ends the codebook line.
    code_bytes = b''
    code_idx = []
    while True:
        idx = struct.unpack('B', f.read(1))[0]  # Get the idx of code first
        code_idx.append(idx)
        b0 = f.read(1)
        code_bytes += b0
        while b0 != b',' and b0 != b'.':
            b0 = f.read(1)
            code_bytes += b0
        if b0 == b'.':
            break
    code_bytes = code_bytes[:-1]

    code_str = code_bytes.decode('utf-8')
    codes = code_str.split(',')
    
    # Map order of symbols: ZTR, IZ, 1, 2, ...
    # This must match the order in write_huffman_codes!
    symbol_order = ['ZTR', 'IZ', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    codebook = {}
    for i, code in zip(code_idx, codes):
        # Invert the huffman_codes dict: codeword -> symbol
        codebook[code] = symbol_order[i]

    return codebook

def huffman_decode(code_to_symbol, encoded_data, encoded_data_len):
    # Convert bytes to bitstring
    bits = ""
    for b in encoded_data:
        # Each byte
        bits += format(b, '08b')
    
    # Remove padding
    bits = bits[:encoded_data_len]

    # Traverse bitstring to decode symbols
    symbol_sequence = []
    idx = 0
    max_code_len = max(len(code) for code in code_to_symbol.keys())
    n = len(bits)
    while idx < n:
        found = False
        for clen in range(1, max_code_len+1):
            if idx + clen > n:
                break
            code = bits[idx:idx+clen]
            if code in code_to_symbol:
                symbol = code_to_symbol[code]
                idx += clen
                if isinstance(symbol, int):
                    num_bits = symbol
                    value_bits = bits[idx:idx+num_bits]
                    idx += num_bits
                    val = int(value_bits, 2)
                    # Convert sign-magnitude form if negative
                    range_mid = 2**(num_bits-1)
                    if val < range_mid:
                        val = val - (2**num_bits - 1)
                    symbol_sequence.append((num_bits, val))
                else:
                    symbol_sequence.append((symbol,))
                found = True
                break
        if not found:
            # Padding/trailing bits
            break
    return symbol_sequence



def inverse_scan(symbol_sequence):
    # reconstruct the 5-level DWT coefficient structure
    sizes = [16, 32, 64, 128, 256]  # DWT pyramid (for 5 levels)
    coeffs = []

    # 1. LL band (5th level, size 16x16)
    idx = 0
    LL_size = sizes[0]
    ll_band = np.zeros((LL_size, LL_size), dtype=np.float64)
    # Raster scan, with inverse prediction
    predicted_ll = np.zeros_like(ll_band)
    for i in range(LL_size):
        for j in range(LL_size):
            sym = symbol_sequence[idx]
            idx += 1
            if sym[0] == 'IZ':
                val = 0
            else:
                _, val = sym
            # Inverse prediction
            if i == 0 and j == 0:
                predicted_ll[i, j] = val
            elif i == 0:
                predicted_ll[i, j] = val + predicted_ll[i, j - 1]
            elif j == 0:
                predicted_ll[i, j] = val + predicted_ll[i - 1, j]
            else:
                pred = np.floor((predicted_ll[i - 1, j] + predicted_ll[i, j - 1]) / 2.0)
                predicted_ll[i, j] = val + pred
    coeffs.append(predicted_ll)

    # 2. High frequency subbands (EZT scan, 5 levels)
    band_shapes = [
        (sizes[i], sizes[i]) for i in range(5)  # (16,16)...(256,256)
    ]

    # For levels 1 to 5, collect (LH, HL, HH)
    for shape in band_shapes:
        bands = []
        for _ in range(3):
            bands.append(np.zeros(shape, dtype=np.float64))
        coeffs.append(tuple(bands))

    # Now, fill the high freq bands from the symbol sequence (EZT order).
    # The function ezt_unscan will fill these arrays, consuming symbols from 'symbol_sequence' (via idx):
    ezt_unscan(coeffs[1:], symbol_sequence, idx)
    # coeffs[1:] passed as reference; idx is current position in symbol_sequence; return value (not used) is new idx

    return coeffs

def ezt_unscan(coeffs, symbol_sequence, start_idx):
    """
    Fills coeffs in-place given symbol_sequence, starting at start_idx.
    coeffs: list of tuples, each tuple is (LH, HL, HH) for a level
    """
    # Create a map to track which coefficients are part of a zerotree
    ztr_map = [[[[False for _ in range(band.shape[1])]
                 for _ in range(band.shape[0])]
                 for band in level]
                 for level in coeffs]
    max_level = len(coeffs)
    idx = start_idx
    for level in range(max_level):
        bands = coeffs[level]
        h, w = bands[0].shape
        for band_idx in range(3):
            for i in range(h):
                for j in range(w):
                    # Skip if already part of a zerotree
                    if ztr_map[level][band_idx][i][j]:
                        continue
                    sym = symbol_sequence[idx]
                    idx += 1
                    # Mark zero tree    
                    if sym[0] == 'ZTR':
                        start_i, start_j = i, j
                        for sub_level in range(1, max_level):
                            start_i, start_j = 2 * start_i, 2 * start_j
                            size = 2**sub_level
                            for di in range(size):
                                for dj in range(size):
                                    ztr_map[sub_level][band_idx][start_i+di][start_j+dj] = True
                    if sym[0] in ('ZTR', 'IZ'):
                        val = 0
                    else:
                        _, val = sym
                    bands[band_idx][i, j] = val
    return idx



def dwt_53_inverse(coeffs):
    """
    Perform inverse 2D DWT using (5,3) biorthogonal wavelet
    Takes coefficients in format: [cA_n, (cH_n, cV_n, cD_n), ... (cH_1, cV_1, cD_1)]
    """
    current_cA = coeffs[0]
    
    for level in range(1, len(coeffs)):
        cH, cV, cD = coeffs[level]
        
        # Merge subbands
        h, w = current_cA.shape
        merged = np.zeros((h*2, w*2), dtype=np.float64)
        
        merged[:h, :w] = current_cA
        merged[:h, w:] = cH
        merged[h:, :w] = cV
        merged[h:, w:] = cD
        
        # Inverse transform columns
        cols_reconstructed = np.zeros_like(merged)
        for j in range(merged.shape[1]):
            cols_reconstructed[:,j] = idwt_53_1d(merged[:,j])
        
        # Inverse transform rows
        rows_reconstructed = np.zeros_like(cols_reconstructed)
        for i in range(cols_reconstructed.shape[0]):
            rows_reconstructed[i,:] = idwt_53_1d(cols_reconstructed[i,:])
        
        current_cA = rows_reconstructed
    
    return current_cA

def idwt_53_1d(transformed):
    """
    1D inverse (5,3) wavelet transform
    """
    n = len(transformed)
    even = transformed[:n//2].copy()
    odd = transformed[n//2:].copy()
    
    # Padding
    odd = np.insert(odd, 0, odd[0])   # d_(-1) is actually d_0
    even = np.append(even, even[-1])  # s_(l+1) is actually s_l
    odd = np.append(odd, odd[-2])     # d_(l+1) is actually d_(l-1)
    
    # Inverse update
    even -= (odd[:-1] + odd[1:]) / 4
    
    # Inverse predict
    odd[1:-1] += (even[:-1] + even[1:]) / 2
    
    # Merge samples
    signal = np.zeros_like(transformed)
    signal[::2] = even[:-1]
    signal[1::2] = odd[1:-1]
    
    return signal






## ------------------------- Main Functions ------------------------- ##

def main_encoder():
    if len(sys.argv) != 4:
        print("Usage: python image_compression.py encode <image_file> <quantization_step>")
        return
    
    image_file = sys.argv[2]
    q_step = float(sys.argv[3])
    
    bitrate = imageEncoder(image_file, q_step)
    print(f"Compression complete. Bitrate: {bitrate:.4f} bpp")

def main_decoder():
    if len(sys.argv) < 5:
        print("Usage: python image_compression.py decode <compressed_file> <quantization_step> <original_image> [--display]")
        return
    
    compressed_file = sys.argv[2]
    q_step = float(sys.argv[3])
    original_image = sys.argv[4]
    display_images = '--display' in sys.argv
    
    psnr = imageDecoder(compressed_file, q_step, original_image)
    print(f"Decompression complete. PSNR: {psnr:.2f} dB")
    
    if display_images:
        # Display the original and reconstructed images
        original = cv2.imread(original_image, cv2.IMREAD_GRAYSCALE)
        reconstructed = cv2.imread('reconstructed_' + original_image, cv2.IMREAD_GRAYSCALE)
        
        if original is None or reconstructed is None:
            print("Error: Could not load images for display")
            return
        
        # Create a window for display
        cv2.namedWindow('Comparison', cv2.WINDOW_NORMAL)
        
        # Stack images horizontally
        comparison = np.hstack((original, reconstructed))
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'Reconstructed', (522, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, f'PSNR: {psnr:.2f} dB', (200, 50), font, 1, (255, 255, 255), 2)
        
        # Show the comparison
        cv2.imshow('Comparison', comparison)
        
        # Wait for key press and then close
        print("Press any key in the image window to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main_send():
    if len(sys.argv) != 5:
        print("Usage: python image_compression.py send <compressed_file> <host> <port>")
        return
    
    compressed_file = sys.argv[2]
    host = sys.argv[3]
    port = int(sys.argv[4])
    
    print(f"Sending {compressed_file} to {host}:{port}...")
    send_file(compressed_file, host, port)
    print("File sent successfully")

def main_receive():
    if len(sys.argv) != 5:
        print("Usage: python image_compression.py receive <host> <port> <output_file>")
        return
    
    host = sys.argv[2]
    port = int(sys.argv[3])
    output_file = sys.argv[4]
    
    print(f"Listening on {host}:{port}...")
    received_file = receive_file(host, port, output_file)
    print(f"File received and saved as {received_file}")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Encode: python image_compression.py encode <image_file> <quantization_step>")
        print("  Decode: python image_compression.py decode <compressed_file> <quantization_step> <original_image>")
        print("  Send:   python image_compression.py send <compressed_file> <host> <port>")
        print("  Receive: python image_compression.py receive <host> <port> <output_file>")
        sys.exit(1)
    
    mode = sys.argv[1]
    if mode == "encode":
        main_encoder()
    elif mode == "decode":
        main_decoder()
    elif mode == "send":
        main_send()
    elif mode == "receive":
        main_receive()
    else:
        print("Invalid mode. Use 'encode', 'decode', 'send', or 'receive'")