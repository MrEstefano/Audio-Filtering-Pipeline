# stream_process.py
import numpy as np
import sounddevice as sd
import soxr
from scipy import signal
import os
from fir_filter import create_fir_filter
from plot_filter import plot_filter_response

# === Audio Configuration ===
SAMPLERATE = 44100
UPSAMPLE_FACTOR = 4  # Changed to 1x,2x,3x,4x
UPSAMPLE_RATE = SAMPLERATE * UPSAMPLE_FACTOR  
CHANNELS = 1
BLOCKSIZE = 1024
NUM_TAPS = 101  # Must be odd

# === Filter Configuration ===
# if you choose FILTER_TYPE = 'remez' method selected, only high and lowpass works
# if you choose FILTER_TYPE = 'window' method selected, all lowpass | highpass | bandpass | bandstop  will work 
FILTER_TYPE = 'lowpass'
#CUTOFF = [800, 10000] #for bands
CUTOFF = 10000  # Below Nyquist (44.1kHz)
WINDOW_TYPE = 'nuttall'#'hamming'#'nuttall' #'boxcar'#('kaiser', 12)#'hamming'# #'blackman'#


# Filter Type   Cutoff Format   Example
# lowpass       Single float    0.2
# highpass      Single float    0.3
# bandpass      List or tuple of 2      [200, 4000]
# bandstop      List or tuple of 2      [4500, 12500]

# 0.00  0 Hz    DC (no frequency)
# 0.05  2,205 Hz        Low-frequency bass
# 0.10  4,410 Hz        Lower midrange
# 0.15  6,615 Hz        Midrange
# 0.20  8,820 Hz        Upper mids
# 0.25  11,025 Hz       Treble starts
# 0.30  13,230 Hz       High treble
# 0.35  15,435 Hz       Bright upper highs
# 0.40  17,640 Hz       Near audible upper limit
# 0.45  19,845 Hz       Almost Nyquist
# 0.50  22,050 Hz       Nyquist / Folding point

# Window Name   Function name (string)  Characteristics
# Rectangular   "boxcar"        Sharp transitions, poor sidelobe performance (a lot of ringing).
# Hamming       "hamming"       Good general-purpose window with low sidelobes.
# Hann (Hanning)"hann"          Similar to Hamming, but slightly wider main lobe.
# Blackman      "blackman"      Better sidelobe suppression, but wider main lobe.
# Kaiser        "kaiser"        Parameterized (with beta), tradeoff between main lobe width and sidelobes.
# Gaussian      "gaussian"      Parameterized by std, smooth window shape.
# Bartlett      "bartlett"      Triangular shape, simple and fast.
# Tukey         "tukey"         Combines rectangular and Hann via alpha parameter.
# Flat Top      "flattop"       Flat frequency response, useful for accurate amplitude measurement.
# Nuttall       "nuttall"       Very low sidelobes, useful in precision filtering.



# Create filter
fir_coeff = create_fir_filter(
    method='window',
    cutoff=CUTOFF,
    numtaps=NUM_TAPS,
    window_type=WINDOW_TYPE,
    filter_type=FILTER_TYPE,
    samplerate=UPSAMPLE_RATE
    #min_phase=False,
    #weight=[1, 20]  # Strong stopband emphasis
)

# Plot filter response
plot_filter_response(fir_coeff, fs=UPSAMPLE_RATE, filter_type=FILTER_TYPE)

# === Buffer Setup ===
input_buffer_size = NUM_TAPS + (BLOCKSIZE * UPSAMPLE_FACTOR) - 1
input_buffer = np.zeros(input_buffer_size, dtype=np.float32)
print(f"Buffer size: {input_buffer_size}")  # Should be 801 + 2048 - 1 = 2848

def apply_dither(audio, bit_depth=24):
    """Apply TPDF dithering"""
    dither = (np.random.random(len(audio)) - 0.5) * (2 / (2**bit_depth))
    return audio + dither

def audio_callback(indata, outdata, frames, time, status):
    global input_buffer
    
    if status:
        print(f"Stream status: {status}")
    
    # 1. Upsample (1024 ? 2048 samples)
    upsampled = soxr.resample(
        indata[:, 0],
        SAMPLERATE,
        UPSAMPLE_RATE,
        quality='VHQ'
    )
    
    # 2. Validate sizes
    if len(upsampled) != BLOCKSIZE * UPSAMPLE_FACTOR:
        print(f"Warning: Expected {BLOCKSIZE*UPSAMPLE_FACTOR} samples, got {len(upsampled)}")
    
    # 3. Update buffer (efficient roll)
    input_buffer[:-len(upsampled)] = input_buffer[len(upsampled):]
    input_buffer[-len(upsampled):] = upsampled
    
    # 4. Process with FFT convolution
    processed = signal.fftconvolve(input_buffer, fir_coeff, mode='valid')
    
    # 5. Downsample (2048 ? 1024 samples)
    downsampled = processed[::UPSAMPLE_FACTOR][:frames]
    
    # 6. Output
    outdata[:, 0] = apply_dither(downsampled)

if __name__ == "__main__":
    print(f"Starting DSP processing with {UPSAMPLE_FACTOR}x upsampling...")
    print(f"Input buffer size: {len(input_buffer)}")
    print(f"Upsampled block size: {BLOCKSIZE * UPSAMPLE_FACTOR}")
    
    try:
        with sd.Stream(
            samplerate=SAMPLERATE,
            blocksize=BLOCKSIZE,
            channels=CHANNELS,
            dtype='float32',
            latency='high',
            callback=audio_callback,
            device=(1, 0)
        ):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"Error: {e}")
