# stream_process.py
import numpy as np
import sounddevice as sd  # Audio I/O library
import soxr  # High-quality sample rate conversion
from scipy import signal  # Signal processing functions
import os
from fir_filter import create_fir_filter  # Custom FIR filter design
from plot_filter import plot_filter_response  # Filter visualization

# === Audio Configuration ===
SAMPLERATE = 44100  # Base sampling rate (Hz)
UPSAMPLE_FACTOR = 4  # Upsampling ratio (1x, 2x, 3x, or 4x)
UPSAMPLE_RATE = SAMPLERATE * UPSAMPLE_FACTOR  # Final processing rate
CHANNELS = 1  # Mono audio
BLOCKSIZE = 1024  # Audio block size (power of 2 for efficiency)
NUM_TAPS = 101  # FIR filter length (odd number prevents phase distortion)

# === Filter Configuration ===
# Note: 'remez' method only supports highpass/lowpass, 'window' supports all types
FILTER_TYPE = 'lowpass'  # Options: 'lowpass', 'highpass', 'bandpass', 'bandstop'

# Cutoff frequency configuration:
# - Single value for lowpass/highpass (Hz)
# - List/tuple of 2 values for bandpass/bandstop [low, high] (Hz)
CUTOFF = 10000  # Example: 10kHz lowpass cutoff

# Window type for FIR design (affects frequency response tradeoffs):
WINDOW_TYPE = 'nuttall'  # Options: 'hamming', 'blackman', 'kaiser', etc.

# 0 Hz    DC (no frequency)
# 2,205 Hz        Low-frequency bass
# 4,410 Hz        Lower midrange
# 6,615 Hz        Midrange
# 8,820 Hz        Upper mids
# 11,025 Hz       Treble starts
# 13,230 Hz       High treble
# 15,435 Hz       Bright upper highs
# 17,640 Hz       Near audible upper limit
# 19,845 Hz       Almost Nyquist
# 22,050 Hz       Nyquist / Folding point

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



# Create and visualize the FIR filter
fir_coeff = create_fir_filter(
    method='window',  # Design method ('window' or 'remez')
    cutoff=CUTOFF,
    numtaps=NUM_TAPS,
    window_type=WINDOW_TYPE,
    filter_type=FILTER_TYPE,
    samplerate=UPSAMPLE_RATE  # Filter designed for upsampled rate
)

# Plot the filter's frequency response
plot_filter_response(fir_coeff, fs=UPSAMPLE_RATE, filter_type=FILTER_TYPE)

# === Buffer Setup ===
# Calculate buffer size to accommodate:
# - Filter taps (NUM_TAPS)
# - Upsampled audio blocks (BLOCKSIZE*UPSAMPLE_FACTOR)
# - Additional space for convolution
input_buffer_size = NUM_TAPS + (BLOCKSIZE * UPSAMPLE_FACTOR) - 1
input_buffer = np.zeros(input_buffer_size, dtype=np.float32)
print(f"Buffer size: {input_buffer_size}")  # Debug output

def apply_dither(audio, bit_depth=24):
    """
    Apply triangular probability density function (TPDF) dithering
    to reduce quantization artifacts in low-level signals.
    
    Args:
        audio: Input audio signal
        bit_depth: Target bit depth (default 24-bit)
    
    Returns:
        Dithered audio signal
    """
    dither = (np.random.random(len(audio)) - 0.5) * (2 / (2**bit_depth))
    return audio + dither

def audio_callback(indata, outdata, frames, time, status):
    """
    Real-time audio processing callback function called by sounddevice for each block.
    
    Processing chain:
    1. Upsample input
    2. Update processing buffer
    3. FIR filtering via FFT convolution
    4. Downsample output
    5. Apply dither and send to output
    """
    global input_buffer
    
    if status:
        print(f"Stream status: {status}")
    
    # 1. Upsample using high-quality SOXR resampler
    upsampled = soxr.resample(
        indata[:, 0],  # Input mono audio
        SAMPLERATE,
        UPSAMPLE_RATE,
        quality='VHQ'  # Very High Quality mode
    )
    
    # 2. Validate block sizes match expectations
    if len(upsampled) != BLOCKSIZE * UPSAMPLE_FACTOR:
        print(f"Warning: Expected {BLOCKSIZE*UPSAMPLE_FACTOR} samples, got {len(upsampled)}")
    
    # 3. Efficient buffer update using array rolling
    input_buffer[:-len(upsampled)] = input_buffer[len(upsampled):]
    input_buffer[-len(upsampled):] = upsampled
    
    # 4. FIR filtering using FFT-based convolution (efficient for long filters)
    processed = signal.fftconvolve(input_buffer, fir_coeff, mode='valid')
    
    # 5. Downsample by taking every Nth sample
    downsampled = processed[::UPSAMPLE_FACTOR][:frames]
    
    # 6. Apply dither and send to output
    outdata[:, 0] = apply_dither(downsampled)

if __name__ == "__main__":
    print(f"Starting DSP processing with {UPSAMPLE_FACTOR}x upsampling...")
    print(f"Input buffer size: {len(input_buffer)}")
    print(f"Upsampled block size: {BLOCKSIZE * UPSAMPLE_FACTOR}")
    
    try:
        # Configure and start audio stream
        with sd.Stream(
            samplerate=SAMPLERATE,
            blocksize=BLOCKSIZE,
            channels=CHANNELS,
            dtype='float32',
            latency='high',  # Favor stability over low latency
            callback=audio_callback,
            device=(1, 0)  # Input/output device indices
        ):
            # Keep stream alive until interrupted
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"Error: {e}")
