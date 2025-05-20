import numpy as np
import sounddevice as sd
import soxr
from scipy import signal
from scipy.signal import oaconvolve  # 30% faster than fftconvolve
import os
from collections import deque
from fir_filter import create_fir_filter
from plot_filter import plot_filter_response
import resource

# Lock memory to prevent swapping
resource.setrlimit(resource.RLIMIT_MEMLOCK, (-1, -1))

# Pin to CPU core 0 (change if you have a dedicated core)
os.system('sudo cpufreq-set -g performance')  # Lock max CPU speed
os.sched_setaffinity(0, {0})               # Use cores 2-3 (less heat)
# === Audio Configuration ===
SAMPLERATE = 44100
UPSAMPLE_FACTOR = 4  # Changed to 2x
UPSAMPLE_RATE = SAMPLERATE * UPSAMPLE_FACTOR  # Now 88.2kHz
CHANNELS = 1
BLOCKSIZE = 4096
NUM_TAPS = 301  # Must be odd

# === Filter Configuration ===
FILTER_TYPE = 'lowpass'
#CUTOFF = [60, 10000]
CUTOFF = 11000  # Below Nyquist (44.1kHz)
WINDOW_TYPE = 'hamming'#'hamming'#'nuttall' #'boxcar'#('kaiser', 12)#'hamming'# #'blackman'#

# Define EQUALAZITION bands: [(low, high), gain]
EQ_BANDS = [
    ((60, 250), 1.0),     # Bass boost
    ((500, 2000), 1.2),   # Midrange neutral
    ((4000, 16000), 1.5)  # Treble cut
]

# Create Main FIR filter
fir_coeff = create_fir_filter(
    method='window',
    cutoff=CUTOFF,
    numtaps=NUM_TAPS,
    window_type=WINDOW_TYPE,
    filter_type=FILTER_TYPE,
    samplerate=UPSAMPLE_RATE
)

# Create EQ FIR filters for each band
eq_filters = []
for (cutoff, gain) in EQ_BANDS:
    coeffs = create_fir_filter(
        method='window',
        cutoff=cutoff,
        numtaps=NUM_TAPS,
        window_type=WINDOW_TYPE,
        filter_type='bandpass',
        samplerate=UPSAMPLE_RATE
    )
    eq_filters.append((coeffs, gain))
    
# Plot filter response
plot_filter_response(fir_coeff, fs=UPSAMPLE_RATE, filter_type=FILTER_TYPE)
# === Buffer Setup ===
input_buffer_size = NUM_TAPS + (BLOCKSIZE * UPSAMPLE_FACTOR) - 1
input_buffer = np.zeros(input_buffer_size, dtype=np.float32)
print(f"Buffer size: {input_buffer_size}")

# Audio buffer for underflow protection (stores 3 blocks)
audio_buffer = deque(maxlen=4)
silence_block = np.zeros(BLOCKSIZE, dtype=np.float32)

# Manual FFT convolution implementation
def fft_convolve(x, h):
    n = len(x) + len(h) - 1
    n_fft = 1 << (int(np.log2(n)) + 1)  # Next power of 2
    X = np.fft.fft(x, n_fft)
    H = np.fft.fft(h, n_fft)
    return np.fft.ifft(X * H)[:len(x) - len(h) + 1].real

def apply_dither(audio, bit_depth=24):
    """Apply TPDF dithering"""
    dither = (np.random.random(len(audio)) - 0.5) * (2 / (2**bit_depth))
    return audio + dither

def safe_upsample(data):
    """Protected upsampling with fallback"""
    try:
        return soxr.resample(
            data[:, 0] if data.ndim > 1 else data,
            SAMPLERATE,
            UPSAMPLE_RATE,
            quality='VHQ'
        )
    except Exception as e:
        print(f"Upsampling failed: {str(e)}")
        return np.zeros(BLOCKSIZE * UPSAMPLE_FACTOR, dtype=np.float32)

def audio_callback(indata, outdata, frames, time, status):
    global input_buffer

    if status:
        print(f"Stream status: {status}")
        if status.input_underflow or status.output_underflow:
            print("?? UNDERFLOW DETECTED!")

    try:
        # 1. Upsample safely
        upsampled = safe_upsample(indata)
        if len(upsampled) != BLOCKSIZE * UPSAMPLE_FACTOR:
            print(f"Size mismatch: got {len(upsampled)}")
            if len(upsampled) < BLOCKSIZE * UPSAMPLE_FACTOR:
                padded = np.zeros(BLOCKSIZE * UPSAMPLE_FACTOR, dtype=np.float32)
                padded[:len(upsampled)] = upsampled
                upsampled = padded
            else:
                upsampled = upsampled[:BLOCKSIZE * UPSAMPLE_FACTOR]

        # 2. Update input buffer
        input_buffer[:-len(upsampled)] = input_buffer[len(upsampled):]
        input_buffer[-len(upsampled):] = upsampled

        # 3. EQ Filtering (bandpass + gain)
        eq_len = len(input_buffer) - NUM_TAPS + 1
        
        eq_output = np.zeros_like(input_buffer)
        for coeffs, gain in eq_filters:
            band = oaconvolve(input_buffer, coeffs, mode='same')
            eq_output += gain * band

        # 4. ChooseFFT convolution method
        #final_output = oaconvolve(eq_output, fir_coeff, mode='same')   # tapping present
        #final_output = signal.fftconvolve(eq_output, fir_coeff, mode='valid')  # tapping present
        final_output = oaconvolve(eq_output, fir_coeff, mode='valid', axes=0)    # tapping not present
        #final_output = fft_convolve(eq_output, fir_coeff)
        #final_output = oaconvolve(eq_output, fir_coeff, mode='valid')

        # 5. Downsample
        downsampled = final_output[::UPSAMPLE_FACTOR]
        #downsampled = final_output[::UPSAMPLE_FACTOR][:frames] # I am testing both options

        # 6. Trim or pad to exact frame length
        if len(downsampled) < frames:
            padded = np.zeros(frames, dtype=np.float32)
            padded[:len(downsampled)] = downsampled
            downsampled = padded
        else:
            downsampled = downsampled[:frames]

        # 7. Buffer & output
        audio_buffer.append(downsampled)
        
        outdata[:, 0] = apply_dither(downsampled)

    except Exception as e:
        print(f"? Processing error: {str(e)}")
        if audio_buffer:
            safe_block = audio_buffer[-1]
            if len(safe_block) < frames:
                padded = np.zeros(frames, dtype=np.float32)
                padded[:len(safe_block)] = safe_block
                outdata[:, 0] = padded
            else:
                outdata[:, 0] = safe_block[:frames]
        else:
            outdata[:, 0] = silence_block[:frames]

       
    except Exception as e:
        print(f"Processing error: {str(e)}")
        if audio_buffer:
            safe_block = audio_buffer[-1]
            if len(safe_block) < frames:
                padded = np.zeros(frames, dtype=np.float32)
                padded[:len(safe_block)] = safe_block
                outdata[:, 0] = padded
            else:
                outdata[:, 0] = safe_block[:frames]
        else:
            outdata[:, 0] = silence_block[:frames]

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
            print("Audio stream started. Press Ctrl+C to stop.")
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
