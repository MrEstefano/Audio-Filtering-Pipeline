# stream_process.py
import numpy as np
import sounddevice as sd
from fir_filter import create_fir_filter

# Filter Type   Cutoff Format   Example
# lowpass       Single float    0.2
# highpass      Single float    0.3
# bandpass      List or tuple of 2      [0.2, 0.4]
# bandstop      List or tuple of 2      [0.45, 0.5]

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

# === FIR Filter Parameters ===
NUM_TAPS = 101
CUTOFF = 13000  # Use list for bandpass/bandstop
WINDOW_TYPE = 'nuttall'
FILTER_TYPE = 'lowpass'  # lowpass | highpass | bandpass | bandstop ( if remez method selected, only high and lowpass works)
SAMPLERATE = 44100
CHANNELS = 1


# === Design FIR Filter ===
fir_coeff = create_fir_filter(
    method='window',   # remez
    cutoff=CUTOFF,
    numtaps=NUM_TAPS,
    window_type=WINDOW_TYPE,
    filter_type=FILTER_TYPE,
    samplerate = SAMPLERATE
)


# === Create Filter Buffer ===
buffer = np.zeros(NUM_TAPS - 1)

def audio_callback(indata, outdata, frames, time, status):
    global buffer
#    start = time_module.time()  # ← you'll need to import time as time_module
    if status:
        print(f"Stream status: {status}")

    # Flatten input for mono processing
    samples = indata[:, 0]

    # Concatenate buffer and input
    x = np.concatenate((buffer, samples))

    # Apply FIR filter via convolution
    y = np.convolve(x, fir_coeff, mode='valid')

    # Update buffer for next callback
    buffer = x[-(NUM_TAPS - 1):]

    # Output to speaker/DAC
    outdata[:, 0] = y.astype(np.float32)
#    if callback_counter % 50 == 0:
#       print(f"Block time: {time_module.time() - start:.5f}s")


if __name__ == "__main__":

    print("Streaming to PCM5102 DAC... Press Ctrl+C to stop.")
    try:
        with sd.Stream(
            samplerate=SAMPLERATE,
            blocksize=1024,  # Try 1024, 2048, or higher
            channels=CHANNELS,
            dtype='float32',
            latency='high',  # You can try 'low', 'high', or explicit like 0.1
            callback=audio_callback,
            device=(1,0)  # or replace with ("input_device", "output_device")
        ):
            while True:
                sd.sleep(1000)  # effectively run forever
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"Error: {e}")
