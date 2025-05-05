# stream_process.py
import numpy as np
import sounddevice as sd
from fir_filter import create_fir_filter
import matplotlib
import matplotlib.pyplot as plt
from plot_filter import plot_filter_response



# === FIR Filter Parameters ===
NUM_TAPS = 101
CUTOFF = 9100
#CUTOFF = [550, 12900]  # Use list for bandpass/bandstop
WINDOW_TYPE = 'nuttall'
FILTER_TYPE = 'lowpass'
SAMPLERATE = 44100
CHANNELS = 1
# if you choose FILTER_TYPE = 'remez' method selected, only high and lowpass works
# if you choose FILTER_TYPE = 'window' method selected, all lowpass | highpass | bandpass | bandstop  will work 



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



# === Design FIR Filter ===
# if you choose method = 'remez' method selected, only high and lowpass works
# if you choose method = 'window' method selected, all lowpass | highpass | bandpass | bandstop  will work 




# [Rest of your existing code...]
fir_coeff = create_fir_filter(
    method='window',
    cutoff=CUTOFF,
    numtaps=NUM_TAPS,
    window_type=WINDOW_TYPE,
    filter_type=FILTER_TYPE,
    samplerate=SAMPLERATE
)

# Show filter response (added delay to ensure plot appears)
plot_filter_response(fir_coeff,fs=SAMPLERATE,filter_type=FILTER_TYPE)
plt.pause(0.1)  # Give the plot window time to initialize

# [Keep your existing audio callback and main loop...]
buffer = np.zeros(NUM_TAPS - 1)

def audio_callback(indata, outdata, frames, time, status):
    global buffer
    if status:
        print(f"Stream status: {status}")

    samples = indata[:, 0]
    x = np.concatenate((buffer, samples))
    y = np.convolve(x, fir_coeff, mode='valid')
    buffer = x[-(NUM_TAPS - 1):]
    outdata[:, 0] = y.astype(np.float32)

if __name__ == "__main__":
    print("Streaming to PCM5102 DAC... Press Ctrl+C to stop.")
    try:
        with sd.Stream(
            samplerate=SAMPLERATE,
            blocksize=1024,
            channels=CHANNELS,
            dtype='float32',
            latency='high',
            callback=audio_callback,
            device=(1,0)
        ):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"Error: {e}")
