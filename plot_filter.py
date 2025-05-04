import matplotlib
matplotlib.use('Agg')  # Set backend to Agg for non-GUI environments
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz

def plot_filter_response(coefficients, fs=44100):
    # Use freqz to compute the frequency response
    w, h = freqz(coefficients, worN=8000, fs=fs)  # worN is the number of frequency points
    magnitude = 20 * np.log10(np.abs(h) + 1e-8)  # Add small number to avoid log(0)

    # Plot magnitude response
    plt.figure(figsize=(10, 6))

    # Magnitude plot
    plt.subplot(2, 1, 1)
    plt.plot(w, magnitude)
    plt.title('FIR Filter Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid()

    # Phase plot (optional)
    plt.subplot(2, 1, 2)
    plt.plot(w, np.angle(h))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.grid()

    plt.tight_layout()
