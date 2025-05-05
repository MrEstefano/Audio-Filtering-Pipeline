import numpy as np
import sounddevice as sd
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Add this BEFORE importing matplotlib
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import freqz
import sys
#from plot_filter import plot_filter_response
from fir_filter import create_fir_filter

def plot_filter_response(coefficients, fs=44100, filter_type=None):
    """Enhanced plotting with backend checking"""
    try:
        # Try using Qt backend first (most compatible)
        matplotlib.use('Qt5Agg', force=True)
        plt.switch_backend('Qt5Agg')
       
        # Compute frequency response
        w, h = freqz(coefficients, worN=8000, fs=fs)
        magnitude = 20 * np.log10(np.abs(h) + 1e-8)
        phase = np.angle(h)
       
        # Create figure
        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle(f'FIR Filter Response | Type: {filter_type} | Taps: {len(coefficients)}')  
       
                # Magnitude plot
        ax_mag.plot(w, magnitude)
        ax_mag.set_ylabel('Magnitude (dB)')
        ax_mag.grid(True)
        ax_mag.set_ylim(-120, 5)
        
        # Phase plot
        ax_phase.plot(w, phase)
        ax_phase.set_xlabel('Frequency (Hz)')
        ax_phase.set_ylabel('Phase (radians)')
        ax_phase.grid(True)
        
        # Set x-axis ticks to 1000Hz intervals for both subplots
        max_freq = fs//2  # Nyquist frequency
        xticks = np.arange(0, max_freq+1000, 1000)  # 1000Hz steps
        for ax in [ax_mag, ax_phase]:
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{int(x)}" for x in xticks], rotation=45)
            ax.set_xlim(0, max_freq)
        
        plt.tight_layout()
        plt.show(block=False)
        
    except Exception as e:
        print(f"Plotting error: {str(e)}")
