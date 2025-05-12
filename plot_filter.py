import numpy as np
import sounddevice as sd
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Add this BEFORE importing matplotlib
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import freqz
import sys
from fir_filter import create_fir_filter

def plot_filter_response(coefficients, fs=44100, filter_type=None):
    """Enhanced filter analysis with centered impulse response and positive magnitude range"""
    try:
        # Try using Qt backend first (most compatible)
        matplotlib.use('Qt5Agg', force=True)
        plt.switch_backend('Qt5Agg')
        # Create figure with subplots
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
        ax_mag = fig.add_subplot(gs[0, 0])
        ax_phase = fig.add_subplot(gs[1, 0])
        ax_impulse = fig.add_subplot(gs[2, 0])

        # ===== Frequency Response =====
        w, h = freqz(coefficients, worN=8000, fs=fs)
        magnitude = 20 * np.log10(np.abs(h) + 1e-8)
        phase = np.angle(h)
        #phase = np.unwrap(np.angle(h))

        # Adaptive frequency scaling
        nyquist = fs / 2
        if nyquist > 20000:
            x_scale = 'log'
            x_ticks = [20, 100, 1000, 10000, nyquist]
            x_ticks = [x for x in x_ticks if x <= nyquist]
            x_lim = (20, nyquist)
        else:
            x_scale = 'linear'
            x_ticks = np.linspace(0, nyquist, num=9)
            x_lim = (0, nyquist)

        # Magnitude plot with extended range
        if x_scale == 'log':
            ax_mag.semilogx(w, magnitude, color='C0')
        else:
            ax_mag.plot(w, magnitude, color='C0')
        
        # Find peak magnitude to adjust y-axis
        peak_mag = np.max(magnitude)
        mag_range = max(5, peak_mag + 5)  # Ensure we see positive values
        
        ax_mag.set_title(f'{filter_type.capitalize()} Filter Response (Fs={fs/1000:.1f}kHz)')
        ax_mag.set_ylabel('Magnitude (dB)')
        ax_mag.set_ylim(-120, mag_range)  # Now shows positive values
        ax_mag.set_xticks(x_ticks)
        ax_mag.set_xticklabels([f"{int(x)}" if x >= 1 else f"{x:.1f}" for x in x_ticks])
        ax_mag.grid(True, which='both', linestyle=':')
        ax_mag.set_xlim(x_lim)

        # Phase plot
        if x_scale == 'log':
            ax_phase.semilogx(w, phase, color='C1')
        else:
            ax_phase.plot(w, phase, color='C1')
        ax_phase.set_ylabel('Phase (radians)')
        ax_phase.set_xlabel('Frequency (Hz)')
        ax_phase.grid(True, which='both', linestyle=':')
        ax_phase.set_xticks(x_ticks)
        ax_phase.set_xticklabels([f"{int(x)}" if x >= 1 else f"{x:.1f}" for x in x_ticks])
        ax_phase.set_xlim(x_lim)

        # ===== Centered Impulse Response =====
        impulse_response = coefficients.copy()
        center = len(impulse_response) // 2
        plot_range = min(200, len(impulse_response))  # Max 200 samples
        start = max(0, center - plot_range//2)
        end = min(len(impulse_response), start + plot_range)
        
        markerline, stemlines, baseline = ax_impulse.stem(
            np.arange(start, end),
            impulse_response[start:end],
            linefmt='C2-',
            markerfmt='C2o',
            basefmt='C7:'
        )
        plt.setp(stemlines, 'linewidth', 0.5)
        plt.setp(markerline, 'markersize', 3)
        
        ax_impulse.set_title('Impulse Response (Centered)')
        ax_impulse.set_xlabel('Samples')
        ax_impulse.set_ylabel('Amplitude')
        ax_impulse.grid(True, linestyle=':')
        ax_impulse.set_xlim(start, end)
        #ax_impulse.set_ylim(-0.5, 1)  # Fixed amplitude range

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    except Exception as e:
        print(f"Plotting error: {str(e)}")
        raise
