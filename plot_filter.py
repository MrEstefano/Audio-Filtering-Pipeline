import numpy as np
import sounddevice as sd
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Required for Linux GUI compatibility
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import freqz
import sys
from fir_filter import create_fir_filter

def plot_filter_response(coefficients, fs=44100, filter_type=None):
    """
    Visualizes FIR filter characteristics including frequency response and impulse response.
    
    Features:
    - Magnitude response in dB scale
    - Phase response (wrapped or unwrapped)
    - Centered impulse response
    - Adaptive frequency scaling (log/linear)
    - Automatic layout adjustment
    
    Parameters:
        coefficients (ndarray): FIR filter coefficients
        fs (float): Sampling frequency in Hz (default: 44100)
        filter_type (str): Filter type for title (e.g., 'lowpass', 'bandpass')
    
    Returns:
        None (displays interactive plot)
    
    Raises:
        Exception: Propagates any plotting errors with context
    """
    try:
        # === Backend Configuration ===
        # Set Qt backend for better interactive features
        matplotlib.use('Qt5Agg', force=True)
        plt.switch_backend('Qt5Agg')

        # === Figure Setup ===
        fig = plt.figure(figsize=(12, 8))
        # Grid layout with 3 rows (magnitude:phase:impulse = 2:1:1 ratio)
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
        ax_mag = fig.add_subplot(gs[0, 0])  # Magnitude response
        ax_phase = fig.add_subplot(gs[1, 0])  # Phase response
        ax_impulse = fig.add_subplot(gs[2, 0])  # Impulse response

        # === Frequency Response Analysis ===
        # Compute frequency response (8000 points for smooth curves)
        w, h = freqz(coefficients, worN=8000, fs=fs)
        magnitude = 20 * np.log10(np.abs(h) + 1e-8)  # dB scale with epsilon for numerical stability
        phase = np.angle(h)  # Wrapped phase (-π to π)
        # phase = np.unwrap(np.angle(h))  # Unwrapped continuous phase

        # === Adaptive Axis Scaling ===
        nyquist = fs / 2  # Nyquist frequency
        if nyquist > 20000:  # Use log scale for high sample rates
            x_scale = 'log'
            # Logarithmically spaced tick marks
            x_ticks = [20, 100, 1000, 10000, nyquist]
            x_ticks = [x for x in x_ticks if x <= nyquist]  # Filter valid ticks
            x_lim = (20, nyquist)  # Start from 20Hz for log scale
        else:  # Linear scale for lower sample rates
            x_scale = 'linear'
            x_ticks = np.linspace(0, nyquist, num=9)  # 9 evenly spaced ticks
            x_lim = (0, nyquist)

        # === Magnitude Plot ===
        if x_scale == 'log':
            ax_mag.semilogx(w, magnitude, color='C0')  # Blue
        else:
            ax_mag.plot(w, magnitude, color='C0')
        
        # Dynamic Y-axis scaling to show passband ripple
        peak_mag = np.max(magnitude)
        mag_range = max(5, peak_mag + 5)  # Ensure 5dB headroom above peak
        
        ax_mag.set_title(f'{filter_type.capitalize()} Filter Response (Fs={fs/1000:.1f}kHz)')
        ax_mag.set_ylabel('Magnitude (dB)')
        ax_mag.set_ylim(-120, mag_range)  # -120dB floor to show stopband
        ax_mag.set_xticks(x_ticks)
        # Format ticks (integers for >=1Hz, decimals below)
        ax_mag.set_xticklabels([f"{int(x)}" if x >= 1 else f"{x:.1f}" for x in x_ticks])
        ax_mag.grid(True, which='both', linestyle=':')  # Light dotted grid
        ax_mag.set_xlim(x_lim)

        # === Phase Plot ===
        if x_scale == 'log':
            ax_phase.semilogx(w, phase, color='C1')  # Orange
        else:
            ax_phase.plot(w, phase, color='C1')
        ax_phase.set_ylabel('Phase (radians)')
        ax_phase.set_xlabel('Frequency (Hz)')
        ax_phase.grid(True, which='both', linestyle=':')
        ax_phase.set_xticks(x_ticks)
        ax_phase.set_xticklabels([f"{int(x)}" if x >= 1 else f"{x:.1f}" for x in x_ticks])
        ax_phase.set_xlim(x_lim)

        # === Impulse Response Plot ===
        impulse_response = coefficients.copy()
        center = len(impulse_response) // 2  # Middle sample
        plot_range = min(200, len(impulse_response))  # Show max 200 samples
        
        # Center the display around the middle
        start = max(0, center - plot_range//2)
        end = min(len(impulse_response), start + plot_range)
        
        # Stem plot for discrete-time visualization
        markerline, stemlines, baseline = ax_impulse.stem(
            np.arange(start, end),
            impulse_response[start:end],
            linefmt='C2-',  # Green stems
            markerfmt='C2o',  # Green circles
            basefmt='C7:'  # Gray baseline
        )
        plt.setp(stemlines, 'linewidth', 0.5)  # Thinner stems
        plt.setp(markerline, 'markersize', 3)  # Smaller markers
        
        ax_impulse.set_title('Impulse Response (Centered)')
        ax_impulse.set_xlabel('Samples')
        ax_impulse.set_ylabel('Amplitude')
        ax_impulse.grid(True, linestyle=':')
        ax_impulse.set_xlim(start, end)
        # ax_impulse.set_ylim(-0.5, 1)  # Optionally fix amplitude range

        # === Final Touches ===
        plt.tight_layout()  # Prevent label overlap
        plt.show(block=False)  # Non-blocking display
        plt.pause(0.1)  # Required for non-blocking plots to render

    except Exception as e:
        print(f"Plotting error: {str(e)}")
        raise  # Re-raise with full stack trace
