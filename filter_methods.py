
# filter_methods.py
import numpy as np
from scipy.signal import remez
from window_types import get_window  # Importing the get_window function

def design_fir_filter(method='window', cutoff=None, numtaps=101, window=None, filter_type='lowpass', samplerate=44100):
    if method == 'window':
        cutoff_n = np.array(cutoff) / (samplerate / 2)  # Normalize cutoff
        # Use the get_window function from window_types.py to get the window
        if isinstance(window, str):
            window = get_window(window, numtaps)  # Get the window from window_types.py
        elif isinstance(window, np.ndarray):
            # It's already an array, no need to call get_window()
            pass
        else:
            window = np.ones(numtaps)  # Default to a rectangular window
        t = np.arange(numtaps) - (numtaps - 1) / 2

        if filter_type == 'lowpass':
            h = np.sinc(2 * cutoff_n * t) * window
        elif filter_type == 'highpass':
            h = (np.sinc(t) - np.sinc(2 * cutoff_n * t)) * window
        elif filter_type == 'bandpass':
            h = (np.sinc(2 * cutoff_n[1] * t) - np.sinc(2 * cutoff_n[0] * t)) * window
        elif filter_type == 'bandstop':
            h = (np.sinc(t) - (np.sinc(2 * cutoff_n[1] * t) - np.sinc(2 * cutoff_n[0] * t))) * window
        else:
            raise ValueError("Unsupported filter type.")
        return h / np.sum(h)
    elif method == 'remez':
	        if filter_type == 'lowpass':
            trans_width = min(1000, (samplerate / 2 - cutoff) / 2)
            high_cut = min(samplerate / 2, cutoff + trans_width)
            if cutoff >= high_cut:
                raise ValueError("Cutoff frequency too close to Nyquist for given transition width.")
            bands = [0, cutoff, high_cut, samplerate / 2]
            desired = [1, 0]
        elif filter_type == 'highpass':
            trans_width = min(1000, cutoff / 2)  # Prevents overlap near 0 Hz
            low_cut = max(0, cutoff - trans_width)
            bands = [0, low_cut, cutoff, samplerate / 2]
            desired = [0, 1]
        else:
            raise ValueError(f"Filter type '{filter_type}' not supported for remez yet.")
        return remez(numtaps, bands, desired, fs=samplerate)
    else:
        raise NotImplementedError(f"Method '{method}' not implemented.")

