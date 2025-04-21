
# filter_methods.py
import numpy as np
from scipy.signal import remez

def design_fir_filter(method='window', cutoff=0.3, numtaps=101, window=None, filter_type='lowpass', samplerate=44100):
    if method == 'window':
        h = np.sinc(2 * cutoff * (np.arange(numtaps) - (numtaps - 1) / 2)) * window
        return h
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
