# filter_methods.py
import numpy as np
from scipy.signal import freqz
from window_types import get_window

def design_fir_filter(method='window', cutoff=None, numtaps=101, window='hamming', filter_type='lowpass', samplerate=44100):
    """
    Design FIR filters using either window method or Remez algorithm.
    Parameters:
        method (str): 'window' or 'remez' design method
        cutoff: Single value (low/highpass) or [low,high] pair (bandpass/stop)
        numtaps (int): Number of filter coefficients (must be odd)
        window: Window function (string name or array)
        filter_type (str): 'lowpass', 'highpass', 'bandpass', or 'bandstop'
        samplerate (float): Sampling rate in Hz
    Returns:
        ndarray: FIR filter coefficients
    """ 
    if method == 'window':
        if cutoff is None:
            raise ValueError("Cutoff frequency must be provided.")
        n = np.arange(numtaps)
        t = n - (numtaps - 1) / 2
        t[t == 0] = 1e-20  # avoid division by zero in sinc
        
        cutoff = np.asarray(cutoff)
        nyq = samplerate / 2
        cutoff_n = cutoff / nyq  # Normalize to Nyquist

        if isinstance(window, str):
            window_vals = get_window(window, numtaps)
        else:
            window_vals = np.ones(numtaps)

        if filter_type == 'lowpass':
            h = cutoff_n * np.sinc(cutoff_n * t)
        
        elif filter_type == 'highpass':
            h = np.sinc(t) - cutoff_n * np.sinc(cutoff_n * t)

        elif filter_type == 'bandpass':
            h = ( cutoff_n[1] * np.sinc( cutoff_n[1] * t) -
                cutoff_n[0] * np.sinc(cutoff_n[0] * t))

        elif filter_type == 'bandstop':
            h = np.sinc(t) - (
                cutoff_n[1] * np.sinc(cutoff_n[1] * t) -
                cutoff_n[0] * np.sinc(cutoff_n[0] * t))
        else:
            raise ValueError("Invalid filter type")

        # Apply window AFTER full impulse response is computed
        h *= window_vals
        '''
        Normalize for unity gain
        '''
        # Ensures unity gain at DC (i.e., low frequencies pass through unaffected)
        if filter_type in ['lowpass']:
            h /= np.sum(h)
        elif filter_type in ['highpass']:
            # Normalize unity gain towards at Nyquist
            h /= np.sum(h * np.cos(2 * np.pi * 0.5 * t))  
        elif filter_type == 'bandpass':
            # Normalize at center frequency
            w, H = freqz(h, worN=8000, fs=samplerate)
            center_freq = np.sqrt(cutoff[0] * cutoff[1])
            center_idx = np.argmin(np.abs(w - center_freq))
            h /= np.abs(H[center_idx])
        elif filter_type == 'bandstop':
            w, H = freqz(h, worN=8000, fs=samplerate)
            idx1 = np.argmin(np.abs(w - cutoff[0]))
            idx2 = np.argmin(np.abs(w - cutoff[1]))
            gain = (np.abs(H[idx1]) + np.abs(H[idx2])) / 2
            h /= gain
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
        elif filter_type == 'bandpass':
            bands = [0, cutoff[0], cutoff[1], samplerate / 2]
            desired = [0, 1, 0]
        elif filter_type == 'bandstop':
            bands = [0, cutoff[0], cutoff[1], samplerate / 2]
            desired = [1, 0, 1] 
        return remez(numtaps, bands, desired, fs=samplerate)

    else:
        raise NotImplementedError(f"Method '{method}' not implemented.")




