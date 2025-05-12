# filter_methods.py
import numpy as np
from scipy.signal import freqz, remez
from window_types import get_window  # Importing the get_window function

def design_fir_filter(method='window', cutoff=None, numtaps=101, window=None, 
                     filter_type='lowpass', samplerate=44100):
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
        # Window method implementation (supports all filter types)
        cutoff = np.asarray(cutoff)
        nyq = samplerate / 2  # Nyquist frequency
        cutoff_n = cutoff / nyq  # Normalized cutoff (0-1)
        
        # Get window function
        if isinstance(window, str):
            window = get_window(window, numtaps)  # Built-in window function
        else:
            window = np.ones(numtaps)  # Default rectangular window
            
        # Create centered time vector (symmetric around zero)
        n = np.arange(numtaps)
        t = n - (numtaps - 1)/2
        t[t == 0] = 1e-20  # Avoid division by zero
        
        # === Filter Type Implementations ===
        if filter_type == 'lowpass':
            # Ideal lowpass = sinc(cutoff * t)
            h = np.sinc(cutoff_n * t) * window * cutoff
            return h / np.sum(h)  # Normalize for unity DC gain
            
        elif filter_type == 'highpass':
            # Highpass = delta(t) - lowpass
            h = np.sinc(t) - (cutoff_n * np.sinc(cutoff_n * t))
            h = h * window
            # Normalize at Nyquist frequency
            w, H = freqz(h, worN=8000, fs=samplerate)
            nyq_idx = len(H)//2
            return h / np.abs(H[nyq_idx])
            
        elif filter_type == 'bandpass':
            if len(cutoff_n) != 2:
                raise ValueError("Bandpass requires [low,high] cutoff")
            # Bandpass = highpass - lowpass
            h = (2*cutoff_n[1]*np.sinc(cutoff_n[1]*t) - 
                 2*cutoff_n[0]*np.sinc(cutoff_n[0]*t)) * window
            # Normalize at geometric mean frequency
            w, H = freqz(h, worN=8000, fs=samplerate)
            center_freq = np.sqrt(cutoff[0]*cutoff[1])
            center_idx = np.argmin(np.abs(w - center_freq))
            return h / np.abs(H[center_idx])
            
        elif filter_type == 'bandstop':
            if len(cutoff_n) != 2:
                raise ValueError("Bandstop requires [low,high] cutoff")
            # Bandstop = allpass - bandpass
            omega = np.pi * cutoff / nyq  # Convert to digital frequencies
            n = np.arange(numtaps)
            h = np.sinc(n - (numtaps-1)/2)  # Allpass component
            # Subtract bandpass components
            h -= (omega[1]/np.pi)*np.sinc((omega[1]/np.pi)*(n - (numtaps-1)/2))
            h += (omega[0]/np.pi)*np.sinc((omega[0]/np.pi)*(n - (numtaps-1)/2))
            # Apply window and normalize
            if isinstance(window, str):
                window = get_window(window, numtaps)
            h = h * window
            return h / np.sum(h)
            
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
            
    elif method == 'remez':
        # Parks-McClellan optimal equiripple design
        # (Currently only supports lowpass/highpass)
        
        if filter_type == 'lowpass':
            # Calculate transition width (auto-adjusts near Nyquist)
            trans_width = min(1000, (samplerate / 2 - cutoff) / 2)
            high_cut = min(samplerate / 2, cutoff + trans_width)
            if cutoff >= high_cut:
                raise ValueError("Cutoff too close to Nyquist")
            bands = [0, cutoff, high_cut, samplerate / 2]
            desired = [1, 0]  # Passband=1, Stopband=0
            
        elif filter_type == 'highpass':
            # Prevent transition band overlap at DC
            trans_width = min(1000, cutoff / 2)
            low_cut = max(0, cutoff - trans_width)
            bands = [0, low_cut, cutoff, samplerate / 2]
            desired = [0, 1]  # Stopband=0, Passband=1
            
        else:
            raise ValueError(f"Remez doesn't support {filter_type} yet")
            
        return remez(numtaps, bands, desired, fs=samplerate)
        
    else:
        raise NotImplementedError(f"Method '{method}' not implemented")
