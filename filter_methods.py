# filter_methods.py
import numpy as np
from scipy import signal
from scipy.signal import freqz, remez
from window_types import get_window  # Importing the get_window function

def design_fir_filter(method='window', cutoff=None, numtaps=101, window=None, filter_type='lowpass', samplerate=44100):
    if method == 'window':
        #cutoff = np.asarray(cutoff)
        #cutoff_n = cutoff / (samplerate / 2)  # Normalize cutoff
        """Correct bandpass filter implementation without frequency doubling"""
        cutoff = np.asarray(cutoff)
        nyq = samplerate / 2
        cutoff_n = cutoff / nyq  # Normalize to Nyquist
        
        # Get window coefficients - use 'window' parameter consistently
        if isinstance(window, str):  # Changed window_type to window
            window = get_window(window, numtaps)
        else:
            window = np.ones(numtaps)
            

        n = np.arange(numtaps)
        t = n - (numtaps - 1)/2  # Centered time vector
        t[t == 0] = 1e-20  # Avoid division by zero
        
        if filter_type == 'lowpass':
 
            #h = np.sinc(cutoff_n * t) * window * cutoff
            h = 2 * cutoff_n * np.sinc(2 * cutoff_n * t) * window

            
            # Normalize for unity gain at DC
            return h / np.sum(h)

            
        elif filter_type == 'highpass':
                     # Correct highpass formula
            h = np.sinc(t) - cutoff_n * np.sinc(cutoff_n * t)
            h *= window
            h[(numtaps - 1) // 2] = 1 - cutoff_n
            #return h / np.sum(h)
            '''
            # Normalize for unity gain at Nyquist
            w, H = freqz(h, worN=8000, fs=samplerate)
            nyq_idx = len(H)//2  # Index at Nyquist frequency
            h = h / np.abs(H[nyq_idx])
            '''
            return h
            
        elif filter_type == 'bandpass':

           
            # Enhanced bandpass with sharper transition
            h = (2*cutoff_n[1]*np.sinc(cutoff_n[1]*t) - 
                 2*cutoff_n[0]*np.sinc(cutoff_n[0]*t)) * window
            # Normalize at geometric mean frequency
            w, H = freqz(h, worN=8000, fs=samplerate)
            center_freq = np.sqrt(cutoff[0] * cutoff[1])
            center_idx = np.argmin(np.abs(w - center_freq))
            h = h / np.abs(H[center_idx])
           
            
            return h          
        elif filter_type == 'bandstop':
            omega = np.pi * cutoff / nyq  # Convert to digital frequencies
            n = np.arange(numtaps)
            h = np.sinc(n - (numtaps-1)/2)  # Allpass component
            # Subtract bandpass components
            h -= (omega[1]/np.pi)*np.sinc((omega[1]/np.pi)*(n - (numtaps-1)/2))
            h += (omega[0]/np.pi)*np.sinc((omega[0]/np.pi)*(n - (numtaps-1)/2))
            h *= window
            h /= np.sum(h)

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


