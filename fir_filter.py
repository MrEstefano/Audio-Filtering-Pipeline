# fir_filter.py
from filter_methods import design_fir_filter  # Core filter design functions
from window_types import get_window  # Window function generator

def create_fir_filter(method='window', cutoff=0.3, numtaps=101, 
                     window_type='hamming', filter_type='lowpass', 
                     samplerate=44100):
    """
    Creates FIR filter coefficients with simplified interface.
    
    This is a wrapper function that provides a more user-friendly interface
    to the underlying filter design methods while handling window selection.
    
    Parameters:
        method (str): Design method - 'window' or 'remez'
        cutoff: Cutoff frequency/frequencies:
               - Single value for lowpass/highpass (Hz or normalized 0-1)
               - [low, high] pair for bandpass/bandstop (Hz or normalized)
        numtaps (int): Number of filter coefficients (should be odd)
        window_type (str): Window function name (only used for 'window' method)
        filter_type (str): Filter type - 'lowpass', 'highpass', 'bandpass', 'bandstop'
        samplerate (float): Sampling rate in Hz (required for Hz cutoff values)
    
    Returns:
        ndarray: Designed FIR filter coefficients
        
    Example Usage:
        # Lowpass filter with 101 taps at 1kHz cutoff
        coeffs = create_fir_filter(cutoff=1000, samplerate=44100)
        
        # Bandpass filter using Kaiser window
        coeffs = create_fir_filter(filter_type='bandpass',
                                 cutoff=[500, 5000],
                                 window_type='kaiser')
    """
    
    # Handle window selection based on design method
    if method == 'window':
        # Get the specified window function coefficients
        window = get_window(window_type, numtaps)
    else:
        # Remez method doesn't use window functions
        window = None

    # Design the filter using the core method
    coeffs = design_fir_filter(
        method=method,
        cutoff=cutoff,
        numtaps=numtaps,
        window=window_type,  # Pass window type name directly
        filter_type=filter_type,
        samplerate=samplerate
    )
    
    return coeffs

