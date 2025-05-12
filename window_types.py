# window_types.py
import numpy as np
from scipy.signal import get_window as scipy_get_window  # SciPy's window function generator

def get_window(name, numtaps):
    """
    Enhanced window function generator with custom defaults and error handling.
    
    This wrapper provides:
    - Custom default parameters for certain windows (like Kaiser)
    - Unified exception handling
    - Case-insensitive window name matching
    
    Parameters:
        name (str): Name of the window function. Supported types:
                   'boxcar', 'hamming', 'hann', 'blackman', 'kaiser', 
                   'bartlett', 'flattop', 'nuttall', etc.
        numtaps (int): Number of points in the window (must be >= 1)
    
    Returns:
        ndarray: The window function values
    
    Raises:
        ValueError: If window name is not recognized
    
    Example Usage:
        # Get a 101-point Hamming window
        hamming = get_window('hamming', 101)
        
        # Get a Kaiser window with default beta
        kaiser = get_window('kaiser', 101)
    """
    try:
        # Handle special cases with custom parameters
        if name.lower() == 'kaiser':
            # Default beta value provides good balance between:
            # - Main lobe width (frequency resolution)
            # - Side lobe attenuation (stopband rejection)
            beta = 8.6  # Typical value for many audio applications
            
            # SciPy requires Kaiser window as a tuple (name, beta)
            return scipy_get_window(('kaiser', beta), numtaps)
        else:
            # Standard window types pass through directly
            return scipy_get_window(name, numtaps)
    # Handle exceptions        
    except Exception as e:
        # Convert all scipy exceptions to ValueError with clear message
        raise ValueError(
            f"Window type '{name}' not recognized or invalid parameters. "
            f"Supported types: boxcar, hamming, hann, blackman, kaiser, "
            f"bartlett, flattop, nuttall, etc. Original error: {str(e)}"
        )
