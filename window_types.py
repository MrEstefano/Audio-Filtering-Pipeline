# window_types.py
import numpy as np
from scipy.signal import get_window as scipy_get_window

def get_window(name, numtaps):
    try:
        if name.lower() == 'kaiser':
            beta = 8.6  # Or another value you want
            return scipy_get_window(('kaiser', beta), numtaps)
        else:
            return scipy_get_window(name, numtaps)
    except Exception as e:
        raise ValueError(f"Window type '{name}' not recognized: {e}")


