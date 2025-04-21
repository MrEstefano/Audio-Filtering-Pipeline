# fir_filter.py
from filter_methods import design_fir_filter
from window_types import get_window

def create_fir_filter(method='window', cutoff=0.3, numtaps=101, window_type='hamming', filter_type='lowpass', samplerate=44100):
    if method == 'window':
        window = get_window(window_type, numtaps)
    else:
        window = None

    coeffs = design_fir_filter(
        method=method,
        cutoff=cutoff,
        numtaps=numtaps,
        window=window,
        filter_type=filter_type,
        samplerate=samplerate
    )
    return coeffs
