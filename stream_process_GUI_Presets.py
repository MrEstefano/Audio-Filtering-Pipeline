import os
import soxr
import time
import psutil
import resource
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk
import sounddevice as sd
from collections import deque
import matplotlib.pyplot as plt
from fir_filter import create_fir_filter
from scipy.signal import fftconvolve, freqz, minimum_phase
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Global buffers (can be managed within GUI class if preferred, but for clarity)
silence_block = np.zeros(65536, dtype=np.float32) # Will be sized based on blocksize
audio_buffer = deque(maxlen=4)
# input_buffer is now managed internally by OverlapAddFilter instances

def is_symmetric(h, tol=1e-8):
    return np.allclose(h, h[::-1], atol=tol)

def normalize_filter(h, fs):
    """Normalize filter gain to 0 dB at DC or center of band."""
    w, H = freqz(h, worN=8000, fs=fs)
    max_gain = np.max(np.abs(H))
    # Avoid division by zero if filter is all zeros or near zero
    if max_gain < 1e-10:
        return h
    return h / max_gain


class OverlapAddFilter:
    """
    Implements Overlap-Add convolution for real-time FIR filtering.
    """
    def __init__(self, filter_coeffs, block_size):
        # We don't use fft_overlap_factor directly here in the __init__,
        # as the standard Overlap-Add derives the FFT size from block_size and filter_length.
        self.set_coefficients(filter_coeffs, block_size)

    def set_coefficients(self, filter_coeffs, block_size):
        if len(filter_coeffs) == 0:
            self.coeffs = np.array([1.0], dtype=np.float32) # Identity filter
        else:
            self.coeffs = np.array(filter_coeffs, dtype=np.float32)

        self.block_size = block_size # L (input block size)
        self.filter_length = len(self.coeffs) # N (filter length)

        # In Overlap-Add, the output of convolving a block of L samples with a filter of N taps
        # is L + N - 1 samples long.
        # The FFT length (M) must be >= L + N - 1 and typically a power of 2 for efficiency.
        required_fft_length = self.block_size + self.filter_length - 1
        self.fft_length = int(2**np.ceil(np.log2(required_fft_length)))
       
        # The overlap length is N - 1.
        self.overlap_length = self.filter_length - 1

        # Pre-compute FFT of the filter coefficients, padded to FFT length.
        self.filter_fft = np.fft.rfft(self.coeffs, self.fft_length)

        # Initialize the overlap buffer (the tail from the previous block).
        # Its size is filter_length - 1.
        self.overlap_buffer = np.zeros(self.overlap_length, dtype=np.float32)

    def process_block(self, input_block):
        # Ensure input_block has the correct length (L)
        if len(input_block) != self.block_size:
            # This shouldn't happen if the callback handles it correctly, but as a safeguard.
            if len(input_block) < self.block_size:
                input_block = np.pad(input_block, (0, self.block_size - len(input_block)), 'constant')
            else:
                input_block = input_block[:self.block_size]

        # 1. Pad the input block to the FFT length (M).
        # input_block is L samples. We need M - L zeros for padding.
        padded_input = np.pad(input_block, (0, self.fft_length - self.block_size), 'constant', constant_values=0)

        # 2. Compute FFT of the padded input block.
        input_fft = np.fft.rfft(padded_input)

        # 3. Multiply in the frequency domain.
        convolved_fft = input_fft * self.filter_fft

        # 4. Inverse FFT to get the time domain convolution result.
        # This result is M samples long.
        convolved_time = np.fft.irfft(convolved_fft, self.fft_length)

        # 5. Overlap-Add operation:
        # The output for the current block is the first L samples of convolved_time,
        # plus the overlap from the previous block.
       
        # Allocate output buffer
        output_block = np.zeros(self.block_size, dtype=np.float32)
       
        # Add the overlap from the previous block to the *beginning* of the current block's convolution result.
        # The overlap region length is min(self.block_size, self.overlap_length)
        overlap_add_region_len = min(self.block_size, self.overlap_length)

        if overlap_add_region_len > 0:
            # Add the previous overlap to the first part of the current convolution result
            output_block[:overlap_add_region_len] = convolved_time[:overlap_add_region_len] + self.overlap_buffer[:overlap_add_region_len]
       
        # Copy the non-overlapping part of the current convolution result
        if self.block_size > overlap_add_region_len:
            output_block[overlap_add_region_len:] = convolved_time[overlap_add_region_len : self.block_size]

        # 6. Update the overlap buffer for the *next* block.
        # This is the tail of the current convolution, which has length (N-1)
        # It starts from `self.block_size` and goes for `self.overlap_length` samples.
        # Ensure that `convolved_time` is long enough.
        new_overlap = convolved_time[self.block_size : self.block_size + self.overlap_length]

        # If for some reason new_overlap is shorter than expected (shouldn't be with correct fft_length), pad it.
        if len(new_overlap) < self.overlap_length:
            new_overlap = np.pad(new_overlap, (0, self.overlap_length - len(new_overlap)), 'constant')
       
        self.overlap_buffer = new_overlap

        return output_block

class EqualizerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Real-Time Audio Equalizer")
        self.plot_counter = 0
        self.plot_interval = 5
        self.optimize_process()
        self.setup_variables()  # Initializes all variables first
        self.initialize_buffers()
        self.create_controls()  # Then creates controls that use those variables
        self.precompute_eq_filters()
        self.update_fir_filter()
        self.plot_response(
            self.applied_config["samplerate"] * self.applied_config["upsample_factor"],
            self.applied_config["filter_type"]
        )
        self.canvas.draw()
        
    def create_preset_controls(self):
        frame = ttk.LabelFrame(self.master, text="Presets")
        frame.grid(row=0, column=5, rowspan=5, padx=10, pady=5)
        
        ttk.Combobox(frame, textvariable=self.current_preset, 
                    values=list(self.presets.keys())).pack(padx=5, pady=5)
        ttk.Button(frame, text="Save", command=self.save_preset).pack(padx=5, pady=2)
        ttk.Button(frame, text="Load", command=self.load_preset).pack(padx=5, pady=2)
        ttk.Button(frame, text="Delete", command=self.delete_preset).pack(padx=5, pady=2)

    def save_preset(self):
        """Save current settings as a preset"""
        try:
            name = self.current_preset.get()
            if not name:
                print("Please enter a preset name")
                return
                
            self.presets[name] = {
                'gains': [var.get() for var in self.eq_gains],
                'settings': {k:v for k,v in self.applied_config.items() 
                            if k not in ['samplerate', 'blocksize']}
            }
            # Update combobox
            self.current_preset['values'] = list(self.presets.keys())
        except Exception as e:
            print(f"Error saving preset: {e}")

    def load_preset(self):
        """Load selected preset"""
        try:
            preset_name = self.current_preset.get()
            if preset_name in self.presets:
                preset = self.presets[preset_name]
                for var, gain in zip(self.eq_gains, preset['gains']):
                    var.set(gain)
                self.apply_changes()
        except Exception as e:
            print(f"Error loading preset: {e}")

    def delete_preset(self):
        """Delete the currently selected preset"""
        try:
            preset_name = self.current_preset.get()
            if preset_name in self.presets:
                del self.presets[preset_name]
                # Update combobox values
                self.current_preset['values'] = list(self.presets.keys())
                # Clear the current selection if we deleted it
                if preset_name == self.current_preset.get():
                    self.current_preset.set('')
        except Exception as e:
            print(f"Error deleting preset: {e}")
            
    def optimize_process(self):
        try:
            p = psutil.Process()
            os.nice(-10)  # Higher priority
            p.cpu_affinity(list(range(psutil.cpu_count())))
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (-1, -1))
        except Exception as e:
            print(f"Process optimization warning: {e}")

    def setup_variables(self):
        self.eq_bands = [
            ("Low Bass", (20, 60)),
            ("Mid Bass", (60, 120)),
            ("High Bass", (120, 250)),
            ("Low Midrange", (250, 500)),
            ("Middle Midrange", (500, 1000)),
            ("High Midrange", (1000, 2000)),
            ("Low Treble", (2000, 4000)),
            ("Middle Treble", (4000, 8000)),
            ("High Treble", (8000, 16000))
        ]

        self.eq_gains = [tk.DoubleVar(value=1.0) for _ in self.eq_bands]
        self.cutoff = tk.StringVar(value="16000")
        self.cutoff_low = tk.StringVar(value="500")
        self.cutoff_high = tk.StringVar(value="15000")
        self.numtaps = tk.StringVar(value="257")
        self.window_type = tk.StringVar(value='hamming')
        self.filter_type = tk.StringVar(value='lowpass')
        self.min_phase = tk.BooleanVar(value=False)
       
        self.samplerate = tk.StringVar(value="44100")
        self.upsample_factor = tk.StringVar(value="1")
        self.blocksize = tk.StringVar(value="1024")

        self.show_spectrum = tk.BooleanVar(value=False)
        self.show_waterfall = tk.BooleanVar(value=False)  # Moved here
        
        # Waterfall data storage
        self.waterfall_data = deque(maxlen=50)
        
        # Preset system
        self.presets = {}
        self.current_preset = tk.StringVar(value="Default")

        self.applied_config = {
            "samplerate": 44100,
            "upsample_factor": 1,
            "blocksize": 1024,
            "cutoff": 16000,
            "numtaps": 257,
            "window_type": 'hamming',
            "filter_type": 'lowpass',
            "min_phase": False
        }

    def initialize_buffers(self):
        bs = self.applied_config["blocksize"]
        self.silence_block = np.zeros(bs, dtype=np.float32)
        self.last_output = np.zeros(bs, dtype=np.float32)
        
        # Safety for very large blocks
        if bs > 8192:
            print(f"Reducing block size from {bs} to 4096")
            self.silence_block = np.zeros(4096, dtype=np.float32)
            self.last_output = np.zeros(4096, dtype=np.float32)
            self.applied_config["blocksize"] = 4096
            
    def precompute_eq_filters(self):
        self.eq_filter_objects = []
        sr_processing = self.applied_config["samplerate"] * self.applied_config["upsample_factor"]
        processing_block_size = self.applied_config["blocksize"] * self.applied_config["upsample_factor"]
        
        # Limit maximum processing block size
        max_processing_size = 8192 * 4  # 32768 samples at 4x upsample
        if processing_block_size > max_processing_size:
            print(f"Reducing processing block size from {processing_block_size} to {max_processing_size}")
            processing_block_size = max_processing_size


        
        for i, (label, freq_range) in enumerate(self.eq_bands):
            coeffs = create_fir_filter(
                method='window',
                cutoff=freq_range,
                numtaps=self.applied_config["numtaps"],
                window_type='hamming',
                filter_type='bandpass',
                samplerate=sr_processing
            )
            if self.applied_config["min_phase"] and is_symmetric(coeffs):
                coeffs = minimum_phase(coeffs, method="hilbert")
                coeffs = normalize_filter(coeffs, sr_processing) # Normalize after min-phase
            self.eq_filter_objects.append(OverlapAddFilter(coeffs, processing_block_size))

    def create_controls(self):
        for i, (label, freq_range) in enumerate(self.eq_bands):
            tk.Label(self.master, text=label).grid(row=i, column=0, padx=5, pady=5, sticky="w")
            scale = tk.Scale(
                self.master, from_=0.0, to=2.0, resolution=0.01,
                orient="horizontal", variable=self.eq_gains[i], length=200,
                showvalue=False
            )
            scale.grid(row=i, column=1, padx=5, pady=5)
            value_label = tk.Label(self.master, textvariable=self.eq_gains[i], width=5)
            value_label.grid(row=i, column=3, padx=5, pady=5, sticky="w")

        param_start_row = len(self.eq_bands) + 1
        fields = [
            ("Cutoff Frequency", self.cutoff),
            ("Low Cutoff (for band)", self.cutoff_low),
            ("High Cutoff (for band)", self.cutoff_high),
            ("Taps", self.numtaps),
            ("Sample Rate", self.samplerate),
            ("Upsample Factor (max 4)", self.upsample_factor),
            ("Block Size", self.blocksize)
        ]

        for idx, (label, var) in enumerate(fields):
            ttk.Label(self.master, text=label).grid(row=param_start_row + idx, column=0, padx=5, pady=2, sticky="w")
            tk.Entry(self.master, textvariable=var).grid(row=param_start_row + idx, column=1, columnspan=2, padx=5, pady=2, sticky="we")

        row = param_start_row + len(fields)
        ttk.Label(self.master, text="Window").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(self.master, textvariable=self.window_type,
                     values=[
                         'boxcar', 'triang', 'blackman', 'hamming', 'hann',
                         'bartlett', 'flattop', 'parzen', 'bohman',
                         'blackmanharris', 'nuttall', 'barthann'
                     ]).grid(row=row, column=1, columnspan=2, padx=5, pady=2, sticky="we")

        row += 1
        ttk.Label(self.master, text="Filter Type").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(self.master, textvariable=self.filter_type,
                     values=['lowpass', 'highpass', 'bandpass', 'bandstop']
                     ).grid(row=row, column=1, columnspan=2, padx=5, pady=2, sticky="we")

        row += 1
        ttk.Checkbutton(self.master, text="Show Spectrum", variable=self.show_spectrum
                        ).grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)

        row += 1
        ttk.Checkbutton(self.master, text="Minimum Phase Filter", variable=self.min_phase
                        ).grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)

        row += 1
        ttk.Button(self.master, text="Apply Settings", command=self.apply_changes
                   ).grid(row=row, column=0, columnspan=3, pady=10)
        
        row += 1
        ttk.Checkbutton(self.master, text="Waterfall Display", 
                       variable=self.show_waterfall).grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        
        # Create preset controls frame
        self.create_preset_controls()
        self.figure = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().grid(row=0, column=4, rowspan=row+1, padx=10, pady=5)
        


    def apply_changes(self):
        try:

            # Validate blocksize is power of 2 for better FFT performance
            blocksize = max(256, min(8192, int(self.blocksize.get())))
            blocksize = 1 << (blocksize - 1).bit_length()  # Round to nearest power of 2
            
            # Rest of your apply_changes code...
            self.applied_config["blocksize"] = blocksize
            # Safely parse parameters with validation
            upf = min(max(1, int(self.upsample_factor.get())), 4)
            blocksize = max(256, min(4096, int(self.blocksize.get())))
            
            ftype = self.filter_type.get()
            if ftype in ["bandpass", "bandstop"]:
                cutoff = [( max(20, min(float(self.cutoff_low.get()), self.applied_config["samplerate"]/2)),
                    max(20, min(float(self.cutoff_high.get())), self.applied_config["samplerate"]/2))
                ]
            else:
                cutoff = max(20, min(float(self.cutoff.get()), self.applied_config["samplerate"]/2))

            self.applied_config = {
                "samplerate": max(8000, min(192000, int(self.samplerate.get()))),
                "upsample_factor": upf,
                "blocksize": blocksize,
                "cutoff": cutoff,
                "numtaps": max(16, min(2048, int(self.numtaps.get()))),
                "window_type": self.window_type.get(),
                "filter_type": ftype,
                "min_phase": self.min_phase.get()
            }

            self.precompute_eq_filters()
            self.update_fir_filter()
            self.plot_response(self.applied_config["samplerate"] * upf, ftype)

        except Exception as e:
            print(f"Error applying changes: {e}")
            # Fall back to safe defaults if needed
            self.applied_config["blocksize"] = 1024
            self.applied_config["upsample_factor"] = 1
            
    def get_gains(self):
        return [var.get() for var in self.eq_gains]

    def get_filter_config(self):
        return (
            self.applied_config["cutoff"],
            self.applied_config["numtaps"],
            self.applied_config["window_type"],
            self.applied_config["filter_type"]
        )

    def get_dsp_config(self):
        return (
            self.applied_config["samplerate"],
            self.applied_config["upsample_factor"],
            self.applied_config["blocksize"]
        )

    def update_fir_filter(self):
        try:
            sr = self.applied_config["samplerate"]
            upf = self.applied_config["upsample_factor"]
            bs = self.applied_config["blocksize"]
            
            # Calculate and limit processing block size
            processing_block_size = bs * upf
            max_processing_size = 16384  # Experiment with this value
            
            if processing_block_size > max_processing_size:
                print(f"Reducing processing block size from {processing_block_size} to {max_processing_size}")
                processing_block_size = max_processing_size
                
            samplerate_processing = sr * upf
            config = self.get_filter_config()
            cutoff = config[0]
           
            coeffs = create_fir_filter(
                method='window',
                cutoff=cutoff,
                numtaps=config[1],
                window_type=config[2],
                filter_type=config[3],
                samplerate=samplerate_processing
            )
            
            if self.applied_config["min_phase"] and is_symmetric(coeffs):
                coeffs = minimum_phase(coeffs, method="hilbert")
                coeffs = normalize_filter(coeffs, samplerate_processing)
           
            # Create or update the OverlapAddFilter instance
            self.fir_filter_object = OverlapAddFilter(coeffs, processing_block_size)

        except Exception as e:
            print(f"Error updating FIR: {e}")
            
    def plot_response(self, fs, filter_type):
        if self.show_waterfall.get():
            self.plot_waterfall(fs)
        else:
            self.plot_standard(fs, filter_type)
    
    def plot_waterfall(self, fs):
        self.figure.clf()
        ax = self.figure.add_subplot(111, projection='3d')
        
        if len(self.waterfall_data) == 0:
            ax.text(0.5, 0.5, "No spectrum data available", 
                   ha='center', va='center', transform=ax.transAxes)
            self.canvas.draw()
            return
            
        try:
            # Convert to array and check all spectra have same length
            spectra_lengths = [len(spec) for spec in self.waterfall_data]
            if len(set(spectra_lengths)) > 1:
                # Find most common length
                target_length = max(set(spectra_lengths), key=spectra_lengths.count)
                # Truncate all spectra to common length
                self.waterfall_data = deque([spec[:target_length] for spec in self.waterfall_data], 
                                          maxlen=self.waterfall_data.maxlen)
            
            spectra = np.array(self.waterfall_data)
            times = np.arange(len(self.waterfall_data)) * (self.applied_config["blocksize"]/self.applied_config["samplerate"])
            
            # Calculate frequency axis based on FFT size
            fft_size = (len(self.waterfall_data[0]) - 1) * 2  # Reconstruct original window size
            freqs = np.fft.rfftfreq(fft_size, d=1.0/(self.applied_config["samplerate"] * self.applied_config["upsample_factor"]))
            
            X, Y = np.meshgrid(freqs, times)
            ax.plot_surface(X, Y, spectra, cmap='viridis', rstride=1, cstride=1)
            
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Time (s)')
            ax.set_zlabel('Magnitude (dB)')
            ax.set_title('Waterfall Spectrum Display')
        except Exception as e:
            print(f"Waterfall plotting error: {e}")
            ax.text(0.5, 0.5, "Error displaying waterfall", 
                   ha='center', va='center', transform=ax.transAxes)
        
        self.canvas.draw()
        
    def plot_standard(self, fs, filter_type):
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        
        # Plot filter response
        if self.fir_filter_object and len(self.fir_filter_object.coeffs) > 0:
            w, h = freqz(self.fir_filter_object.coeffs, worN=8000, fs=fs)
            ax.plot(w, 20 * np.log10(np.abs(h) + 1e-6), label="Filter Response")
        
        # Plot spectrum if enabled
        if self.show_spectrum.get() and hasattr(self, 'last_output') and len(self.last_output) > 0:
            try:
                audio_data = self.last_output.copy()
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data / max_val
                
                window = np.hanning(len(audio_data))
                spectrum = np.fft.rfft(audio_data * window)
                spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-12)
                freqs = np.fft.rfftfreq(len(audio_data), d=1.0/self.applied_config["samplerate"])
                
                ax.plot(freqs, spectrum_db, color='orange', alpha=0.6, label="Output Spectrum")
                
                # Peak marker
                peak_idx = np.argmax(spectrum_db)
                peak_freq = freqs[peak_idx]
                peak_db = spectrum_db[peak_idx]
                ax.plot(peak_freq, peak_db, 'ro', markersize=8)
                ax.annotate(f'{peak_freq:.1f}Hz\n{peak_db:.1f}dB', 
                           xy=(peak_freq, peak_db),
                           xytext=(10, 10),
                           textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
                
                # Store for waterfall
                if self.show_waterfall.get():
                    self.waterfall_data.append(spectrum_db)
                
            except Exception as e:
                print(f"Spectrum plotting error: {e}")

        ax.set_title(f"Filter Response ({filter_type.capitalize()})")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.grid(True)
        ax.legend()
        ax.set_xlim(20, self.applied_config["samplerate"]/2)
        self.canvas.draw()


def safe_upsample(data, sr, upsr):
    try:
        # soxr.resample expects mono input for 1D array
        if data.ndim > 1:
            data = data[:, 0]
        # Pad with zeros if data length is zero to avoid soxr error
        if len(data) == 0:
            return np.zeros(0, dtype=np.float32)
        return soxr.resample(data, sr, upsr, quality='VHQ')
    except Exception as e:
        print(f"Resampling error: {e}")
        # Return a zero array of appropriate size if resampling fails
        return np.zeros(int(len(data) * (upsr / sr)), dtype=np.float32)

def apply_dither(audio, bit_depth=24):
    if len(audio) == 0:
        return audio
    dither = (np.random.random(len(audio)) - 0.5) * (2 / (2**bit_depth))
    return audio + dither

def start_audio_stream(gui):
    sr, upf, bs = gui.get_dsp_config()
    try:
        with sd.Stream(
            samplerate=sr,
            blocksize=bs,
            channels=1,
            dtype='float32',
            latency='low',
            callback=make_audio_callback(gui),
            device=(0, 0) # Replace with your actual device indices
        ):
            print("Audio stream started. Press Ctrl+C to stop.")
            threading.Event().wait()  # Keeps the thread alive
    except Exception as e:
        print(f"Audio stream error: {e}")
        print("Ensure your audio device indices (0, 0) are correct or remove 'device' argument.")
        print("You can list devices using: python -m sounddevice")
       

    # Internal buffer to store upsampled input for processing
    # This buffer needs to be large enough to hold multiple upsampled blocks
    # if you were doing manual shifting, but with OverlapAddFilter,
    # it primarily processes the current upsampled block.
    # The actual processing happens on the upsampled_input_block.
def process_audio_block(gui, upsampled_block):
    """Process a single upsampled audio block through EQ and FIR filters"""
    # EQ processing
    eq_result = upsampled_block.copy()
    current_eq_sum = np.zeros_like(eq_result)
    
    for eq_filter, gain_var in zip(gui.eq_filter_objects, gui.eq_gains):
        band_output = eq_filter.process_block(upsampled_block)
        current_eq_sum += gain_var.get() * band_output
        
    eq_result = current_eq_sum
    
    # FIR processing
    if gui.fir_filter_object:
        return gui.fir_filter_object.process_block(eq_result)
    return eq_result

    return audio_callback  

def make_audio_callback(gui):
    # Persistent buffers
    input_residual = np.zeros(0, dtype=np.float32)
    output_residual = np.zeros(0, dtype=np.float32)
    
    def audio_callback(indata, outdata, frames, time_info, status):
        nonlocal input_residual, output_residual
        sr, upf, bs = gui.get_dsp_config()
        upsr = sr * upf
        
        try:
            # 1. Collect input and combine with residual
            input_mono = indata[:, 0] if indata.ndim > 1 else indata
            full_input = np.concatenate((input_residual, input_mono))
            
            # 2. Process in complete blocks
            num_blocks = len(full_input) // bs
            processed_chunks = []
            
            for i in range(num_blocks):
                block = full_input[i*bs:(i+1)*bs]
                
                # 3. Upsample
                upsampled = safe_upsample(block, sr, upsr)
                
                # 4. Process through EQ
                eq_result = upsampled.copy()
                current_eq_sum = np.zeros_like(eq_result)
                for eq_filter, gain_var in zip(gui.eq_filter_objects, gui.eq_gains):
                    band_output = eq_filter.process_block(upsampled)
                    current_eq_sum += gain_var.get() * band_output
                eq_result = current_eq_sum
                
                # 5. Process through FIR
                if gui.fir_filter_object:
                    filtered = gui.fir_filter_object.process_block(eq_result)
                else:
                    filtered = eq_result
                
                # 6. Downsample
                downsampled = safe_upsample(filtered, upsr, sr)
                processed_chunks.append(downsampled)
            
            # 7. Handle residuals
            input_residual = full_input[num_blocks*bs:]
            
            # 8. Combine output
            if processed_chunks:
                full_output = np.concatenate([output_residual] + processed_chunks)
                output_residual = full_output[frames:] if len(full_output) > frames else np.zeros(0, dtype=np.float32)
                outdata[:, 0] = full_output[:frames]
                gui.last_output = full_output[:frames].copy()
                
                # 9. Waterfall data collection
                if gui.show_waterfall.get() and len(filtered) > 0:
                    try:
                        windowed = filtered * np.hanning(len(filtered))
                        spectrum = np.fft.rfft(windowed)
                        spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-12)
                        gui.waterfall_data.append(spectrum_db)
                    except Exception as e:
                        print(f"Waterfall error: {str(e)}")
            else:
                outdata[:, 0] = gui.silence_block[:frames]
                
        except Exception as e:
            print(f"Processing error: {str(e)}")
            outdata[:, 0] = gui.silence_block[:frames]
            
    return audio_callback


if __name__ == '__main__':
    # Ensure these are executed with appropriate permissions if needed (e.g., sudo)
    # resource.setrlimit(resource.RLIMIT_MEMLOCK, (-1, -1))
    # os.system('sudo cpufreq-set -g performance')
    # os.sched_setaffinity(0, {0}) # Binds to CPU 0, consider multiple cores if available

    root = tk.Tk()
   
    gui = EqualizerGUI(root)
   
    # Start the audio stream in a separate thread
    audio_thread = threading.Thread(target=start_audio_stream, args=(gui,), daemon=True)
    audio_thread.start()

    # Start the GUI main loop
    root.mainloop()
	

