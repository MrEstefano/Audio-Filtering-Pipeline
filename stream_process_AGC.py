
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
import scipy.signal as signal
import samplerate
from collections import deque
import matplotlib.pyplot as plt
from fir_filter import create_fir_filter
from scipy.signal import fftconvolve, freqz, minimum_phase, oaconvolve
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import queue

# Global thread-safe queues for audio processing
audio_queue = queue.Queue(maxsize=20)
processed_queue = queue.Queue(maxsize=20)

# Global variables for double buffering filter parameters
filter_lock = threading.Lock()
filter_update_lock = threading.Lock()
filter_update_in_progress = False
pending_filter_update = False
active_filter_params = None
inactive_filter_params = None
active_eq_filters = None
inactive_eq_filters = None

def is_symmetric(h, tol=1e-8):
    return np.allclose(h, h[::-1], atol=tol)

def normalize_filter(h, fs):
    w, H = freqz(h, worN=8000, fs=fs)
    max_gain = np.max(np.abs(H))
    return h / max_gain if max_gain != 0 else h

def apply_agc(signal, target_level=0.1, window_size=512, alpha=0.01, max_gain=10.0, attack=0.01, release=0.1):
    """
    Improved AGC with:
    - Proper gain smoothing
    - Gain limiting
    - Attack/release control
    - Better shape handling
    """
    try:
        # Ensure proper shape (N,1)
        signal = np.atleast_2d(signal).T if signal.ndim == 1 else signal
        
        # Convert time constants to samples
        attack_samples = int(attack * window_size)
        release_samples = int(release * window_size)
        
        # Calculate RMS with moving window
        squared = np.square(signal)
        window = np.ones(window_size)/window_size
        rms = np.sqrt(np.convolve(squared.squeeze(), window, mode='same'))
        
        # Calculate desired gain (with limiting)
        desired_gain = np.clip(target_level / (rms + 1e-10), 0, max_gain)
        
        # Smooth gain with different attack/release times
        smoothed_gain = np.zeros_like(desired_gain)
        smoothed_gain[0] = desired_gain[0]
        
        for i in range(1, len(desired_gain)):
            if desired_gain[i] > smoothed_gain[i-1]:
                # Attack phase
                alpha_eff = 1 - np.exp(-1.0/attack_samples)
            else:
                # Release phase
                alpha_eff = 1 - np.exp(-1.0/release_samples)
            
            smoothed_gain[i] = alpha_eff * desired_gain[i] + (1 - alpha_eff) * smoothed_gain[i-1]
        
        # Apply final limiter to prevent extreme gains
        smoothed_gain = np.clip(smoothed_gain, 0.1, max_gain)
        
        # Apply gain and maintain shape
        return signal * smoothed_gain[:, np.newaxis]
    
    except Exception as e:
        print(f"AGC error: {e}, shape={signal.shape}")
        return signal
    
def process_audio(gui):
    global filter_update_in_progress, pending_filter_update
    
    while True:
        # Initialize default output
        output = np.zeros((gui.applied_config["blocksize"], 1), dtype=np.float32)
        expected_output_len = gui.applied_config["blocksize"]
        
        try:
            start_time = time.time()
            
            # Check for pending filter updates
            if pending_filter_update and not filter_update_in_progress:
                with filter_update_lock:
                    filter_update_in_progress = True
                    gui.update_fir_filter()  # Perform the actual update
                    pending_filter_update = False
                    filter_update_in_progress = False
            
            # Get audio data with timeout
            try:
                indata = audio_queue.get(timeout=0.1)
            except queue.Empty:
                print("Input queue timeout, processing silence")
                indata = np.zeros((expected_output_len, 1), dtype=np.float32)
            
            sr, upf, bs = gui.get_dsp_config()
            source_sr = gui.get_source_sample_rate()
            upsr = sr * upf

            # Ensure proper input shape (N,1)
            indata = np.atleast_2d(indata).T if indata.ndim == 1 else indata

            # ASRC
            if source_sr != sr:
                converter = samplerate.Resampler(converter_type='sinc_best')
                indata_1d = converter.process(indata.squeeze(), sr / source_sr)
                indata_1d = np.pad(indata_1d, (0, max(0, bs - len(indata_1d))), mode='constant')[:bs]
                indata = indata_1d[:, np.newaxis]

            # Apply AGC with safe parameters
            if gui.agc_enabled.get():
                try:
                    indata = apply_agc(
                        indata,
                        target_level=float(gui.agc_target_level.get()),
                        window_size=min(512, bs//2),
                        max_gain=10.0,
                        attack=0.01,
                        release=0.1
                    )
                    indata = np.clip(indata, -0.99, 0.99)
                except Exception as e:
                    print(f"AGC error: {e}")

            # Upsample
            upsampled = safe_upsample(indata, sr, upsr)
            if len(upsampled) < bs * upf:
                upsampled = np.pad(upsampled, (0, bs * upf - len(upsampled)), mode='constant')
            elif len(upsampled) > bs * upf:
                upsampled = upsampled[:bs * upf]

            # Filter processing - single atomic copy
            with filter_lock:
                local_filter_params = active_filter_params.copy() if active_filter_params is not None else None
                local_eq_filters = active_eq_filters.copy() if active_eq_filters is not None else gui.eq_filters
                eq_gains = gui.get_gains()

            # EQ processing
            combined_eq_filter = np.zeros_like(local_eq_filters[0])
            for coeffs, gain in zip(local_eq_filters, eq_gains):
                combined_eq_filter += gain * coeffs
            eq_output = signal.oaconvolve(upsampled, combined_eq_filter, mode='same')

            # Main filter
            if local_filter_params is not None:
                final_output = signal.oaconvolve(eq_output, local_filter_params, mode='same')
            else:
                final_output = eq_output

            # Downsample
            downsampled = safe_downsample(final_output, upsr, sr, expected_output_len)
            if len(downsampled) < expected_output_len:
                downsampled = np.pad(downsampled, (0, expected_output_len - len(downsampled))), mode='constant')
            elif len(downsampled) > expected_output_len:
                downsampled = downsampled[:expected_output_len]

            # Final output processing
            downsampled = np.atleast_2d(downsampled).T if downsampled.ndim == 1 else downsampled
            output = np.clip(downsampled, -0.99, 0.99)
            
        except Exception as e:
            print(f"Processing error: {e}")
            output = np.zeros((expected_output_len, 1), dtype=np.float32)
        
        finally:
            # Ensure we always put something in the queue
            try:
                processed_queue.put(output, timeout=0.1)
                audio_queue.task_done()
                
                # Log processing time
                processing_time = time.time() - start_time
                frame_time = bs / sr
                if processing_time > frame_time:
                    print(f"Processing overrun: {processing_time*1000:.1f}ms > {frame_time*1000:.1f}ms")
                    
            except queue.Full:
                print("Output queue full, dropping frame")
            except Exception as e:
                print(f"Queue operation error: {e}")

class EqualizerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Real-Time Audio Equalizer")
        self.optimize_process()
        try:
            self.setup_variables()
            print("After setup_variables, attributes:",
                  f"samplerate={hasattr(self, 'samplerate')}, "
                  f"upsample_factor={hasattr(self, 'upsample_factor')}, "
                  f"blocksize={hasattr(self, 'blocksize')}")  # Debug print
        except Exception as e:
            print(f"Error in setup_variables: {e}")
        if not hasattr(self, 'samplerate'):
            self.samplerate = tk.StringVar(value="44100")
            print("Warning: samplerate was not initialized, using fallback")
        if not hasattr(self, 'upsample_factor'):
            self.upsample_factor = tk.StringVar(value="2")
            print("Warning: upsample_factor was not initialized, using fallback")
        if not hasattr(self, 'blocksize'):
            self.blocksize = tk.StringVar(value="2048")
            print("Warning: blocksize was not initialized, using fallback")
        self.initialize_buffers()
        self.create_controls()
        self.precompute_eq_filters()
        self.update_fir_filter()
        self.plot_response(
            self.applied_config["samplerate"] * self.applied_config["upsample_factor"],
            self.applied_config["filter_type"]
        )
        self.canvas.draw()
        for _ in range(15):
            processed_queue.put(self.silence_block.copy())
        self.processing_thread = threading.Thread(target=process_audio, args=(self,), daemon=True)
        self.processing_thread.start()

    def optimize_process(self):
        try:
            p = psutil.Process()
            os.nice(-10)
            p.cpu_affinity(list(range(psutil.cpu_count())))
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (-1, -1))
        except Exception as e:
            print(f"Process optimization warning: {e}")

    def setup_variables(self):
        self.eq_bands = [
            ("Low Bass", "20-60 Hz", (20, 60)),
            ("Mid Bass", "60-120 Hz", (60, 120)),
            ("High Bass", "120-250 Hz", (120, 250)),
            ("Low Midrange", "250-500 Hz", (250, 500)),
            ("Middle Midrange", "500-1000 Hz", (500, 1000)),
            ("High Midrange", "1000-2000 Hz", (1000, 2000)),
            ("Low Treble", "2000-4000 Hz", (2000, 4000)),
            ("Middle Treble", "4000-8000 Hz", (4000, 8000)),
            ("High Treble", "8000-16000 Hz", (8000, 16000))
        ]
        self.eq_gains = [tk.DoubleVar(value=1.0) for _ in self.eq_bands]
        self.cutoff = tk.StringVar(value="14000")
        self.cutoff_low = tk.StringVar(value="500")
        self.cutoff_high = tk.StringVar(value="15000")
        self.numtaps = tk.StringVar(value="129")
        self.window_type = tk.StringVar(value='hamming')
        self.filter_type = tk.StringVar(value='lowpass')
        self.min_phase = tk.BooleanVar(value=False)
        self.show_spectrum = tk.BooleanVar(value=False)
        self.applied_config = {
            "samplerate": 44100,
            "upsample_factor": 2,
            "blocksize": 2048,
            "cutoff": 14000,
            "numtaps": 129,
            "window_type": 'hamming',
            "filter_type": 'lowpass',
            "min_phase": False
        }
        self.samplerate = tk.StringVar(value="44100")
        self.upsample_factor = tk.StringVar(value="2")
        self.blocksize = tk.StringVar(value="2048")
        self.upsampled_rate = tk.StringVar(value="88200 Hz")  # Initial upsampled rate
        # ASRC and AGC variables
        self.source_sample_rate = tk.StringVar(value="44100")  # Default source sample rate
        self.agc_enabled = tk.BooleanVar(value=False)
        self.agc_target_level = tk.StringVar(value="0.1")  # Default RMS target

    def initialize_buffers(self):
        self.input_buffer = np.zeros(self.applied_config["blocksize"] * 4 + 512, dtype=np.float32)
        self.eq_output = np.zeros_like(self.input_buffer)
        self.audio_buffer = deque(maxlen=4)
        self.silence_block = np.zeros(self.applied_config["blocksize"], dtype=np.float32)
        self.last_output = np.zeros(self.applied_config["blocksize"], dtype=np.float32)

    def precompute_eq_filters(self):
        global inactive_eq_filters
        sr = self.applied_config["samplerate"] * self.applied_config["upsample_factor"]
        self.eq_filters = []
        for _, _, freq_range in self.eq_bands:
            coeffs = create_fir_filter(
                method='window',
                cutoff=freq_range,
                numtaps=self.applied_config["numtaps"],
                window_type='hamming',
                filter_type='bandpass',
                samplerate=sr
            )
            if self.applied_config["min_phase"] and is_symmetric(coeffs):
                coeffs = minimum_phase(coeffs, method="hilbert")
            self.eq_filters.append(coeffs)
        with filter_lock:
            inactive_eq_filters = self.eq_filters.copy()

    def create_controls(self):
        for i, (band_name, freq_range, _) in enumerate(self.eq_bands):
            tk.Label(self.master, text=band_name).grid(row=i, column=0, padx=5, pady=5, sticky="w")
            tk.Label(self.master, text=freq_range).grid(row=i, column=1, padx=(0, 5), pady=5, sticky="w")
            scale = tk.Scale(
                self.master, from_=0.0, to=2.0, resolution=0.01,
                orient="horizontal", variable=self.eq_gains[i], length=200,
                showvalue=False
            )
            scale.grid(row=i, column=2, padx=5, pady=5)
            value_label = tk.Label(self.master, textvariable=self.eq_gains[i], width=5)
            value_label.grid(row=i, column=3, padx=5, pady=5, sticky="w")

        param_start_row = len(self.eq_bands) + 1
        fields = [
            ("Cutoff Frequency", self.cutoff),
            ("Low Cutoff (for band)", self.cutoff_low),
            ("High Cutoff (for band)", self.cutoff_high),
            ("Taps", self.numtaps),
            ("Sample Rate", self.samplerate),
            ("Source Sample Rate", self.source_sample_rate),  # Added for ASRC
            ("Upsample Factor (max 4)", self.upsample_factor),
            ("Block Size", self.blocksize),
            ("AGC Target Level", self.agc_target_level)  # Added for AGC
        ]
        for idx, (label, var) in enumerate(fields):
            ttk.Label(self.master, text=label).grid(row=param_start_row + idx, column=0, padx=5, pady=2, sticky="w")
            tk.Entry(self.master, textvariable=var).grid(row=param_start_row + idx, column=1, columnspan=2, padx=5, pady=2, sticky="we")

        row = param_start_row + len(fields)
        ttk.Label(self.master, text="Window").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(self.master, textvariable=self.window_type,
                     values=['boxcar', 'triang', 'blackman', 'hamming', 'hann',
                             'bartlett', 'flattop', 'parzen', 'bohman',
                             'blackmanharris', 'nuttall', 'barthann']).grid(row=row, column=1, columnspan=2, padx=5, pady=2, sticky="we")

        row += 1
        ttk.Label(self.master, text="Filter Type").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(self.master, textvariable=self.filter_type,
                     values=['lowpass', 'highpass', 'bandpass', 'bandstop']).grid(row=row, column=1, columnspan=2, padx=5, pady=2, sticky="we")

        row += 1
        ttk.Checkbutton(self.master, text="Show Spectrum", variable=self.show_spectrum).grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)

        row += 1
        ttk.Checkbutton(self.master, text="Minimum Phase Filter", variable=self.min_phase).grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)

        row += 1
        ttk.Checkbutton(self.master, text="Enable AGC", variable=self.agc_enabled).grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)

        row += 1
        ttk.Label(self.master, text="Upsampled Rate:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(self.master, textvariable=self.upsampled_rate).grid(row=row, column=1, columnspan=2, sticky="w", padx=5, pady=2)

        row += 1
        ttk.Button(self.master, text="Apply Settings", command=self.apply_changes).grid(row=row, column=0, columnspan=1, padx=5, pady=10)
        ttk.Button(self.master, text="Reset to Defaults", command=self.reset_to_defaults).grid(row=row, column=1, columnspan=1, padx=5, pady=10)

        self.figure = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().grid(row=0, column=4, rowspan=row+1, padx=10, pady=5)

    def apply_changes(self):
        global pending_filter_update
        
        try:
            # Calculate new parameters without locking audio thread
            upf = min(int(self.upsample_factor.get()), 4)
            ftype = self.filter_type.get()
            
            if ftype in ["bandpass", "bandstop"]:
                cutoff = [float(self.cutoff_low.get()), float(self.cutoff_high.get())]
            else:
                cutoff = float(self.cutoff.get())
    
            self.applied_config = {
                "samplerate": int(self.samplerate.get()),
                "upsample_factor": upf,
                "blocksize": int(self.blocksize.get()),
                "cutoff": cutoff,
                "numtaps": int(self.numtaps.get()),
                "window_type": self.window_type.get(),
                "filter_type": ftype,
                "min_phase": self.min_phase.get()
            }
            
            # Signal filter update needed
            with filter_update_lock:
                pending_filter_update = True
                
            self.precompute_eq_filters()
            self.plot_response(self.applied_config["samplerate"] * upf, ftype)
            self.upsampled_rate.set(f"{self.applied_config['samplerate'] * upf} Hz")
            
        except Exception as e:
            print(f"Error applying changes: {e}")

    def reset_to_defaults(self):
        for gain in self.eq_gains:
            gain.set(1.0)
        self.cutoff.set("14000")
        self.cutoff_low.set("500")
        self.cutoff_high.set("15000")
        self.numtaps.set("129")
        self.samplerate.set("44100")
        self.source_sample_rate.set("44100")  # Reset source sample rate
        self.upsample_factor.set("2")
        self.blocksize.set("2048")
        self.window_type.set('hamming')
        self.filter_type.set('lowpass')
        self.min_phase.set(False)
        self.show_spectrum.set(False)
        self.agc_enabled.set(False)  # Disable AGC by default
        self.agc_target_level.set("0.1")  # Reset AGC target
        self.applied_config = {
            "samplerate": 44100,
            "upsample_factor": 2,
            "blocksize": 2048,
            "cutoff": 14000,
            "numtaps": 129,
            "window_type": 'hamming',
            "filter_type": 'lowpass',
            "min_phase": False
        }
        self.precompute_eq_filters()
        self.update_fir_filter()
        self.plot_response(
            self.applied_config["samplerate"] * self.applied_config["upsample_factor"],
            self.applied_config["filter_type"]
        )
        self.upsampled_rate.set(f"{self.applied_config['samplerate'] * self.applied_config['upsample_factor']} Hz")
        self.canvas.draw()

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

    def get_source_sample_rate(self):
        try:
            return int(self.source_sample_rate.get())
        except ValueError:
            return self.applied_config["samplerate"]  # Fallback to target sample rate

    def update_fir_filter(self):
        global inactive_filter_params
        
        try:
            # Pre-calculate to minimize lock time
            samplerate = self.applied_config["samplerate"] * self.applied_config["upsample_factor"]
            config = self.get_filter_config()
            
            # Perform the heavy computation outside the lock
            fir_coeff = create_fir_filter(
                method='window',
                cutoff=config[0],
                numtaps=config[1],
                window_type=config[2],
                filter_type=config[3],
                samplerate=samplerate
            )
            
            if self.applied_config["min_phase"] and is_symmetric(fir_coeff):
                fir_coeff = minimum_phase(fir_coeff, method="hilbert")
                fir_coeff = normalize_filter(fir_coeff, samplerate)
            
            # Quick atomic update
            with filter_lock:
                inactive_filter_params = fir_coeff.copy()
                active_filter_params = inactive_filter_params
                active_eq_filters = inactive_eq_filters
                
        except Exception as e:
            print(f"Filter update error: {e}")

    def plot_response(self, fs, filter_type):
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        
        # Plot filter response
        w, h = freqz(self.fir_coeff, worN=8000, fs=fs)
        ax.semilogx(w, 20 * np.log10(np.abs(h) + 1e-6), label="Filter")
        
        if self.show_spectrum.get() and hasattr(self, 'last_output') and self.last_output is not None:
            try:
                window_type = self.applied_config["window_type"]
                signal_length = len(self.last_output)
                
                # Ensure we have enough samples
                if signal_length < 2:
                    raise ValueError("Not enough samples for FFT")
                    
                window = signal.get_window(window_type, signal_length)
                
                # Apply window and compute FFT
                windowed_signal = self.last_output.squeeze() * window
                spectrum = np.fft.rfft(windowed_signal)
                spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-6)
                
                # Calculate correct frequency bins
                freqs = np.fft.rfftfreq(signal_length, d=1.0/fs)
                freqs_scaled = freqs / self.applied_config["upsample_factor"]
                
                # Ensure matching dimensions
                if len(freqs_scaled) != len(spectrum_db):
                    min_length = min(len(freqs_scaled), len(spectrum_db))
                    freqs_scaled = freqs_scaled[:min_length]
                    spectrum_db = spectrum_db[:min_length]
                
                # Plot spectrum
                ax.semilogx(freqs_scaled, spectrum_db, color='orange', alpha=0.6, label="Spectrum")
                
                # Find and mark peak frequency
                if len(spectrum_db) > 0:
                    peak_idx = np.argmax(spectrum_db)
                    peak_freq = freqs_scaled[peak_idx]
                    peak_db = spectrum_db[peak_idx]
                    ax.plot(peak_freq, peak_db, 'ro', markersize=8)
                    ax.annotate(f'{peak_freq:.1f}Hz\n{peak_db:.1f}dB',
                                xy=(peak_freq, peak_db),
                                xytext=(10, 10),
                                textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
            except Exception as e:
                print(f"Spectrum display error: {e}")
        
        # Configure plot
        ax.set_title(f"{filter_type.capitalize()} Filter Response")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_xlim(10, fs/2)
        ax.grid(True, which="both", ls="--")
        ax.legend()
        self.canvas.draw()

def safe_upsample(data, sr, upsr):
    try:
        if upsr <= sr:  # Bypass resampling if upf=1
            return data[:, 0] if data.ndim > 1 else data
        return soxr.resample(data[:, 0] if data.ndim > 1 else data, sr, upsr, quality='HQ')
    except Exception as e:
        print(f"Upsampling error: {e}")
        return np.zeros(int(len(data) * (upsr / sr)), dtype=np.float32)

def safe_downsample(data, upsr, sr, expected_len):
    try:
        if upsr <= sr:  # Bypass resampling if upf=1
            return data[:expected_len]
        return soxr.resample(data, upsr, sr, quality='HQ')[:expected_len]
    except Exception as e:
        print(f"Downsampling error: {e}")
        return np.zeros(expected_len, dtype=np.float32)

def apply_dither(audio, bit_depth=24):
    dither = (np.random.random(len(audio)) - 0.5) * (2 / (2**bit_depth))
    return audio + dither

def start_audio_stream(gui):
    sr, _, bs = gui.get_dsp_config()
    try:
        with sd.Stream(
            samplerate=sr,
            blocksize=bs,
            channels=1,
            dtype='float32',
            latency='high',
            callback=make_audio_callback(gui),
            device=(0, 0)
        ) as stream:
            while True:
                cpu_load = stream.cpu_load
                if cpu_load > 0.8:
                    print(f"Warning: High CPU load detected: {cpu_load:.2f}")
                time.sleep(1)
    except Exception as e:
        print(f"Audio stream error: {e}")

def make_audio_callback(gui):
    def audio_callback(indata, outdata, frames, time_info, status):
        if status:
            print(f"Stream status: {status}")
        try:
            # Ensure proper input shape
            indata = np.atleast_2d(indata).T if indata.ndim == 1 else indata
            audio_queue.put_nowait(indata.copy())
           
            try:
                processed_data = processed_queue.get_nowait()
                # Ensure proper output shape
                processed_data = np.atleast_2d(processed_data).T if processed_data.ndim == 1 else processed_data
               
                if len(processed_data) < frames:
                    processed_data = np.pad(processed_data, ((0, frames - len(processed_data)), (0, 0)), mode='constant')
                elif len(processed_data) > frames:
                    processed_data = processed_data[:frames]
               
                outdata[:, 0] = apply_dither(processed_data.squeeze())
                gui.last_output = processed_data.copy()
                processed_queue.task_done()
            except queue.Empty:
                print("Processed queue empty, outputting blended audio")
                last_audio = gui.last_output[:frames] if hasattr(gui, 'last_output') else gui.silence_block[:frames]
                blended = 0.8 * last_audio.squeeze() + 0.2 * gui.silence_block[:frames]
                outdata[:, 0] = apply_dither(blended)
        except Exception as e:
            print(f"Callback error: {str(e)}")
            outdata[:, 0] = gui.last_output[:frames].squeeze() if hasattr(gui, 'last_output') else gui.silence_block[:frames]
    return audio_callback
                        
if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (-1, -1))
    os.system('sudo cpufreq-set -g performance')
    os.sched_setaffinity(0, {0})
    root = tk.Tk()
    gui = EqualizerGUI(root)
    audio_thread = threading.Thread(target=start_audio_stream, args=(gui,), daemon=True)
    audio_thread.start()
    root.mainloop()


