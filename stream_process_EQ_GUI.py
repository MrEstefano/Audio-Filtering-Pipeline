'''
  Date : 1st June 2025
  Name : Stefan Zakutanskly
  Program name: Audio filtering pipeline with Equalizer
  Description:  This project lays the foundation for professional DSP DAC behavior,
                preserving analog-style phase characteristics and clean plotting under
                upsampled conditions. Equalazer expanded on more in frequency ranges for user to adjust.
                Minimum-Phase Filtering Support via scipy.signal.minimum_phase.
                A new GUI toggle to enable/disable Frequency Spectrum display and minimum-phase behavior.
                Proper upsampling-adjusted scaling of cutoff frequencies for accurate spectral alignment.
                Main DSP method process audio goes as following:
                # --- STEP 1: Handle any stream status issues ---
                # --- STEP 2: Get EQ gain levels from GUI sliders ---
                # --- STEP 3: Get DSP configuration (sample rate, upsample factor, block size) ---
                # --- STEP 4: Upsample the input audio to match processing rate ---
                # --- STEP 5: Shift input buffer and insert the new upsampled audio ---
                # --- STEP 6: Initialize EQ output buffer ---
                # --- STEP 7: Apply EQ filters for each band (bass, mid, treble) ---
                # --- STEP 8: Apply additional FIR filter if available ---
                # --- STEP 9: Downsample back to original sampling rate ---
                # --- STEP 10: Pad with zeros if not enough frames (at stream start) ---
                # --- STEP 11: Save processed audio and output it ---
                # --- STEP 12: If no FIR filter, output silence (placeholder) ---
                # --- STEP 13: On error, log and fall back to last good buffer or silence ---

'''

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

def process_audio(gui):
    while True:
        try:
            start_time = time.time()
            indata = audio_queue.get()
            sr, upf, bs = gui.get_dsp_config()
            upsr = sr * upf
            expected_output_len = bs

            # Upsample input
            upsampled = safe_upsample(indata, sr, upsr)
            if len(upsampled) < bs * upf:
                upsampled = np.pad(upsampled, (0, bs * upf - len(upsampled)), mode='constant')
            elif len(upsampled) > bs * upf:
                upsampled = upsampled[:bs * upf]

            with filter_lock:
                local_filter_params = active_filter_params.copy() if active_filter_params is not None else None
                local_eq_filters = active_eq_filters.copy() if active_eq_filters is not None else gui.eq_filters
                eq_gains = gui.get_gains()

            # Combine EQ filters and use overlap-add convolution
            combined_eq_filter = np.zeros_like(local_eq_filters[0])
            for coeffs, gain in zip(local_eq_filters, eq_gains):
                combined_eq_filter += gain * coeffs
            eq_output = signal.oaconvolve(upsampled, combined_eq_filter, mode='same')

            if local_filter_params is not None:
                final_output = signal.oaconvolve(eq_output, local_filter_params, mode='same')
            else:
                final_output = eq_output

            # Downsample to original rate
            downsampled = safe_downsample(final_output, upsr, sr, expected_output_len)
            if len(downsampled) < expected_output_len:
                downsampled = np.pad(downsampled, (0, expected_output_len - len(downsampled)), mode='constant')
            elif len(downsampled) > expected_output_len:
                downsampled = downsampled[:expected_output_len]

            processed_queue.put(downsampled)
            audio_queue.task_done()

            # Log overruns sparingly
            processing_time = time.time() - start_time
            frame_time = bs / sr
            if processing_time > frame_time and start_time % 10 < 0.1:  # Log every ~10 seconds
                print(f"Processing overrun: {processing_time*1000:.1f}ms > {frame_time*1000:.1f}ms")
        except Exception as e:
            print(f"Audio processing error: {e}")

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
        # Fallback initializations
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
        for _ in range(15):  # Increased pre-fill
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
        for _, _, freq_range in self.eq_bands:  # Updated to unpack three elements
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
            orient="horizontal", variable=self.eq_gains[i], length=180,
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
            ("Upsample Factor (max 4)", self.upsample_factor),
            ("Block Size", self.blocksize)
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
        ttk.Label(self.master, text="Upsampled Rate:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(self.master, textvariable=self.upsampled_rate).grid(row=row, column=1, columnspan=2, sticky="we", padx=5, pady=2)
        
        row += 1
        ttk.Checkbutton(self.master, text="Show Spectrum", variable=self.show_spectrum).grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)

        row += 1
        ttk.Checkbutton(self.master, text="Minimum Phase Filter", variable=self.min_phase).grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)


        row += 1
        ttk.Button(self.master, text="Apply Settings", command=self.apply_changes).grid(row=row, column=1, columnspan=1, padx=5, pady=10)
        ttk.Button(self.master, text="Reset to Defaults", command=self.reset_to_defaults).grid(row=row, column=0, columnspan=1, padx=5, pady=10)

        self.figure = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().grid(row=0, column=4, rowspan=row+1, padx=10, pady=5)

    def apply_changes(self):
        global inactive_filter_params, inactive_eq_filters
        try:
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
            self.precompute_eq_filters()
            self.update_fir_filter()
            self.plot_response(self.applied_config["samplerate"] * upf, ftype)
            # Update upsampled rate display
            self.upsampled_rate.set(f"{self.applied_config['samplerate'] * upf} Hz")
        except Exception as e:
            print(f"Error applying changes: {e}")

    def reset_to_defaults(self):
        # Reset equalizer band gains
        for gain in self.eq_gains:
            gain.set(1.0)
        # Reset other parameters
        self.cutoff.set("14000")
        self.cutoff_low.set("500")
        self.cutoff_high.set("15000")
        self.numtaps.set("129")
        self.samplerate.set("44100")
        self.upsample_factor.set("2")
        self.blocksize.set("2048")
        self.window_type.set('hamming')
        self.filter_type.set('lowpass')
        self.min_phase.set(False)
        self.show_spectrum.set(False)
        # Update applied_config
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
        # Recompute filters and update plot
        self.precompute_eq_filters()
        self.update_fir_filter()
        self.plot_response(
            self.applied_config["samplerate"] * self.applied_config["upsample_factor"],
            self.applied_config["filter_type"]
        )
        # Update upsampled rate display
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

    def update_fir_filter(self):
        global inactive_filter_params
        try:
            samplerate = self.applied_config["samplerate"] * self.applied_config["upsample_factor"]
            config = self.get_filter_config()
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
            with filter_lock:
                inactive_filter_params = fir_coeff.copy()
                global active_filter_params
                active_filter_params = inactive_filter_params
                global active_eq_filters
                active_eq_filters = inactive_eq_filters
            self.fir_coeff = fir_coeff
        except Exception as e:
            print(f"Error updating FIR: {e}")

    def plot_response(self, fs, filter_type):
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        w, h = freqz(self.fir_coeff, worN=8000, fs=fs)
        ax.semilogx(w, 20 * np.log10(np.abs(h) + 1e-6), label="Filter")
        if self.show_spectrum.get():
            window_type = self.applied_config["window_type"]
            window = signal.get_window(window_type, len(self.last_output))
            spectrum = np.fft.rfft(self.last_output * window)
            spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-6)
            freqs = np.fft.rfftfreq(len(self.last_output), d=1/fs)
            freqs_scaled = freqs / self.applied_config["upsample_factor"]
            ax.semilogx(freqs_scaled, spectrum_db, color='orange', alpha=0.6, label="Spectrum")
            peak_idx = np.argmax(spectrum_db)
            peak_freq = freqs_scaled[peak_idx]
            peak_db = spectrum_db[peak_idx]
            ax.plot(peak_freq, peak_db, 'ro', markersize=8)
            ax.annotate(f'{peak_freq:.1f}Hz\n{peak_db:.1f}dB',
                        xy=(peak_freq, peak_db),
                        xytext=(10, 10),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
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
        return soxr.resample(data[:, 0] if data.ndim > 1 else data, sr, upsr, quality='HQ')  # Reduced quality for speed
    except Exception as e:
        print(f"Upsampling error: {e}")
        return np.zeros(int(len(data) * (upsr / sr)), dtype=np.float32)

def safe_downsample(data, upsr, sr, expected_len):
    try:
        if upsr <= sr:  # Bypass resampling if upf=1
            return data[:expected_len]
        return soxr.resample(data, upsr, sr, quality='HQ')[:expected_len]  # Reduced quality for speed
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
            audio_queue.put_nowait(indata.copy())
            processed_data = processed_queue.get_nowait()
            if len(processed_data) < frames:
                processed_data = np.pad(processed_data, (0, frames - len(processed_data)), mode='constant')
            elif len(processed_data) > frames:
                processed_data = processed_data[:frames]
            outdata[:, 0] = apply_dither(processed_data)
            gui.last_output = processed_data.copy()
            processed_queue.task_done()
        except queue.Empty:
            print("Processed queue empty, outputting blended audio")
            last_audio = gui.last_output[:frames] if hasattr(gui, 'last_output') else gui.silence_block[:frames]
            blended = 0.8 * last_audio + 0.2 * gui.silence_block[:frames]
            outdata[:, 0] = apply_dither(blended)
        except Exception as e:
            print(f"Callback error: {str(e)}")
            outdata[:, 0] = gui.last_output[:frames] if hasattr(gui, 'last_output') else gui.silence_block[:frames]
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







