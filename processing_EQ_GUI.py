
'''
- Minimum-Phase Filtering Support via scipy.signal.minimum_phase.
- A new GUI toggle to enable/disable minimum-phase behavior.
- Proper upsampling-adjusted scaling of cutoff frequencies for accurate spectral alignment.

This lays the foundation for professional DSP DAC behavior,
preserving analog-style phase characteristics and
clean plotting under upsampled conditions.

'''

import tkinter as tk
from tkinter import ttk
import numpy as np
import sounddevice as sd
import soxr
from scipy.signal import fftconvolve, freqz, minimum_phase
from collections import deque
import os
import resource
import psutil
from fir_filter import create_fir_filter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

silence_block = np.zeros(65536, dtype=np.float32)
audio_buffer = deque(maxlen=4)
input_buffer = np.zeros(65536, dtype=np.float32)

def is_symmetric(h, tol=1e-8):
    return np.allclose(h, h[::-1], atol=tol)

def normalize_filter(h, fs):
    """Normalize filter gain to 0 dB at DC or center of band."""
    w, H = freqz(h, worN=8000, fs=fs)
    max_gain = np.max(np.abs(H))
    return h / max_gain

class EqualizerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Real-Time Audio Equalizer")

        self.plot_counter = 0
        self.plot_interval = 5
        self.optimize_process()
        self.setup_variables()
        self.initialize_buffers()
        self.create_controls()
        self.precompute_eq_filters()
        self.update_fir_filter()
        self.plot_response(
            self.applied_config["samplerate"] * self.applied_config["upsample_factor"],
            self.applied_config["filter_type"]
        )
        self.canvas.draw()

    def optimize_process(self):
        try:
            p = psutil.Process()
            os.nice(-10)
            p.cpu_affinity(list(range(psutil.cpu_count())))
        except Exception as e:
            print(f"Couldn't optimize process: {e}")

    def setup_variables(self):
        self.bass_gain = tk.DoubleVar(value=1.0)
        self.mid_gain = tk.DoubleVar(value=1.0)
        self.treble_gain = tk.DoubleVar(value=1.0)

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
        max_block_size = 4096
        max_upsample_factor = 4

        self.input_buffer = np.zeros(max_block_size * max_upsample_factor + 1000, dtype=np.float32)
        self.eq_output = np.zeros_like(self.input_buffer)
        self.audio_buffer = deque(maxlen=4)
        self.silence_block = np.zeros(max_block_size, dtype=np.float32)
        self.last_output = np.zeros(max_block_size, dtype=np.float32)

    def precompute_eq_filters(self):
        self.eq_filters = []
        sr = self.applied_config["samplerate"] * self.applied_config["upsample_factor"]
        self.EQ_BANDS = [((60, 250)), ((500, 2000)), ((4000, 16000))]
        for cutoff in self.EQ_BANDS:
            coeffs = create_fir_filter(
                method='window',
                cutoff=cutoff,
                numtaps=self.applied_config["numtaps"],
                window_type='hamming',
                filter_type='bandpass',
                samplerate=sr
            )
            if self.applied_config["min_phase"] and is_symmetric(coeffs):
                coeffs = minimum_phase(coeffs, method="hilbert")
            self.eq_filters.append(coeffs)

    def create_controls(self):
        ttk.Label(self.master, text="Bass").grid(row=0, column=0)
        ttk.Scale(self.master, from_=0.0, to=2.0, variable=self.bass_gain).grid(row=0, column=1)
        ttk.Label(self.master, text="Mid").grid(row=1, column=0)
        ttk.Scale(self.master, from_=0.0, to=2.0, variable=self.mid_gain).grid(row=1, column=1)
        ttk.Label(self.master, text="Treble").grid(row=2, column=0)
        ttk.Scale(self.master, from_=0.0, to=2.0, variable=self.treble_gain).grid(row=2, column=1)

        fields = [
            ("Cutoff Frequency", self.cutoff),
            ("Low Cutoff (for band)", self.cutoff_low),
            ("High Cutoff (for band)", self.cutoff_high),
            ("Taps", self.numtaps),
            ("Sample Rate", self.samplerate),
            ("Upsample Factor (max 4)", self.upsample_factor),
            ("Block Size", self.blocksize)
        ]

        for idx, (label, var) in enumerate(fields, start=3):
            ttk.Label(self.master, text=label).grid(row=idx, column=0)
            tk.Entry(self.master, textvariable=var).grid(row=idx, column=1)

        ttk.Label(self.master, text="Window").grid(row=10, column=0)
        ttk.Combobox(self.master, textvariable=self.window_type,
                     values=['hamming', 'hann', 'blackman', 'nuttall']).grid(row=10, column=1)
        ttk.Label(self.master, text="Filter Type").grid(row=11, column=0)
        ttk.Combobox(self.master, textvariable=self.filter_type,
                     values=['lowpass', 'highpass', 'bandpass', 'bandstop']).grid(row=11, column=1)
        ttk.Checkbutton(self.master, text="Show Spectrum", variable=self.show_spectrum).grid(row=12, column=0, columnspan=2)
        ttk.Checkbutton(self.master, text="Minimum Phase Filter", variable=self.min_phase).grid(row=13, column=0, columnspan=2)
        ttk.Button(self.master, text="Apply Settings", command=self.apply_changes).grid(row=14, columnspan=2, pady=10)

        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().grid(row=0, column=2, rowspan=15)
        
        

    def apply_changes(self):
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
        except Exception as e:
            print(f"Error applying changes: {e}")

    def get_gains(self):
        return self.bass_gain.get(), self.mid_gain.get(), self.treble_gain.get()

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
            samplerate = self.applied_config["samplerate"] * self.applied_config["upsample_factor"]
            config = self.get_filter_config()
            cutoff = config[0]
            self.fir_coeff = create_fir_filter(
                method='window',
                cutoff=cutoff,
                numtaps=config[1],
                window_type=config[2],
                filter_type=config[3],
                samplerate=samplerate
            )
            if self.applied_config["min_phase"] and is_symmetric(self.fir_coeff):
                self.fir_coeff = minimum_phase(self.fir_coeff, method="hilbert")
                self.fir_coeff = normalize_filter(self.fir_coeff, samplerate)

        except Exception as e:
            print(f"Error updating FIR: {e}")

    def plot_response(self, fs, filter_type):
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        w, h = freqz(self.fir_coeff, worN=8000, fs=fs)
        ax.plot(w, 20 * np.log10(np.abs(h) + 1e-6), label="Filter")
        if self.show_spectrum.get():
            spectrum = np.fft.rfft(self.last_output * np.hanning(len(self.last_output)))
            spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-6)
            freqs = np.fft.rfftfreq(len(self.last_output), d=1/fs)
            ax.plot(freqs, spectrum_db, color='orange', alpha=0.6, label="Spectrum")
        ax.set_title(f"{filter_type.capitalize()} Filter Response")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.grid(True)
        ax.legend()
        self.canvas.draw()

def safe_upsample(data, sr, upsr):
    try:
        return soxr.resample(data[:, 0] if data.ndim > 1 else data, sr, upsr, quality='VHQ')
    except Exception as e:
        print(f"Resampling error: {e}")
        return np.zeros(len(data) * (upsr // sr), dtype=np.float32)

def apply_dither(audio, bit_depth=24):
    dither = (np.random.random(len(audio)) - 0.5) * (2 / (2**bit_depth))
    return audio + dither

def make_audio_callback(gui):
    def audio_callback(indata, outdata, frames, time, status):
        if status:
            print(f"Stream status: {status}")
        try:
            bass, mid, treble = gui.get_gains()
            eq_gains = [bass, mid, treble]
            sr, upf, bs = gui.get_dsp_config()
            upsr = sr * upf
            upsampled = safe_upsample(indata, sr, upsr)
            gui.input_buffer[:-len(upsampled)] = gui.input_buffer[len(upsampled):]
            gui.input_buffer[-len(upsampled):] = upsampled
            gui.eq_output.fill(0)
            for i, (coeffs, gain) in enumerate(zip(gui.eq_filters, eq_gains)):
                band = fftconvolve(gui.input_buffer, coeffs, mode='same')
                gui.eq_output += gain * band
            if gui.fir_coeff is not None:
                final_output = fftconvolve(gui.eq_output, gui.fir_coeff, mode='valid')
                downsampled = final_output[::upf][:frames]
                if len(downsampled) < frames:
                    downsampled = np.pad(downsampled, (0, frames - len(downsampled)))
                gui.audio_buffer.append(downsampled)
                outdata[:, 0] = apply_dither(downsampled)
                gui.last_output = downsampled.copy()
            else:
                outdata[:, 0] = gui.silence_block[:frames]
        except Exception as e:
            print(f"Processing error: {str(e)}")
            if gui.audio_buffer:
                outdata[:, 0] = gui.audio_buffer[-1][:frames]
            else:
                outdata[:, 0] = gui.silence_block[:frames]
    return audio_callback

if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (-1, -1))
    os.system('sudo cpufreq-set -g performance')
    os.sched_setaffinity(0, {0})
    root = tk.Tk()
    gui = EqualizerGUI(root)
    try:
        sr, upf, bs = gui.get_dsp_config()
        with sd.Stream(
            samplerate=sr,
            blocksize=bs,
            channels=1,
            dtype='float32',
            latency='low',
            callback=make_audio_callback(gui),
            device=(1, 0)
        ):
            root.mainloop()
    except Exception as e:
        print(f"Startup error: {e}")

	

