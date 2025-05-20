import tkinter as tk
from tkinter import ttk
import numpy as np
import sounddevice as sd
import soxr
from scipy.signal import oaconvolve
from collections import deque
import os
import resource
from fir_filter import create_fir_filter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from plot_filter import plot_filter_response

class EqualizerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Real-Time Audio Equalizer")

        # EQ gain sliders
        self.bass_gain = tk.DoubleVar(value=1.0)
        self.mid_gain = tk.DoubleVar(value=1.0)
        self.treble_gain = tk.DoubleVar(value=1.0)

        # Filter parameters
        self.cutoff = tk.StringVar(value="16000")
        self.numtaps = tk.StringVar(value="301")
        self.window_type = tk.StringVar(value='hamming')
        self.filter_type = tk.StringVar(value='lowpass')

        # Global DSP settings
        self.samplerate = tk.StringVar(value="44100")
        self.upsample_factor = tk.StringVar(value="4")
        self.blocksize = tk.StringVar(value="4096")

        self.applied_config = {
            "samplerate": 44100,
            "upsample_factor": 4,
            "blocksize": 4096,
            "cutoff": 16000,
            "numtaps": 301,
            "window_type": 'hamming',
            "filter_type": 'lowpass'
        }

        self.fir_coeff = None
        self.canvas = None
        self.figure = None

        self.create_controls()
        self.update_fir_filter()

    def create_controls(self):
        ttk.Label(self.master, text="Bass").grid(row=0, column=0)
        ttk.Scale(self.master, from_=0.0, to=2.0, variable=self.bass_gain).grid(row=0, column=1)

        ttk.Label(self.master, text="Mid").grid(row=1, column=0)
        ttk.Scale(self.master, from_=0.0, to=2.0, variable=self.mid_gain).grid(row=1, column=1)

        ttk.Label(self.master, text="Treble").grid(row=2, column=0)
        ttk.Scale(self.master, from_=0.0, to=2.0, variable=self.treble_gain).grid(row=2, column=1)

        fields = [
            ("Cutoff Frequency", self.cutoff),
            ("Taps", self.numtaps),
            ("Sample Rate", self.samplerate),
            ("Upsample Factor", self.upsample_factor),
            ("Block Size", self.blocksize)
        ]

        for idx, (label, var) in enumerate(fields, start=3):
            ttk.Label(self.master, text=label).grid(row=idx, column=0)
            tk.Entry(self.master, textvariable=var).grid(row=idx, column=1)

        ttk.Label(self.master, text="Window").grid(row=8, column=0)
        ttk.Combobox(self.master, textvariable=self.window_type, values=['hamming', 'hann', 'blackman', 'nuttall']).grid(row=8, column=1)

        ttk.Label(self.master, text="Filter Type").grid(row=9, column=0)
        ttk.Combobox(self.master, textvariable=self.filter_type, values=['lowpass', 'highpass', 'bandpass', 'bandstop']).grid(row=9, column=1)

        ttk.Button(self.master, text="Apply Settings", command=self.apply_changes).grid(row=10, columnspan=2, pady=10)

        # Matplotlib plot
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().grid(row=0, column=2, rowspan=11)

    def apply_changes(self):
        try:
            self.applied_config = {
                "samplerate": int(self.samplerate.get()),
                "upsample_factor": int(self.upsample_factor.get()),
                "blocksize": int(self.blocksize.get()),
                "cutoff": float(self.cutoff.get()),
                "numtaps": int(self.numtaps.get()),
                "window_type": self.window_type.get(),
                "filter_type": self.filter_type.get()
            }
            self.update_fir_filter()
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
            self.fir_coeff = create_fir_filter(
                method='window',
                cutoff=config[0],
                numtaps=config[1],
                window_type=config[2],
                filter_type=config[3],
                samplerate=samplerate
            )
            self.plot_response(samplerate, config[3])
        except Exception as e:
            print(f"Error updating FIR: {e}")
        return self.fir_coeff

    def plot_response(self, fs, filter_type):
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        from scipy.signal import freqz
        w, h = freqz(self.fir_coeff, worN=8000, fs=fs)
        ax.plot(w, 20 * np.log10(np.abs(h) + 1e-6))
        ax.set_title(f"{filter_type.capitalize()} Filter Response")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.grid(True)
        self.canvas.draw()

# System setup
resource.setrlimit(resource.RLIMIT_MEMLOCK, (-1, -1))
os.system('sudo cpufreq-set -g performance')
os.sched_setaffinity(0, {0})

input_buffer = np.zeros(4096 * 4 + 1000, dtype=np.float32)
audio_buffer = deque(maxlen=4)
silence_block = np.zeros(4096, dtype=np.float32)

EQ_BANDS = [((60, 250)), ((500, 2000)), ((4000, 16000))]

def apply_dither(audio, bit_depth=24):
    dither = (np.random.random(len(audio)) - 0.5) * (2 / (2**bit_depth))
    return audio + dither

def safe_upsample(data, sr, upsr):
    try:
        return soxr.resample(data[:, 0] if data.ndim > 1 else data, sr, upsr, quality='VHQ')
    except:
        return np.zeros(4096 * 4, dtype=np.float32)

def make_audio_callback(gui):
    def audio_callback(indata, outdata, frames, time, status):
        global input_buffer

        bass, mid, treble = gui.get_gains()
        eq_gains = [bass, mid, treble]

        try:
            sr, upf, bs = gui.get_dsp_config()
        except Exception as e:
            print(f"Invalid DSP config: {e}")
            outdata[:, 0] = silence_block[:frames]
            return

        upsr = sr * upf

        if status:
            print(f"Stream status: {status}")

        try:
            upsampled = safe_upsample(indata, sr, upsr)
            if len(upsampled) != bs * upf:
                upsampled = np.resize(upsampled, bs * upf)

            input_buffer[:-len(upsampled)] = input_buffer[len(upsampled):]
            input_buffer[-len(upsampled):] = upsampled

            eq_output = np.zeros_like(input_buffer)
            for i, cutoff in enumerate(EQ_BANDS):
                coeffs = create_fir_filter(
                    method='window',
                    cutoff=cutoff,
                    numtaps=301,
                    window_type='hamming',
                    filter_type='bandpass',
                    samplerate=upsr
                )
                band = oaconvolve(input_buffer, coeffs, mode='same')
                eq_output += eq_gains[i] * band

            fir_coeff = gui.fir_coeff
            if fir_coeff is None:
                outdata[:, 0] = silence_block[:frames]
                return

            final_output = oaconvolve(eq_output, fir_coeff, mode='valid', axes=0)
            downsampled = final_output[::upf][:frames]
            if len(downsampled) < frames:
                padded = np.zeros(frames, dtype=np.float32)
                padded[:len(downsampled)] = downsampled
                downsampled = padded

            audio_buffer.append(downsampled)
            outdata[:, 0] = apply_dither(downsampled)

        except Exception as e:
            print(f"Processing error: {str(e)}")
            if audio_buffer:
                outdata[:, 0] = audio_buffer[-1][:frames]
            else:
                outdata[:, 0] = silence_block[:frames]

    return audio_callback

if __name__ == '__main__':
    root = tk.Tk()
    gui = EqualizerGUI(root)
    try:
        sr, upf, bs = gui.get_dsp_config()
        with sd.Stream(
            samplerate=sr,
            blocksize=bs,
            channels=1,
            dtype='float32',
            latency='high',
            callback=make_audio_callback(gui),
            device=(1, 0)
        ):
            root.mainloop()
    except Exception as e:
        print(f"Startup error: {e}")
