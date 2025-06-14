'''
  Date : 28th May 2025
  Name : Stefan Zakutanskly
  Program name: Audio filtering pipeline with Equalizer
  Description:  This project lays the foundation for professional DSP DAC behavior,
                preserving analog-style phase characteristics and clean plotting under
                upsampled conditions. Equalazer expanded on more in frequency ranges for user to adjust.
                Minimum-Phase Filtering Support via scipy.signal.minimum_phase.
                A new GUI toggle to enable/disable Frequency Spectrum display and minimum-phase behavior.
                Proper upsampling-adjusted scaling of cutoff frequencies for accurate spectral alignment.
                Main Audio Callback process goes as following:
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
from scipy.signal import fftconvolve, freqz, minimum_phase
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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
        # --- Set window size (Width x Height) ---
        #master.geometry("1600x1300")  # You can adjust this as needed
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
            os.nice(-10)  # Higher priority
            p.cpu_affinity(list(range(psutil.cpu_count())))
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (-1, -1))
        except Exception as e:
            print(f"Process optimization warning: {e}")

    def setup_variables(self):
        # Define named EQ bands and their frequency ranges
        self.eq_bands = [
            ("Low Bass            20-60 Hz", (20, 60)),
            ("Mid Bass            60-120 Hz", (60, 120)),
            ("High Bass          120-250 Hz", (120, 250)),
            ("Low Midrange      250-500 Hz", (250, 500)),
            ("Middle Midrange  500-1000Hz", (500, 1000)),
            ("High Midrange  1000-2000 Hz", (1000, 2000)),
            ("Low Treble      2000-4000 Hz", (2000, 4000)),
            ("Middle Treble   4000-8000 Hz", (4000, 8000)),
            ("High Treble     8000-16000 Hz", (8000, 16000))
        ]

        # One DoubleVar per band (initialized to 1.0)
        self.eq_gains = [tk.DoubleVar(value=1.0) for _ in self.eq_bands]
        #self.eq_gains = [tk.DoubleVar() for _ in range(10)]
        self.cutoff = tk.StringVar(value="14000")
        self.cutoff_low = tk.StringVar(value="500")
        self.cutoff_high = tk.StringVar(value="15000")
        self.numtaps = tk.StringVar(value="257")
        self.window_type = tk.StringVar(value='hamming')
        self.filter_type = tk.StringVar(value='lowpass')
        self.min_phase = tk.BooleanVar(value=False)
        self.show_waterfall = tk.BooleanVar(value=False)
        self.waterfall_data = deque(maxlen=50)
        self.samplerate = tk.StringVar(value="44100")
        self.upsample_factor = tk.StringVar(value="1")
        self.blocksize = tk.StringVar(value="1024")

        self.show_spectrum = tk.BooleanVar(value=False)

        self.applied_config = {
            "samplerate": 44100,
            "upsample_factor": 1,
            "blocksize": 1024,
            "cutoff": 14000,
            "numtaps": 257,
            "window_type": 'hamming',
            "filter_type": 'lowpass',
            "min_phase": False
        }

    def initialize_buffers(self):

        self.input_buffer = np.zeros(self.applied_config["blocksize"] * 4 + 512, dtype=np.float32)
        self.eq_output = np.zeros_like(self.input_buffer)
        self.audio_buffer = deque(maxlen=4)
        self.silence_block = np.zeros(self.applied_config["blocksize"], dtype=np.float32)
        self.last_output = np.zeros(self.applied_config["blocksize"], dtype=np.float32)

    def precompute_eq_filters(self):
        self.eq_filters = []
        sr = self.applied_config["samplerate"] * self.applied_config["upsample_factor"]
        #self.eq_bands = []
        for i, (label, freq_range) in enumerate(self.eq_bands):
            coeffs = create_fir_filter(
                method='window',
                cutoff=freq_range,
                numtaps=self.applied_config["numtaps"],
                window_type='hamming',
                filter_type='bandpass',
                samplerate = sr
            )
            if self.applied_config["min_phase"] and is_symmetric(coeffs):
                coeffs = minimum_phase(coeffs, method="hilbert")
            self.eq_filters.append(coeffs)

    def create_controls(self):
        for i, (label, freq_range) in enumerate(self.eq_bands):
            # Label for each EQ band
            tk.Label(self.master, text=label).grid(row=i, column=0, padx=5, pady=5, sticky="w")

            # Slider
            scale = tk.Scale(
                self.master, from_=0.0, to=2.0, resolution=0.01,
                orient="horizontal", variable=self.eq_gains[i], length=200,
                showvalue=False
            )
            scale.grid(row=i, column=1, padx=5, pady=5)

            # Value display on the right (moved to column 3)
            value_label = tk.Label(self.master, textvariable=self.eq_gains[i], width=5)
            value_label.grid(row=i, column=3, padx=5, pady=5, sticky="w")

        # Now start other parameter controls at a row lower than the EQ bands
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

        # Dropdowns and buttons
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
        ttk.Checkbutton(self.master, text="Waterfall Display", variable=self.show_waterfall).grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        # Place the Matplotlib plot to the right of all controls
        self.figure = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().grid(row=0, column=4, rowspan=row+1, padx=10, pady=5)

        
        

    def apply_changes(self):
        try:
            # --- STEP 1: Safely parse the upsample factor, limiting to 4 ---
            upf = min(int(self.upsample_factor.get()), 4)

            # --- STEP 2: Get filter type (e.g.,            if gui.show_waterfall.get() and len(filtered) > 0:
                
            ftype = self.filter_type.get()

            # --- STEP 3: Parse cutoff frequency/frequencies depending on filter type ---
            if ftype in ["bandpass", "bandstop"]:
                # For bandpass or bandstop, get both low and high cutoff values
                cutoff = [float(self.cutoff_low.get()), float(self.cutoff_high.get())]
            else:
                # For lowpass/highpass, get a single cutoff value
                cutoff = float(self.cutoff.get())

            # --- STEP 4: Store all updated parameters into applied_config dictionary ---
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

            # --- STEP 5: Recalculate EQ filter coefficients (for bass, mid, treble bands) ---
            self.precompute_eq_filters()

            # --- STEP 6: Update the FIR filter preview/plot with new settings ---
            self.update_fir_filter()

            # --- STEP 7: Plot the frequency response of the new filter config ---
            self.plot_response(self.applied_config["samplerate"] * upf, ftype)

        except Exception as e:
            # --- STEP 8: Catch and report GUI input errors (bad values, etc.) ---
            print(f"Error applying changes: {e}")
            
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
                
            #return fir_coeff 
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
            spectra_lengths = [len(spec) for spec in self.waterfall_data]
            target_length = max(set(spectra_lengths), key=spectra_lengths.count)
            self.waterfall_data = deque([spec[:target_length] for spec in self.waterfall_data],
                                      maxlen=self.waterfall_data.maxlen)
            spectra = np.array(self.waterfall_data)
            times = np.arange(len(self.waterfall_data)) * (self.applied_config["blocksize"]/self.applied_config["samplerate"])
            fft_size = (len(self.waterfall_data[0]) - 1) * 2
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
        
        print("fs:", self.applied_config["samplerate"], "x", self.applied_config["upsample_factor"], "=", fs)

        w, h = freqz(self.fir_coeff, worN=8000, fs=fs)
        # w*self.applied_config["upsample_factor"]        
        ax.plot(w, 20 * np.log10(np.abs(h) + 1e-6), label="Filter")
        if self.show_spectrum.get():
            window_type = self.applied_config["window_type"]
            # Map window type to NumPy function
            window = signal.get_window(window_type, len(self.last_output))
            #window_func = getattr(np, window_type, np.hanning)  # Default to hanning if invalid
            spectrum = np.fft.rfft(self.last_output * window)
            spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-6)
            new_fs= fs/self.applied_config["upsample_factor"]
            freqs = np.fft.rfftfreq(len(self.last_output), d=1/new_fs)
            ax.plot(freqs, spectrum_db, color='orange', alpha=0.6, label="Spectrum") 
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
        ax.set_title(f"{filter_type.capitalize()} Filter Response")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.grid(True)
        #plt.xticks(np.arange(0, fs / 2 + 1, 2000))  # Adjust spacing to fit your needs
        ax.legend()
        self.canvas.draw()



def safe_upsample(data, sr, upsr):
    '''
    Upsampling with polyphase filtering. Anti-imaging filter design (low-pass interpolation) based on the resampling ratio.
       - The quality='VHQ' (Very High Quality) mode means it:
       - Uses a steep, well-designed FIR filter.
       - Minimizes aliasing and spectral distortion.
    '''
    try:
        # soxr.resample() Provides filtering is automatically tailored to the input and
        # output sample rates to avoid aliasing and imaging
        return soxr.resample(data[:, 0] if data.ndim > 1 else data, sr, upsr, quality='VHQ')
    except Exception as e:
        print(f"Resampling error: {e}")
        return np.zeros(len(data) * (upsr // sr), dtype=np.float32)

def apply_dither(audio, bit_depth=24):
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
            device=(0, 0)  # Replace with your actual device indices
        ):
            threading.Event().wait()  # Keeps the thread alive
    except Exception as e:
        print(f"Audio stream error: {e}")
        
def make_audio_callback(gui):
    def audio_callback(indata, outdata, frames, time_info, status):

        # --- STEP 1: Handle any stream status issues ---
        if status:
            print(f"Stream status: {status}")

        try:
            start_time = time.time()  # Use system time for measurement
            # --- STEP 2: Get EQ gain levels from GUI sliders ---
            eq_gains = gui.get_gains() 

            # --- STEP 3: Get DSP configuration (sample rate, upsample factor, block size) ---
            sr, upf, bs = gui.get_dsp_config()
            upsr = sr * upf  # Effective sample rate after upsampling

            # --- STEP 4: Upsample the input audio to match processing rate ---
            upsampled = safe_upsample(indata, sr, upsr)

            # --- STEP 5: Shift input buffer and insert the new upsampled audio ---
            gui.input_buffer[:-len(upsampled)] = gui.input_buffer[len(upsampled):]
            gui.input_buffer[-len(upsampled):] = upsampled

            # --- STEP 6: Initialize EQ output buffer ---
            gui.eq_output.fill(0)

            # --- STEP 7: Apply EQ filters for each band (bass, mid, treble) ---
            for coeffs, gain_var in zip(gui.eq_filters, gui.eq_gains):
                gain = gain_var.get()
                band = fftconvolve(gui.input_buffer, coeffs, mode='same')
                gui.eq_output += gain * band

            # --- STEP 8: Apply additional FIR filter if available ---
            if gui.fir_coeff is not None:
 
                final_output = fftconvolve(gui.eq_output, gui.fir_coeff, mode='valid')

                # --- STEP 9: Downsample back to original sampling rate ---
                downsampled = final_output[::upf][:frames]

                # --- STEP 10: Pad with zeros if not enough frames (at stream start) ---
                                # Frame size handling
                if len(downsampled) < frames:
                    downsampled = np.pad(downsampled, (0, frames - len(downsampled)))
                elif len(downsampled) > frames:
                    downsampled = downsampled[:frames]

                # --- STEP 11: Save processed audio and output it ---
                gui.audio_buffer.append(downsampled)
                outdata[:, 0] = apply_dither(downsampled)
                gui.last_output = downsampled.copy()

            else:
                # --- If no FIR filter, output silence (placeholder) ---
                outdata[:, 0] = gui.silence_block[:frames]
                 
            if gui.show_waterfall.get():
                try:
                    windowed = outdata[:, 0] * np.hanning(len(outdata[:, 0]))
                    spectrum = np.fft.rfft(windowed)
                    spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-12)
                    gui.waterfall_data.append(spectrum_db)
                except Exception as e:
                    print(f"Waterfall error: {str(e)}") 
            processing_time = time.time() - start_time
            frame_time = frames / sr
            if processing_time > frame_time:
                print(f"Overrun: {processing_time*1000:.1f}ms > {frame_time*1000:.1f}ms")

            
        except Exception as e:
            # --- On error, log and fall back to last good buffer or silence ---
            print(f"Processing error: {str(e)}")
            outdata[:, 0] = gui.last_output[:frames] if hasattr(gui, 'last_output') else gui.silence_block[:frames]

    return audio_callback


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (-1, -1))
    os.system('sudo cpufreq-set -g performance')
    os.sched_setaffinity(0, {0})
    root = tk.Tk()
    
    gui = EqualizerGUI(root)
    
        # Start the audio stream in a separate thread
    audio_thread = threading.Thread(target=start_audio_stream, args=(gui,), daemon=True)
    audio_thread.start()

    # Start the GUI main loop
    root.mainloop()

