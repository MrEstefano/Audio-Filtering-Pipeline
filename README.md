# Real-time Audio Digital Signal Procesing DAC with FIR Filters, windows, upsampling

![DSP Pipeline](https://img.shields.io/badge/Realtime-DSP_Processing-blue) 
![Raspberry Pi](https://img.shields.io/badge/Hardware-RPi_RPi_5_PCM5102a-green)
![Hifiberry](https://img.shields.io/badge/Hardware-DAC2_ADC_Pro-pink)
![Python](https://img.shields.io/badge/Python-3.7%2B-yellow)

A real-time audio processing (DSP) system implementing FIR filters, windowing, upsampling on Raspberry Pi 5 with USB soundcard input, I2S DAC output or alternatively Hifiberry DAC2 ADC PRO hat, featuring customizable filter design and visualization.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation & Setup](#installation--setup)
- [Hardware Configuration](#hardware-configuration)
- [Software Architecture](#software-architecture)
- [Usage Examples](#usage-examples)
- [Visualization](#visualization)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Project Overview

This DSP pipeline provides:
- Real-time FIR filtering (lowpass/highpass/bandpass/bandstop)
- Multiple windowing methods (Hamming, Blackman, Kaiser, Nuttall, etc)
- 1x-4x upsampling capabilities
- Interactive filter response and Freq. Spectrum visualization
- Full-spectrum audio equalazir
- Minimum phase filtering

**Key Features**:
- Sample rates from 44.1kHz to 176.4kHz
- Adjustable filter lengths (51-1001 taps)
- Sizeable buffer block 
- Both windowed and Remez exchange algorithms
- Triangular Probability Density Function (TPDF) dithering for improved dynamic range

## Installation & Setup
In your raspberry system bash the following commands, alternatively ssh via PC terminal domain
### 1. Clone Repository
```bash
git clone https://github.com/yourusername/Real-time-DSP-DAC.git
cd Real-time-DSP-DAC
```
### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
**Core Packages**:
- `numpy` - FIR math and array operations
- `sounddevice` - Low-latency audio I/O
- `scipy` - Signal processing functions
- `matplotlib` - Filter visualization
- `soxr` - High-quality resampling

# Hardware Configuration
### PCM5102a+USB Soundcard System Connections
Audio input is connected to USB Soundcard 3.5 mm Jack, Output from DAC is fed to amplifier, then to speakers

| PCM5102 Pin | RPi Zero Pin | GPIO |
|-------------|--------------|------|
| VIN (5V)    | Pin 2        | -    |
| GND         | Pin 6        | -    |
| LCK (LRCK)  | Pin 35       | 19   |
| DIN (DATA)  | Pin 40       | 21   |
| BCK (BCLK)  | Pin 12       | 18   |
| SCK         | GND          | -    |

![IMG_1001](https://github.com/user-attachments/assets/8f219b7e-44ed-46b0-8e0f-c663d9f67b66)

### USB Soundcard
https://www.aliexpress.com/item/1005003192869006.html
### PCM5102a Module
https://www.aliexpress.com/item/1005006104368969.html

### PCM5102a+USB Soundcard System Configuration
Enable I2S in /boot/firmware/config.txt:
```bash
sudo nano /boot/firmware/config.txt
```
Nano GUI:
```ini
dtparam=i2s=on
#dtparam=audio=on
dtoverlay=hifiberry-dac
```
Configure ALSA (/etc/asound.conf):
```bash
sudo nano /etc/asound.conf
```
```conf
pcm.!default {
  type hw
  card 0
}
ctl.!default {
  type hw
  card 0
}
```
Reboot and verify:
```bash
sudo reboot
```
Verify setup 
```bash
aplay -l # Should show HiFiBerry DAC
```
### Hifiberry DAC2 ADC PRO Hat setup
- Link to hardware: https://www.hifiberry.com/shop/boards/dac2adcpro/
- Datasheet: https://www.hifiberry.com/docs/data-sheets/datasheet-dac2-adc-pro/

![IMG_1608](https://github.com/user-attachments/assets/c878ff19-b64a-4717-92c3-71679ce20842)
In order to place the HAT on the top of RPi with heatrsink, the header extention is required.
### Hifiberry DAC2 ADC PRO System Configuration
Enable I2S in /boot/firmware/config.txt:

```bash
sudo nano /boot/firmware/config.txt
```
```ini
dtparam=i2s=on
#dtparam=audio=on
dtoverlay=hifiberry-dacplusadcpro
```
Configure ALSA (/etc/asound.conf):

```bash
sudo nano /etc/asound.conf
```

```conf
defaults.pcm.card 0
defaults.ctl.card 0
```

Reboot and verify:
```bash
sudo reboot
```

Verify setup 
```bash
aplay -l # Should show HiFiBerry DAC/ADC
```

```bash
arecord -l # Should show HiFiBerry DAC/ADC
```
# Software Architecture

## Core Components
| File                      | Description                          |
|---------------------------|--------------------------------------|
| `stream_process.py`       | Real-time processing                 |
| `stream_process_EQ_GUI.py`| Real-time processing with EQ and GUI |
| `fir_filter.py`           | Main filter interface                |
| `filter_methods.py`       | Window/Remez algorithms              |
| `window_types.py`         | Window functions                     |
| `plot_filter.py`          | Response visualization               |


### Available filter types
```python
FILTER_TYPES = ['lowpass', 'highpass', 'bandpass', 'bandstop']
```
### Supported windows
```python
WINDOWS = ['boxcar', 'triang', 'blackman', 'hamming', 'hann',
          'bartlett', 'flattop', 'parzen', 'bohman',
          'blackmanharris', 'nuttall', 'barthann'']
```
### Design methods
```python
METHODS = ['window', 'remez']
```
# Usage Examples
### Basic Filter Design
```python
from fir_filter import create_fir_filter

# Design a bandpass filter
coeffs = create_fir_filter(
    cutoff=[500, 8000],      # Band edges
    numtaps=251,             # Filter length
    window_type='kaiser',    # Window with beta=8.6
    filter_type='bandpass',
    samplerate=44100
)
```
### Real-time Processing

Configure in stream_process.py:
```python
# Audio Config
SAMPLERATE = 44100
UPSAMPLE_FACTOR = 4          # 176.4kHz processing
CHANNELS = 1                 # one output

# Lowpass or highpass Filter Config
FILTER_TYPE = 'lowpass'     
CUTOFF = 16000  # 16kHz 

# Bandpass or Bandstop Filter Config
FILTER_TYPE = 'bandpass'     
CUTOFF = [250, 10000]        # 250Hz-10kHz passband
WINDOW_TYPE = 'blackman'
NUM_TAPS = 501               # Odd number recommended
```
Run processing:
```bash
python stream_process.py
```
For more friendly interface run:
```bash
python stream_process_EQ_GUI.py
```
# Visualization

**The visualization system provides**:
- Magnitude response (dB scale)
- Phase response (radians)
- Centered impulse response
- Automatic scaling for different sample rates
When running 'stream_process.py' following plot is provided, or optionally proceed with other maore advanced streeming script 
 ![Lowpass filter](https://github.com/user-attachments/assets/1ca441d2-7fa3-43b4-8277-95397f7edeed)

User friendly interface allows, for application the changes while streaming, just run 'stream_process_EQ_GUI.py'
 ![image](https://github.com/user-attachments/assets/e1a689fe-3249-4b68-b811-4a980868f2c5)


Example plotting code:
```python
from plot_filter import plot_filter_response

plot_filter_response(
    coefficients=coeffs,
    fs=176400,               # Upsampled rate
    filter_type='bandpass'
)
```
Enjoy quality music with Digital signal processing at your finger tips
# Troubleshooting

## Common Issues

### No Audio Output
1. Verify DAC appears in aplay -l
2. Check physical connections
3. Confirm correct /boot/firmware/config.txt settings

### High CPU Usage
1. Reduce UPSAMPLE_FACTOR
2. Decrease NUM_TAPS
3. Use simpler window (e.g., Hamming)

### Plotting Errors
Linux systems may require:
```bash
sudo apt-get install python3-tk
export QT_QPA_PLATFORM=xcb
```

### Latency Issues
Adjust buffer sizes in stream_process.py:
```python
BLOCKSIZE = 1024  # Try 512 or 2048
```
If getting warnings "Input overflow, output underflow" Adjust  parameter : latency='low',   #  <- change to "high"  in stream_process.py or GUI version:
```python
with sd.Stream(
    samplerate=sr,
    blocksize=bs,
    channels=1,
    dtype='float32',
    latency='low',   #  <- change to "high"
    callback=audio_callback(),
    device=(0, 0) # device=(input_device_index, output_device_index))
):
```
It is a runtime audio streaming error indicating that the audio buffers are either not filled in time (underflow) or not consumed fast enough (overflow). 
1.Input Overflow
Data from the ADC (audio in) is coming in faster than your code is consuming it. The internal buffer overflows → you lose samples.
2. Output Underflow
The code did not provide audio samples in time for the DAC (audio out). The DAC tries to play audio but the buffer is empty → glitch or silence.
# Acknowledgments
- DSP processing by use of SciPy and NumPy libraries for: fftconvolve Function:: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html
- Real-Time Signal Processing in Python: https://bastibe.de/2012-11-02-real-time-signal-processing-in-python.html
- SoundDevice for audio I/O: https://python-sounddevice.readthedocs.io/en/0.5.1/
- HiFiBerry for DAC documentation: https://blog.himbeer.me/2018/12/27/how-to-connect-a-pcm5102-i2s-dac-to-your-raspberry-pi/
- Graphical User Interface (GUI) with Tkinter: https://github.com/spatialaudio/python-sounddevice/blob/master/examples/rec_gui.py 
# License
MIT License - See LICENSE for details.

