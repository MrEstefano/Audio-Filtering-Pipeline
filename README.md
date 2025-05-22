# Audio Filtering Pipeline with FIR Filters

![DSP Pipeline](https://img.shields.io/badge/Realtime-DSP_Processing-blue) 
![Raspberry Pi](https://img.shields.io/badge/Hardware-RPi_Zero_PCM5102-green)
![Python](https://img.shields.io/badge/Python-3.7%2B-yellow)

A real-time audio processing system implementing FIR filters on Raspberry Pi 5 with USB soundcard input, I2S DAC output, featuring customizable filter design and visualization.

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
- Multiple windowing methods (Hamming, Blackman, Kaiser, Nuttall)
- 1x-4x oversampling capabilities
- Interactive filter response visualization
- Optimized for Raspberry Pi Zero performance

**Key Features**:
- Sample rates from 44.1kHz to 176.4kHz
- Adjustable filter lengths (51-1001 taps)
- Both windowed and Remez exchange algorithms
- TPDF dithering for improved dynamic range

## Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/audio-filter-pipeline.git
cd audio-filter-pipeline
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

### PCM5102 I2S DAC Connections
![IMG_1001](https://github.com/user-attachments/assets/8f219b7e-44ed-46b0-8e0f-c663d9f67b66)

| PCM5102 Pin | RPi Zero Pin | GPIO |
|-------------|--------------|------|
| VIN (5V)    | Pin 2        | -    |
| GND         | Pin 6        | -    |
| LCK (LRCK)  | Pin 35       | 19   |
| DIN (DATA)  | Pin 40       | 21   |
| BCK (BCLK)  | Pin 12       | 18   |
| SCK         | GND          | -    |
### Soundcard connection via USB
https://www.aliexpress.com/item/1005003192869006.html

### System Configuration
Enable I2S in /boot/config.txt:

```bash
sudo nano /boot/config.txt
```
```ini
dtparam=i2s=on
#dtparam=audio=on
dtoverlay=hifiberry-dac
Configure ALSA (/etc/asound.conf):
```
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


# Software Architecture

## Core Components
| File                      | Description                          |
|---------------------------|--------------------------------------|
| `stream_process.py`       | Real-time processing                 |
| `stream_process_EQ_GUI.py`| Real-time processing                 |
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
WINDOWS = ['hamming', 'hann', 'blackman', 'kaiser', 'nuttall', 'flattop']
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
CHANNELS = 1                 # Mono

# Filter Config
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
python stream_process_EG_GUI.py
```
# Visualization

**The visualization system provides**:
- Magnitude response (dB scale)
- Phase response (radians)
- Centered impulse response
- Automatic scaling for different sample rates

  ![Lowpass filter](https://github.com/user-attachments/assets/1ca441d2-7fa3-43b4-8277-95397f7edeed)

User friendly interface allows, for application the changes while streaming, just run 'stream_process_EG_GUI.py'
  ![image](https://github.com/user-attachments/assets/9b91ad16-c6b8-49bc-8c4f-6642aa5890ea)

Example plotting code:
```python
from plot_filter import plot_filter_response

plot_filter_response(
    coefficients=coeffs,
    fs=176400,               # Upsampled rate
    filter_type='bandpass'
)
```
# Troubleshooting

## Common Issues

### No Audio Output
1. Verify DAC appears in aplay -l
2. Check physical connections
3. Confirm correct /boot/config.txt settings

### High CPU Usage
1. Reduce UPSAMPLE_FACTOR
2. Decrease NUM_TAPS
3. Use simpler window (e.g., Hamming)

### Plotting Errors
```bash
# Linux systems may require:
sudo apt-get install python3-tk
export QT_QPA_PLATFORM=xcb
```

### Latency Issues
Adjust buffer sizes in stream_process.py:
```python
BLOCKSIZE = 1024  # Try 512 or 2048
```
# Acknowledgments
- SciPy and NumPy communities
- SoundDevice for audio I/O
- HiFiBerry for DAC documentation https://blog.himbeer.me/2018/12/27/how-to-connect-a-pcm5102-i2s-dac-to-your-raspberry-pi/
  
# License
MIT License - See LICENSE for details.

