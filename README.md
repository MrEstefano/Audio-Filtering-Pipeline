# Audio-Filtering-Pipeline
This project involves DSP using FIR filter, windowing. Hardware is RPi Zero, DAC, USB Soundcard as MIC input
Adio FIR pipeline
├── venv/                  ← your virtual environment
├── fir_filter.py          ← Main FIR filter interface
├── filter_methods.py      ← Different FIR design algorithms
├── window_types.py        ← Different window functions
├── stream_process.py      ← Real-time audio capture + filtering
└── filters/               ← Directory to store designed filters

Commands:
cd ~/fir-audio-pipeline
source venv/bin/activate
python3 stream_process.py

