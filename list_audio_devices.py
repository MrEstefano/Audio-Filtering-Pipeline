# list_audio_devices.py
import sounddevice as sd

devices = sd.query_devices()
for idx, dev in enumerate(devices):
    print(f"{idx}: {dev['name']} ({dev['max_input_channels']} in, {dev['max_output_channels']} out)")


