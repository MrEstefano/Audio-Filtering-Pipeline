# Audio-Filtering-Pipeline
This project involves DSP using FIR filter, windowing. Hardware is RPi Zero, PCM5102 I2S DAC, USB Soundcard as MIC input

1. Create a Project Folder
bash : mkdir ~/fir-audio-pipeline
________________________________________
2. Create a Virtual Environment
bash : python3 -m venv venv
Activate it:
bash : source venv/bin/activate
You’ll now see (venv) in your prompt — this means you're inside the environment.
________________________________________
3. Install Required Packages
For real-time audio streaming with FIR filtering, let’s start with:
bash: pip install numpy sounddevice
•	numpy — for your FIR math
•	sounddevice — lightweight audio input/output in real-time (better than pyaudio for ARM boards)

________________________________________
Adio FIR pipeline:

venv/                  virtual environment

fir_filter.py          Main FIR filter interface

filter_methods.py      Different FIR design algorithms

stream_process.py      Real-time audio capture + filtering

window_types.py        mmodular choiceof windows, incl. keiser which requires beta factor
________________________________________
For connections :

PCM5102 DAC Module	with Raspberry Pi Zero
VIN	Pin 2 (5V)
GND	Pin 6 (GND)
LCK	Pin 35/ GPIO 19
DIN	Pin 40/ GPIO 21
BCK	Pin 12/ GPIO 18
SCK	GND
Note: The PCM5102 will generate SCK by itself, but it needs to know that it should do that, this is done by connecting SCK to GND. 

Software setup
This guide explains it quite well, but I will summarise it here, in case something ever happens to that link.

Editing boot.txt
Run this command to open the file in a text editor:

sudo nano /boot/config.txt
You will need to change the following things:
Uncomment (remove the # before the line):

dtparam=i2s=on
Comment (add a # before the line):

#dtparam=audio=on
Append this to the end of the file:

dtoverlay=hifiberry-dac
Creating asound.conf
Run this command to open the file in a text editor:

sudo nano /etc/asound.conf
And paste the following:

pcm.!default  {
 type hw card 0
}
ctl.!default {
 type hw card 0
}
Now reboot your Raspberry Pi
sudo reboot
Testing our changes
Use the command aplay -l to list your audio devices, if your changes were successful, the output should look like this:

pi@raspberrypi:~ $ aplay -l
 **** List of PLAYBACK Hardware Devices ****
 card 0: sndrpihifiberry [snd_rpi_hifiberry_dac], device 0: HifiBerry DAC HiFi pcm5102a-hifi-0 []
   Subdevices: 1/1
   Subdevice #0: subdevice #0
You can try playing a wav file using aplay filename.wav or install mplayer to play other file types.
________________________________________
referances/credits: https://blog.himbeer.me/2018/12/27/how-to-connect-a-pcm5102-i2s-dac-to-your-raspberry-pi/
________________________________________
Commands:
cd ~/fir-audio-pipeline
source venv/bin/activate
python3 stream_process.py
![Capture1](https://github.com/user-attachments/assets/8d929871-fcaf-411b-b259-1539dec50f53)



