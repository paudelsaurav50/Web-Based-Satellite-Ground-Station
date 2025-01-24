#This code is just for the testing purposes. It is based on the script written by: https://iangoegebuer.com/2022/03/decoding-fsk-data-with-an-rtl-sdr-and-python-cheap-30-sdr/

import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import math
import sys

plt.rcParams['figure.figsize'] = [10, 5]
#
# DATA LOAD
#

# read the wave file
# Data6 shows a need for bit stuffing
# fs, rf = wf.read('data6-aq.wav') 
# Data7 makes sure to flip more often 
# fs, rf = wf.read('data7-aq.wav')
# Data8 has differential encoding and bit inversion
#fs, rf = wf.read('data-aq.wav')
#fs, rf = wf.read('sanosat2_1kbps_2kHz_Fs_96000Hz.wav')
fs, rf = wf.read('sanosat1.wav')
sf = {
    np.dtype('int16'): 2**15,
    np.dtype('int32'): 2**32,
    np.dtype('float32'): 1,
    np.dtype('float64'): 1,
}[rf.dtype]

print('dtype:',rf.dtype)
rf = (rf[:]) / sf
#rf = (rf[:, 0] + rf[:, 1]) / sf

print('dtype:',rf.dtype)
print(rf)
print(len(rf))

plt.subplot(2,2, 1)
plt.title('Raw Signal')
plt.xlabel('Time [s]')
plt.ylabel('Magnitude')
plt.plot(np.arange(0, len(rf)) / fs, rf)


decimate_by = 1
# Decimate the data to make processing simpler
fs = fs // decimate_by
rf = rf[::decimate_by]

# Thanks to Tomasz for a lot of inspiration
# https://mightydevices.com/index.php/2019/08/decoding-fsk-transmission-recorded-by-rtl-sdr-dongle/
audio_mag = np.abs(rf)

# mag threshold level
data_start_threshold = 0.0

# indices with magnitude higher than threshold
audio_indices = np.nonzero(audio_mag > data_start_threshold)[0]

# limit the signal
audio = rf[np.min(audio_indices) : np.max(audio_indices)]
print('audio len:',len(audio))
'''if len(audio[0]) != 1:
    audio = audio[:,0]
    '''
#print(audio)

audio_mag = np.abs(audio)
plt.subplot(2,2, 2)
plt.title('Raw Signal')
plt.xlabel('Time [s]')
plt.ylabel('Magnitude')
plt.plot(np.arange(0, len(audio_mag)) / fs, audio_mag)
#plt.show()
# Again, use the max of the first quarter of data to find the end (noise)
data_end_threshold = np.max(audio_mag[:round(len(audio_mag)/4)])*1.05
#data_end_threshold = 0
print('data end',data_end_threshold)
audio_indices = np.nonzero(audio_mag > data_end_threshold)[0]
# Use the data not found above and normalize to the max found
if len(audio_indices) > 0:
   audio = audio[ : np.min(audio_indices)] / (data_end_threshold/1.05)

plt.subplot(2,2, 3)
plt.title('Isolated signal')
plt.xlabel('Time [s]')
plt.ylabel('Magnitude')
plt.plot(np.arange(0, len(audio)) / fs, audio)


'''
This next chunk finds all the zero crossings, bit rate and bits
'''
zero_cross = [] 
for i in range(len(audio)):
#     print(audio[i-1])
    if audio[i -1] < 0 and audio[i] > 0:
        zero_cross += [(i, (audio[i -1]+audio[i])/2)]
    if audio[i -1] > 0 and audio[i] < 0:
        zero_cross += [(i, (audio[i -1]+audio[i])/2)]

# Get the first 10 zero crossings, ignoring the first as it may be offset
#mid=math.floor(len(zero_cross)/2)
first_ten = zero_cross[1:11]

samples_per_bit = first_ten[len(first_ten)-1][0] - first_ten[0][0]
samples_per_bit = samples_per_bit/(len(first_ten)-1)
samples_per_bit_raw = samples_per_bit


# We will be using this to index arrays, so lets floor to the nearest integer
samples_per_bit = math.floor(samples_per_bit)
#Or using 48000/500 (Filesamplerate/bps)=96
samples_per_bit = 96
print('samples per bit:',samples_per_bit)
sampled_bits = []
bits = []
# Let's iterate over the chunks of data between zero crossings
for i in range(len(zero_cross))[:-1]:
    # Now let's iterate over the bits within the zero crossings
    # Note, let's add an extra 1/8th of a sample just in case
    for j in range(math.floor((zero_cross[i+1][0]-zero_cross[i][0] + samples_per_bit/8 )/samples_per_bit)):
        # Let's offset by 1 sample in case we catch the rising and falling edge
        start = zero_cross[i][0]+j*samples_per_bit+1
        end =   zero_cross[i][0]+j*samples_per_bit+samples_per_bit-1
        sampled_bits += [(zero_cross[i][0]+j*samples_per_bit+samples_per_bit/2, np.average(audio[start:end]))]
        bits += [np.average(audio[start:end]) >= 0.1 *1]

# Let's convert the true/false data into uint8s for later use
bits = (np.array(bits)).astype(np.uint8)

print('bits:',bits)

Bits = bits

plt.subplot(2, 2, 4)
plt.title('Sampled bits')
plt.xlabel('Time [s]')
plt.ylabel('Value')
plt.plot(np.arange(0, len(audio)) / fs, audio)
plt.plot([(x[0]-.5) / fs for x in zero_cross], [0 for x in zero_cross], ".r")
plt.plot([(x[0]) / fs for x in sampled_bits], [x[1]*.8 for x in sampled_bits], ".k")
plt.show()


current_data = 0
start_data_offset = 0
data = []
found_start = False
for b in range(len(bits)):
    bit = bits[b]
    # Each byte is sent in order but the bits are sent reverse order
    current_data = current_data >> 1
    current_data += (bit*0x80)
    current_data &= 0xff

    # We've already found the start flag, time to store each byte
    if found_start:
        if ((b - start_data_offset) % 8) == 0:  
            data.append(current_data)
            current_data = 0
        continue

    # Have we found the flag? 0x2D
    if(current_data == 0b00101101) and b > 4 and not found_start:
        found_start = True
        start_data_offset = b
        data.append(0x2D)
        current_data = 0
    if(current_data == 0b10110100) and b > 4 and not found_start:
        found_start = True
        start_data_offset = b
        data.append(0xB4)
        # Invert the bit value since we found an inverted flag
        bits = (np.array([x for x in bits]) < 1 ).astype(np.uint8)
        current_data = 0

plt.title('Data,')
plt.xlabel('Bit Number')
plt.ylabel('Value')
plt.plot(bits)

plt.show()
print("Raw data %s" % data)
hex_string = " ".join("%02x" % b for b in data)
print("As hex %s: " % hex_string)