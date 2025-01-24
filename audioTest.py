from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import math

'''
samplerate, data = wavfile.read('sanosat1.wav')
mean=np.mean(data)
print('samplerate: ',samplerate, ',Mean:',mean)
rx_data = []
for bit in data:
    if bit > mean:
        rx_data.append(int(1))
    else:
        rx_data.append(int(0))
plt.figure(1)
plt.plot(rx_data)
plt.show()
rx_data1 = ''.join(map(str, (rx_data)))
outfile1 = open('FSK_wav6_output_binary.txt', 'w')
outfile1.write(rx_data1)
outfile1.close()
'''

samplerate, data = wavfile.read('sanosat1.wav')

audio_mag = np.abs(data)

data_start_threshold = 0.25
 
# indices with magnitude higher than threshold
audio_indices = np.nonzero(audio_mag > data_start_threshold)[0]
# limit the signal
audio = data[np.min(audio_indices) : np.max(audio_indices)]
 
audio_mag = np.abs(data)
# Again, use the max of the first quarter of data to find the end (noise)
data_end_threshold = np.max(audio_mag[:round(len(audio_mag)/4)])*1.05
 
audio_indices = np.nonzero(audio_mag > data_end_threshold)[0]
# Use the data not found above and normalize to the max found
audio = audio[ : np.min(audio_indices)] / (data_end_threshold/1.05)

#print('samplerate: ',samplerate, ',Mean:',mean)
plt.figure(1)
plt.subplot(1,2,1)
plt.plot(audio)

zero_cross = [] 
for i in range(len(audio)):
    if audio[i -1] < 0 and audio[i] > 0:
        zero_cross += [(i, (audio[i -1]+audio[i])/2)]
    if audio[i -1] > 0 and audio[i] < 0:
        zero_cross += [(i, (audio[i -1]+audio[i])/2)]
 
# Get the first 10 zero crossings, ignoring the first as it may be offset
first_ten = zero_cross[1:11]
 
samples_per_bit = first_ten[-1][0] - first_ten[0][0]
samples_per_bit = samples_per_bit/(len(first_ten)-1)
samples_per_bit_raw = samples_per_bit
 
# We will be using this to index arrays, so lets floor to the nearest integer
samples_per_bit = math.floor(samples_per_bit)

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
        bits += [np.average(audio[start:end]) >= 0]
 
# Let's convert the true/false data into uint8s for later use
bits = (np.array(bits)).astype(np.uint8)

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
        data.append(0x2D)
        # Invert the bit value since we found an inverted flag
        bits = (np.array([x for x in bits]) < 1 ).astype(np.uint8)
        current_data = 0

plt.subplot(1,2,2)
plt.plot(bits)
plt.show()