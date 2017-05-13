# coding=utf-8
from pylab import *
import scipy
from scipy.io import wavfile
import scipy.signal as sigs
import scipy.fftpack as fftPack
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import wave
import contextlib


# This finds the number of samples for the file
def sampler(audiofile):
    with contextlib.closing(wave.open(audiofile, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        length = frames / float(rate)
        print(length, " Seconds")
    print(f.getparams())
    sampleRate, noiseWav = wavfile.read(audiofile)
    print(sampleRate, " Sample Rate")
    numSamples = round(sampleRate * length, 1)
    print(numSamples, " # samples")
    numChan = f.getnchannels()
    print(numChan, " # channels")
    print(noiseWav.dtype, "type")
    print("wav shape", noiseWav.shape)
    snd = (noiseWav / 2. ** 15) # normalized Audio

    return snd, numSamples, sampleRate, numChan, f.getnframes()

#if __name__ == '__main__':
# assign file to work on
# audiofile = 'Project_2017/Noisy_file_1.wav'
audiofile = 'Project_2017/spectogram_test.wav'
# audiofile = 'Project_2017/project.wav'
snd, numSamp, sampRate, numChan, nframe = sampler(audiofile)
s1 = None
s2 = None
if numChan == 2:
    s1 = snd[:, 0]
    s2 = snd[:, 1]
    print(s1, numSamp, sampRate, numChan, N, " number of positive samples")

# create time array for time plot
timearray = arange(0, numSamp, 1)
timearray = timearray / sampRate
timearray = timearray * 1000  # scale to milliseconds

figure(1)
# plot time vs amplitude(normalized to -1 to 1
title("channel 1")
plot(timearray, s1, color='k')

ylabel('Amplitude')
xlabel('Time (ms)')

# figure(2)
# title("channel 2")
# plot(timearray, s2, color='k')
# ylabel('Amplitude')
# xlabel('Time (ms)')

fourier = np.fft.fft(s1)
print(shape(fourier), "fourier shape")
uPts = int(ceil((numSamp + 1) / 2.0))
p = fourier[0:uPts]
p = abs(p)

p = p / float(numSamp)  # scale by the number of points so that
# the magnitude does not depend on the length
# of the signal or on its sampling frequency
p = p ** 2  # square it to get the power

# multiply by two (see technical document for details)
# odd nfft excludes Nyquist point
print(shape(p), "p shape")
plen = shape(p)
plen = plen[0]
print(plen)
if numSamp % 2 > 0:  # we've got odd number of points fft
    p[1:plen] = p[1:plen] * 2
else:
    p[1:plen - 1] = p[1:plen - 1] * 2  # we've got even number of points fft
print(shape(p), "p shape")

figure(2)
freqArray = arange(0, uPts, 1.0) * (sampRate / numSamp);
plot(freqArray / 1000, 10 * log10(p), color='g')
xlabel('Frequency (kHz)')
ylabel('Power (dB)')

import lowpass_filt

sigs.


show()

# logic for dual channel graphing
# chan1 = plt.subplot(timearray, s1, color='k')
# if s2 != None:
#     chan2 = subplot(timearray, s2, color='k')
# ylabel('Amplitude')

# # wav_norm =
# wavfft = fftPack.fft(wavObj)
# #wav_fft_norm = np.abs(wav_fft)
# #print(len(wav_fft))
# #print(wav_fft)
# #print(len(wav_fft_norm))
# #print(wav_fft_norm)
#
# figure(1)
# rcParams['agg.path.chunksize'] = 10000000
# f, t, spx = sigs.spectrogram(x=wavfft, fs=sampRate, mode='psd',return_onesided=True)
# pcolormesh(t, f, spx)
# ylabel("Freq (Hz)")
# xlabel("Time (sec)")
#
# # pxx, freqs, bins, im = plt.specgram(wavObj,nframe,sampRate,vmax=20000)
# #
# # ylabel("Freq (Hz)")
# # xlabel("Time (sec)")
#
# show()
