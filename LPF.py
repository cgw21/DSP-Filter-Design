# coding=utf-8
# Low Pass Filter
# Specifications for Low Pass Filter:
# Passband Edge 1 kHz
# Transition Width 250 Hz
# Passband Ripple < 0.1 dB
# Stopband Attenuation > 50 dB
# Sampling Rate 20 kHz
# Center Frequency 5 kHz

from pylab import *
import scipy
from scipy.io import wavfile
import scipy.signal as sigs
import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy.io import wavfile
import wave
import contextlib

# assign file to work on
audiofile = 'Project_2017/Noisy_file_1.wav'


# This finds the number of samples for the file
def samplecounter(audiofile):
    with contextlib.closing(wave.open(audiofile, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        length = frames / float(rate)
        print(length)
    sampleRate, noiseWav = wavfile.read('Project_2017/Noisy_file_1.wav')
    print(sampleRate)
    numSamples = round(sampleRate * length, 1)
    print(numSamples)
    return numSamples


def calc_dB(amplitude):
    decibel = 20 * log10(amplitude)
    return decibel


SAMPLE_SIZE = samplecounter(audiofile)

# all values are in Hertz
filterRate = 20000
nyquistRate = 10000
passBandWidth = 1000
transWidth = 250
stopBandEdge = 1250
centerFreq = 5000
lpfCut = 1125

pass_nyq = passBandWidth / nyquistRate
tranWid_nyq = transWidth / nyquistRate
stopEdge_nyq = stopBandEdge / nyquistRate
lpfCut_nyq = lpfCut / nyquistRate

ripple_dB = 55.0

numtaps, beta = sigs.kaiserord(ripple_dB, tranWid_nyq)

taps = sigs.firwin(numtaps=numtaps, cutoff=lpfCut, width=transWidth, window=('kaiser', beta), nyquistRate=nyquistRate)



# use filtfilt
def lowpass(testSignal):
    b = sigs.firwin2(numtaps=255, freq=[0, .1, .125, 1.0], gain=[1, 1, 0, 0])

    Fsig = sigs.lfilter(b, 1, testSignal)

    figure(1)
    plot(b, 'bo-', linewidth=2)
    title('coefs')
    grid(True)

    # figure(2)
    # clf()
    # w, h = sigs.freqz(b, worN=SAMPLE_SIZE)
    # plot((w / pi) * nyquistRate, absolute(h), linewidth=2)
    return
