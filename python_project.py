import contextlib
import wave
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, lfilter, firwin, kaiserord, freqz
import scipy
import numpy as np

np.set_printoptions(threshold=np.inf)

# SIGNAL GLOBALS
SND = None
NUMSAMP = None
SAMPRATE = None
NUMCHAN = None
NFRAME = None

# FILTER GLOBALS (HZ)
FILTER_RATE = 20000
NYQUISTRATE = 10000
WIDTH = 250
PASSBAND = 1000
CENTER = 5000
ATTENUATION = 60  # (dB)


def sampler(audiofile):
    with contextlib.closing(wave.open(audiofile, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        length = frames / float(rate)
        print(length, " Seconds")
    print(f.getparams())
    sampleRate, noiseWav = wavfile.read(audiofile)  # read in the audio file into Numpy array
    print(sampleRate, " Sample Rate")
    numSamples = round(sampleRate * length, 1)
    print(numSamples, " # samples")
    numChan = f.getnchannels()  # find the number of channels
    print(numChan, " # channels")
    print(noiseWav.dtype, "type")  # display data type for this system
    print("wav shape", noiseWav.shape)
    snd = (noiseWav / 2. ** 15)  # normalized Audio
    return snd, numSamples, sampleRate, numChan, f.getnframes()


def graph_spectrogram(data, rate, title):
    # rate, data = get_wav_info(wav_file)
    nfft = 4096  # Length of the windowing segments
    fs = 44100  # Sampling frequency
    plt.figure()
    pxx, freqs, bins, im = plt.specgram(data, nfft, rate)
    plt.ylabel('Freq')
    plt.xlabel('Time')
    plt.savefig(title + '_sp.png',
                dpi=600,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)  # Spectrogram saved as a .png

    plt.figure()
    f, t, sxx = spectrogram(data, rate)
    plt.pcolormesh(t, f, sxx)
    plt.ylabel('Freq')
    plt.xlabel('Time')
    plt.savefig(title + '_pmesh.png',
                dpi=600,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)  # Spectrogram saved as a .png
    return


def channel_analyzer(snd):
    s1 = None  # channel 1
    s2 = None  # channel 2

    # create time array for time plot
    timearray = scipy.arange(0, numSamp, 1)
    timearray = timearray / sampRate
    timearray = timearray * 1000  # scale to milliseconds

    if NUMCHAN == 1:
        s1 = snd

        # plot time vs amplitude(normalized to -1 to 1)
        plt.figure()
        plt.title("channel 1")
        plt.plot(timearray, s1, color='k')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (ms)')
        plt.savefig('chan1.png',
                    dpi=600,  # Dots per inch
                    frameon='false',
                    aspect='normal',
                    bbox_inches='tight',
                    pad_inches=0)  # Spectrogram saved as a .png

    if NUMCHAN == 2:  # repeat if there are two channels
        s1 = snd[:, 0]

        # plot time vs amplitude(normalized to -1 to 1)
        plt.figure()
        plt.title("channel 1")
        plt.plot(timearray, s1, color='k')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (ms)')
        plt.savefig('chan1.png',
                    dpi=600,  # Dots per inch
                    frameon='false',
                    aspect='normal',
                    bbox_inches='tight',
                    pad_inches=0)  # Spectrogram saved as a .png

        s2 = snd[:, 1]  # set the right channel equal to s2

        # plot right channel
        plt.figure()
        plt.title("channel 2")
        plt.plot(timearray, s2, color='k')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (ms)')
        plt.savefig('chan2.png',
                    dpi=600,  # Dots per inch
                    frameon='false',
                    aspect='normal',
                    bbox_inches='tight',
                    pad_inches=0)  # Spectrogram saved as a .png

    # spectrogram the left and right channel
    graph_spectrogram(s1, SAMPRATE, 'chan1')
    graph_spectrogram(s2, SAMPRATE, 'chan2')
    return s1, s2


def LOWPASS(x):
    lpf_data = open("lowpf.txt", "w")
    # The Nyquist rate of the signal.
    sample_rate = FILTER_RATE
    nyq_rate = sample_rate / 2.0

    width = 250.0 / nyq_rate

    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)

    # The cutoff frequency of the filter.
    cutoff_hz = 1125.0

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, cutoff_hz / nyq_rate, width=width, window=('kaiser', beta))

    # ------------------------------------------------
    # Apply filter to signal
    # ------------------------------------------------
    filtered_x = lfilter(taps, 1.0, x)

    # ------------------------------------------------
    # Plot the FIR filter coefficients.
    # ------------------------------------------------

    plt.figure()
    plt.plot(taps, 'bo-', linewidth=2)
    plt.title('Filter Coefficients (%d taps)' % N)
    plt.grid(True)

    plt.savefig('lpf_coeff.png',
                dpi=600,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)
    lpf_data.write("Coeffs: \n")
    lpf_data.write(np.array2string(taps, 5, 5, True, ','))
    lpf_data.write("\n\n")

    # ------------------------------------------------
    # Plot the magnitude response of the filter.
    # ------------------------------------------------

    plt.figure()
    plt.clf()
    w, h = freqz(taps, worN=8000)
    plt.plot((w / scipy.pi) * nyq_rate, scipy.absolute(h), linewidth=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('Frequency Response')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)

    # Upper inset plot.
    plt.ax1 = plt.axes([0.42, 0.6, .45, .22])
    plt.title("Passband Ripple")
    plt.plot((w / scipy.pi) * nyq_rate, scipy.absolute(h), linewidth=2)
    plt.xlim(700, 1000)
    plt.ylim(0.9985, 1.0025)
    plt.grid(True)

    # Lower inset plot
    plt.ax2 = plt.axes([0.42, 0.25, .45, .25])
    plt.title("Stopband Ripple")
    plt.plot((w / scipy.pi) * nyq_rate, scipy.absolute(h), linewidth=2)
    plt.xlim(1200.0, 1600.0)
    plt.ylim(0.0, 0.0011)
    plt.grid(True)

    plt.savefig('lpf_Fresponse.png',
                dpi=600,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)
    lpf_data.write("normalized freq: \n")
    lpf_data.write(np.array2string(w, precision=5, suppress_small=True, separator=','))
    lpf_data.write("\n\n")
    lpf_data.write("complex freq: \n")
    lpf_data.write(np.array2string(h, precision=5, suppress_small=True, separator=','))
    lpf_data.write("\n\n")
    # ------------------------------------------------
    # Plot the original and filtered signals.
    # ------------------------------------------------

    # create time array for time plot
    timearray = scipy.arange(0, numSamp, 1)
    timearray = timearray / sampRate
    t = timearray * 1000  # scale to milliseconds

    # The phase delay of the filtered signal.
    delay = 0.5 * (N - 1) / sample_rate

    plt.figure()
    # Plot the original signal.
    plt.plot(t, x)
    # Plot the filtered signal, shifted to compensate for the phase delay.
    plt.plot(t - delay, filtered_x, 'r-')
    # Plot just the "good" part of the filtered signal.  The first N-1
    # samples are "corrupted" by the initial conditions.
    plt.plot(t[N - 1:] - delay, filtered_x[N - 1:], 'g', linewidth=4)

    plt.xlabel('t')
    plt.grid(True)
    plt.savefig('lpf_filtSig.png',
                dpi=600,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)

    graph_spectrogram(filtered_x, sampRate, 'lpf')
    lpf_data.close()
    return


def BANDPASS(x):
    bpf_data = open("bpf.txt", "w")
    # The Nyquist rate of the signal.
    sample_rate = FILTER_RATE
    nyq_rate = sample_rate / 2.0

    width = 250.0 / nyq_rate

    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)

    # The cutoff frequency of the filter.
    bpCut1 = 3875 / nyq_rate
    bpCut2 = 6125 / nyq_rate

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, [bpCut1, bpCut2], width=width, window=('kaiser', beta), pass_zero=False)

    # ------------------------------------------------
    # Apply filter to signal
    # ------------------------------------------------
    filtered_x = lfilter(taps, 1.0, x)

    # ------------------------------------------------
    # Plot the FIR filter coefficients.
    # ------------------------------------------------

    plt.figure()
    plt.plot(taps, 'bo-', linewidth=2)
    plt.title('Filter Coefficients (%d taps)' % N)
    plt.grid(True)

    plt.savefig('bpf_coeff.png',
                dpi=600,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)
    bpf_data.write("Coeffs: \n")
    bpf_data.write(np.array2string(taps, 5, 5, True, ','))
    bpf_data.write("\n\n")

    # ------------------------------------------------
    # Plot the magnitude response of the filter.
    # ------------------------------------------------

    plt.figure()
    plt.clf()
    w, h = freqz(taps, worN=8000)
    plt.plot((w / scipy.pi) * nyq_rate, scipy.absolute(h), linewidth=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('Frequency Response')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)

    # # Upper inset plot.
    # plt.ax1 = plt.axes([0.42, 0.6, .45, .22])
    # plt.title("Passband Ripple")
    # plt.plot((w / scipy.pi) * nyq_rate, scipy.absolute(h), linewidth=2)
    # plt.xlim(700, 1000)
    # plt.ylim(0.9985, 1.0025)
    # plt.grid(True)
    #
    # # Lower inset plot
    # plt.ax2 = plt.axes([0.42, 0.25, .45, .25])
    # plt.title("Stopband Ripple")
    # plt.plot((w / scipy.pi) * nyq_rate, scipy.absolute(h), linewidth=2)
    # plt.xlim(1200.0, 1600.0)
    # plt.ylim(0.0, 0.0011)
    # plt.grid(True)

    plt.savefig('bpf_Fresponse.png',
                dpi=600,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)
    bpf_data.write("normalized freq: \n")
    bpf_data.write(np.array2string(w, precision=5, suppress_small=True, separator=','))
    bpf_data.write("\n\n")
    bpf_data.write("complex freq: \n")
    bpf_data.write(np.array2string(h, precision=5, suppress_small=True, separator=','))
    bpf_data.write("\n\n")

    # ------------------------------------------------
    # Plot the original and filtered signals.
    # ------------------------------------------------

    # create time array for time plot
    timearray = scipy.arange(0, numSamp, 1)
    timearray = timearray / sampRate
    t = timearray * 1000  # scale to milliseconds

    # The phase delay of the filtered signal.
    delay = 0.5 * (N - 1) / sample_rate

    plt.figure()
    # Plot the original signal.
    plt.plot(t, x)
    # Plot the filtered signal, shifted to compensate for the phase delay.
    plt.plot(t - delay, filtered_x, 'r-')
    # Plot just the "good" part of the filtered signal.  The first N-1
    # samples are "corrupted" by the initial conditions.
    plt.plot(t[N - 1:] - delay, filtered_x[N - 1:], 'g', linewidth=4)

    plt.xlabel('t')
    plt.grid(True)
    plt.savefig('bpf_filtSig.png',
                dpi=600,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)

    graph_spectrogram(filtered_x, sampRate, 'bpf')
    bpf_data.close()
    return


def HIGHPASS(x):
    hpf_data = open("hpf.txt", "w")
    # The Nyquist rate of the signal.
    sample_rate = FILTER_RATE
    nyq_rate = sample_rate / 2.0

    width = 250.0 / nyq_rate

    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)

    # The cutoff frequency of the filter.
    bpCut1 = 3875 / nyq_rate
    bpCut2 = 6125 / nyq_rate

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, [bpCut1, bpCut2], width=width, window=('kaiser', beta), pass_zero=False)

    # ------------------------------------------------
    # Apply filter to signal
    # ------------------------------------------------
    filtered_x = lfilter(taps, 1.0, x)

    # ------------------------------------------------
    # Plot the FIR filter coefficients.
    # ------------------------------------------------

    plt.figure()
    plt.plot(taps, 'bo-', linewidth=2)
    plt.title('Filter Coefficients (%d taps)' % N)
    plt.grid(True)

    plt.savefig('hpf_coeff.png',
                dpi=600,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)
    hpf_data.write("Coeffs: \n")
    hpf_data.write(np.array2string(taps, 5, 5, True, ','))
    hpf_data.write("\n\n")
    # ------------------------------------------------
    # Plot the magnitude response of the filter.
    # ------------------------------------------------

    plt.figure()
    plt.clf()
    w, h = freqz(taps, worN=8000)
    plt.plot((w / scipy.pi) * nyq_rate, scipy.absolute(h), linewidth=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('Frequency Response')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)

    # Upper inset plot.
    plt.ax1 = plt.axes([0.22, 0.55, .35, .25])
    plt.title("Passband Ripple")
    plt.plot((w / scipy.pi) * nyq_rate, scipy.absolute(h), linewidth=2)
    plt.xlim(8950, 9200)
    plt.ylim(0.995, 1.0025)
    plt.grid(True)

    # Lower inset plot
    plt.ax2 = plt.axes([0.22, 0.2, .35, .25])
    plt.title("Stopband Ripple")
    plt.plot((w / scipy.pi) * nyq_rate, scipy.absolute(h), linewidth=2)
    plt.xlim(8600.0, 8900.0)
    plt.ylim(0.0, 0.011)
    plt.grid(True)

    plt.savefig('hpf_Fresponse.png',
                dpi=600,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)
    hpf_data.write("normalized freq: \n")
    hpf_data.write(np.array2string(w, precision=5, suppress_small=True, separator=','))
    hpf_data.write("\n\n")
    hpf_data.write("complex freq: \n")
    hpf_data.write(np.array2string(h, precision=5, suppress_small=True, separator=','))
    hpf_data.write("\n\n")

    # ------------------------------------------------
    # Plot the original and filtered signals.
    # ------------------------------------------------

    # create time array for time plot
    timearray = scipy.arange(0, numSamp, 1)
    timearray = timearray / sampRate
    t = timearray * 1000  # scale to milliseconds

    # The phase delay of the filtered signal.
    delay = 0.5 * (N - 1) / sample_rate

    plt.figure()
    # Plot the original signal.
    plt.plot(t, x)
    # Plot the filtered signal, shifted to compensate for the phase delay.
    plt.plot(t - delay, filtered_x, 'r-')
    # Plot just the "good" part of the filtered signal.  The first N-1
    # samples are "corrupted" by the initial conditions.
    plt.plot(t[N - 1:] - delay, filtered_x[N - 1:], 'g', linewidth=4)

    plt.xlabel('t')
    plt.grid(True)
    plt.savefig('hpf_filtSig.png',
                dpi=600,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)

    graph_spectrogram(filtered_x, sampRate, 'hpf')
    hpf_data.close()
    return


if __name__ == '__main__':  # Main function
    audiofile = 'Project_2017/project.wav'  # Filename of the wav file
    snd, numSamp, sampRate, numChan, nframe = sampler(audiofile)  # retrieve audio data

    # set global Audio data
    SND = snd
    NUMSAMP = numSamp
    SAMPRATE = sampRate
    NUMCHAN = numChan
    NFRAME = nframe
    s1, s2 = channel_analyzer(snd)  # display the left and right channel

    # apply filters to audio data
    LOWPASS(s1)
    BANDPASS(s1)
    HIGHPASS(s1)

    # uncomment the next line to see the graphs
    # plt.show()
