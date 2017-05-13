import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram


def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 4096  # Length of the windowing segments
    fs = 44100  # Sampling frequency
    plt.figure(1)
    pxx, freqs, bins, im = plt.specgram(data, nfft, rate)
    plt.ylabel('Freq')
    plt.xlabel('Time')
    plt.savefig('spec.png',
                dpi=600,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)  # Spectrogram saved as a .png

    plt.figure(2)
    f,t,sxx = spectrogram(data,rate)
    plt.pcolormesh(t,f,sxx)
    plt.ylabel('Freq')
    plt.xlabel('Time')
    plt.savefig('pmesh.png',
                dpi=600,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)  # Spectrogram saved as a .png


def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data


if __name__ == '__main__':  # Main function
    wav_file = 'Project_2017/log-sine-sweep.wav'  # Filename of the wav file
    graph_spectrogram(wav_file)
    plt.show()
