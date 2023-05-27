import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np

from scipy.fftpack import fft

# showing the amplitude spectrum for a given signal
def dft_plot_signal(file: str) -> None:
    sample_frequency, recording = scipy.io.wavfile.read('resources/' + file)
    sample_duration= len(recording)/sample_frequency
    time_axis = np.arange(0, sample_frequency/2, sample_frequency / len(recording))
    absolute_frequency_spectrum = abs(fft(recording))
    print("Sample duration: " + str(sample_duration))

    plt.plot(time_axis[:len(time_axis)-1], list(zip(*absolute_frequency_spectrum[:len(recording)//2]))[0]) 
    plt.ylabel('Amplitude |X(n)|')
    plt.xlabel('Frequency [Hz]')
    plt.title('The signal spectrum')
    plt.show()

def plot_signal(file: str) -> None:
    sample_frequency, recording = scipy.io.wavfile.read('resources/' + file)
    sample_duration = len(recording) / sample_frequency
    time_axis = np.arange(0, sample_duration, 1 / sample_frequency)

    plt.plot(time_axis, recording)
    plt.ylabel('Amplitude - x(k)')
    plt.xlabel('Time [s]')
    plt.title(file)
    plt.show()


if __name__ == "__main__":
    # plot_signal("example1.wav")
    dft_plot_signal("example1.wav")