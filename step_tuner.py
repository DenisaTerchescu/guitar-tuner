import os
import time as tm
import copy
import numpy as np
import scipy.fftpack
import sounddevice as sd
import customtkinter as tk
from PIL import ImageTk, Image
from typing import Tuple
import threading
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Tuner settings
SAMPLE_FREQ = 48000  # Sample frequency in Hz
WINDOW_SIZE = 48000  # DFT window size in samples
WINDOW_STEP = 12000  # Window step size
NUM_HPS = 5          # Maximum number of Harmonic Product Spectrum (HPS)
POWER_THRESH = 1e-6  # Tuner is activated if the signal power is greater than this threshold
CONCERT_PITCH = 440  # Defining A4
WHITE_NOISE_THRESH = 0.2

OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
ALL_NOTES = ["LA", "LA#", "SI", "DO", "DO#", "RE", "RE#", "MI", "FA", "FA#", "SOL", "SOL#"]
HANN_WINDOW = np.hanning(WINDOW_SIZE)
DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE

# Global variable to control tuner state
tuner_running = False

def find_closest_note(pitch: float) -> Tuple[str, float]:
    """
    Find the closest note for a given pitch.

    :param pitch: The pitch to find the closest note for
    :return: Tuple with the closest note and its frequency
    """
    # Calculate the pitch difference in semitones (half steps) between the input pitch
    # and the reference concert pitch (A4 = 440Hz). The formula used is:
    # semitones = 12 * log2(pitch / CONCERT_PITCH)
    i = int(np.round(np.log2(pitch / CONCERT_PITCH) * 12))

    # The '+ 9' term is used to correctly adjust the octave when dealing with negative indices.
    closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)

    # To find the frequency of the closest note, we use the formula:
    # closest_pitch = CONCERT_PITCH * 2^(i / 12)
    # This formula calculates the frequency of the note that is 'i' semitones away from the reference
    # concert pitch (A4 = 440Hz) by multiplying the concert pitch by the 2^(i / 12) factor.
    closest_pitch = CONCERT_PITCH * 2**(i / 12)
    
    return closest_note, closest_pitch

def callback(indata, frames, time, status, update_label):
    global tuner_running
    global canvas, axs

    if not tuner_running:
        return

    os.system('cls' if os.name == 'nt' else 'clear')

    # Initialize window_samples and noteBuffer attributes if not already present
    if not hasattr(callback, "window_samples"):
        callback.window_samples = np.zeros(WINDOW_SIZE)
    if not hasattr(callback, "noteBuffer"):
        callback.noteBuffer = ["1", "2"]

    # If there's a status message, print it and return
    if status:
        print(status)
        return

    # Check if there's input data
    if indata.any():

        # STEP 1: Update the buffer
        print("\nStep 1 - Updating the buffer with input data and removing old data")

        # Add new samples and remove old samples
        callback.window_samples = np.concatenate((callback.window_samples, indata[:, 0]))
        callback.window_samples = callback.window_samples[len(indata[:, 0]):]
        
        axs[0].cla()  # Clear the graph
        axs[0].plot(callback.window_samples)  # Plot the new data
        axs[0].set_title("Raw Audio Signal")

        # Calculate signal power and check if it's above the threshold
        # STEP 2: Calculate power of the signal
        print("\nStep 2 - Calculating the power of the signal")
        signal_power = (np.linalg.norm(callback.window_samples, ord=2, axis=0)**2) / len(callback.window_samples)
        print("The signal power is: ", signal_power)
        if signal_power < POWER_THRESH:
            print("\nThe signal power is too low, skipping this frame")
            # os.system('cls' if os.name == 'nt' else 'clear')
            print("The closest note: ...")
            return

        # STEP 3: Apply the Hanning window function to the samples
        print("\nStep 3 - Applying the Hanning window function to the samples")
        hann_samples = callback.window_samples * HANN_WINDOW
        print("The samples after applying the Hanning window function are: ", hann_samples)


        # STEP 4: Compute the magnitude spectrum of the samples
        print("\nStep 4 - Computing the magnitude spectrum of the samples")
        magnitude_spec = abs(scipy.fftpack.fft(hann_samples)[:len(hann_samples) // 2])
        print("The magnitude spectrum is: ", magnitude_spec)
        
        # Update the matplotlib graph
        axs[1].cla()  # Clear the graph
        axs[1].plot(magnitude_spec)  # Plot the new data
        axs[1].set_title("Magnitude Spectrum (After FFT and Hanning Window)")


        # STEP 5: Remove low frequencies
        print("\nStep 5 - Removing low frequencies from the spectrum")
        magnitude_spec[:int(62 / DELTA_FREQ)] = 0 
        print("The magnitude spectrum after removing low frequencies is: ", magnitude_spec)


        # STEP 6: Remove noise in octave bands
        print("\nStep 6 - Removing noise in octave bands")

        # Calculate average energy for each octave band and remove noise below threshold
        for j in range(len(OCTAVE_BANDS) - 1):
            # Calculate the start and end indices in the magnitude spectrum for the current octave band.
            ind_start = int(OCTAVE_BANDS[j] / DELTA_FREQ)
            ind_end = int(OCTAVE_BANDS[j + 1] / DELTA_FREQ)
            # Ensure the end index doesn't exceed the length of the magnitude spectrum.
            ind_end = ind_end if len(magnitude_spec) > ind_end else len(magnitude_spec)

            # Compute medium energy on each octave band (average energy per frequency)
            # and remove noise below threshold WHITE_NOISE_THRESH*avg_energy_per_freq
            avg_energy_per_freq = (np.linalg.norm(magnitude_spec[ind_start:ind_end], ord=2, axis=0)**2) / (ind_end - ind_start)
            avg_energy_per_freq = avg_energy_per_freq**0.5  # hann window
            for i in range(ind_start, ind_end):
                magnitude_spec[i] = magnitude_spec[i] if magnitude_spec[i] > WHITE_NOISE_THRESH * avg_energy_per_freq else 0

        print("The magnitude spectrum after removing noise is: ", magnitude_spec)

        # Update the matplotlib graph
        axs[2].cla()  # Clear the graph
        axs[2].plot(magnitude_spec)  # Plot the new data
        axs[2].set_title("Octave Bands Energy (After applying noise threshold)")

        canvas.draw()  # Redraw the graph


        # STEP 7: Interpolate the spectrum and calculate the HPS
        print("\nStep 7 - Interpolating the spectrum and calculating the Harmonic Product Spectrum (HPS)")
        # Interpolate spectrum to have a fixed number of samples
        mag_spec_ipol = np.interp(np.arange(0, len(magnitude_spec), 1 / NUM_HPS), np.arange(0, len(magnitude_spec)), magnitude_spec)
        mag_spec_ipol = mag_spec_ipol / np.linalg.norm(mag_spec_ipol, ord=2, axis=0)  # normalize
        hps_spec = copy.deepcopy(mag_spec_ipol)

        # Calculate HPS (Harmonic Product Spectrum)
        for i in range(NUM_HPS):
            tmp_hps_spec = np.multiply(hps_spec[:int(np.ceil(len(mag_spec_ipol) / (i + 1)))], mag_spec_ipol[::(i + 1)])
            if not any(tmp_hps_spec):
                break
            hps_spec = tmp_hps_spec


        print("The Harmonic Product Spectrum is: ", hps_spec)

        # Update the HPS plot
        axs[3].cla()  # Clear the graph
        axs[3].plot(hps_spec)  # Plot the new data
        axs[3].set_title("Harmonic Product Spectrum (HPS)")

        canvas.draw()  # Redraw the graph

        print("\nStep 8 - Finding the dominant frequency")
        max_ind = np.argmax(hps_spec)
        max_freq = max_ind * (SAMPLE_FREQ / WINDOW_SIZE) / NUM_HPS
        print("The dominant frequency is: ", max_freq)


        # STEP 9: Find the closest note
        print("\nStep 9 - Finding the closest note to the dominant frequency")
        closest_note, closest_pitch = find_closest_note(max_freq)
        max_freq = round(max_freq, 1)
        closest_pitch = round(closest_pitch, 1)
        

        callback.noteBuffer.insert(0, closest_note)
        callback.noteBuffer.pop()

        tm.sleep(0.5)
        if callback.noteBuffer.count(callback.noteBuffer[0]) == len(callback.noteBuffer):
            print(f"Closest note: {closest_note} {max_freq} / {closest_pitch}")
            update_label(closest_note, max_freq, closest_pitch)

        else:
            print(f"Closest note: ...")
            update_label("..", "..", "..")

    else:
        print('no input detected')

def main():
    global tuner_running
    global canvas, axs

    def update_label(note: str, max_freq: float, closest_pitch: float):
        if tuner_running:
            label_text.set(f"Closest note: {note} {max_freq} / {closest_pitch}")
        else:
            label_text.set("Guitar Tuner!")
        root.update()

    def start_tuner():
        global tuner_running
        try:
            print("Starting HPS guitar tuner...")
            print(sd.query_devices())
            with sd.InputStream(channels=2, callback=lambda *args, **kwargs: callback(*args, **kwargs, update_label=update_label),
                                blocksize=WINDOW_STEP, samplerate=SAMPLE_FREQ):
                while tuner_running:
                    tm.sleep(0.5)
        except Exception as exc:
            print(str(exc))

    def toggle_tuner():
        global tuner_running
        tuner_running = not tuner_running
        if tuner_running:
            button_text.set("Stop Tuner")
            start_thread = threading.Thread(target=start_tuner)
            start_thread.start()
        else:
            button_text.set("Start Tuner")

    # Create the main window
    root = tk.CTk()
    root.geometry("500x600")
    root.title("Guitar Tuner")

    # Create label for displaying the closest note
    label_text = tk.StringVar()
    label_text.set("Guitar Tuner!")
    label = tk.CTkLabel(root, textvariable=label_text, font=("Helvetica", 16))
    
    label.pack(pady=10)

    # Create a matplotlib figure with 5 subplots
    fig, axs = plt.subplots(4, figsize=(10,8))
    plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between subplots

    # Create a matplotlib canvas and add it as a tkinter widget
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Create start/stop button
    button_text = tk.StringVar()
    button_text.set("Start Tuner")
    button = tk.CTkButton(root, textvariable=button_text, command=toggle_tuner)
    button.pack(pady=10)

    # Run the main loop
    root.mainloop()

if __name__ == "__main__":
    main()
