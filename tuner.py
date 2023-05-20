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
    tm.sleep(1)
    """
    Find the closest note for a given pitch.

    :param pitch: The pitch to find the closest note for
    :return: Tuple with the closest note and its frequency
    """
    # Calculate the pitch difference in semitones (half steps) between the input pitch
    # and the reference concert pitch (A4 = 440Hz). The formula used is:
    # semitones = 12 * log2(pitch / CONCERT_PITCH)
    i = int(np.round(np.log2(pitch / CONCERT_PITCH) * 12))
    print("Step 8: Calculating the semitone = 12 * log2(pitch / CONCERT_PITCH): " + str(i))
    tm.sleep(1)

    # The '+ 9' term is used to correctly adjust the octave when dealing with negative indices.
    closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)

    # To find the frequency of the closest note, we use the formula:
    # closest_pitch = CONCERT_PITCH * 2^(i / 12)
    # This formula calculates the frequency of the note that is 'i' semitones away from the reference
    # concert pitch (A4 = 440Hz) by multiplying the concert pitch by the 2^(i / 12) factor.
    closest_pitch = CONCERT_PITCH * 2**(i / 12)
    print("Step 9: Calculating the frequency of the closest pitch with 2^(semitone / 12):  " + str(closest_pitch))
    tm.sleep(3)

    return closest_note, closest_pitch

def callback(indata, frames, time, status, update_label):
    global tuner_running

    if not tuner_running:
        return

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
        # Add new samples and remove old samples
        callback.window_samples = np.concatenate((callback.window_samples, indata[:, 0]))
        callback.window_samples = callback.window_samples[len(indata[:, 0]):]

        # Calculate signal power and check if it's above the threshold
        signal_power = (np.linalg.norm(callback.window_samples, ord=2, axis=0)**2) / len(callback.window_samples)
        if signal_power < POWER_THRESH:
            os.system('cls' if os.name == 'nt' else 'clear')

            print("The closest note: ...")
            return

        # Apply Hanning window to avoid spectral leakage
        hann_samples = callback.window_samples * HANN_WINDOW
        print("Step 1: Applying Hanning window in order to prevent spectral leakage")
        tm.sleep(1)
        magnitude_spec = abs(scipy.fftpack.fft(hann_samples)[:len(hann_samples) // 2])
        print("Step 2: Calculating the magnitude spectrum: " + str(magnitude_spec))
        tm.sleep(1)
        # Remove anything below 62 Hz
        magnitude_spec[:int(62 / DELTA_FREQ)] = 0
        print("Step 3: Removing anything below the frequency of 62Hz")
        tm.sleep(1)       

        # Calculate average energy for each octave band and remove noise below threshold
        print("Step 4: Calculate average energy for each octave band and remove noise below threshold")
        tm.sleep(1)
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

        # Interpolate spectrum to have a fixed number of samples
        mag_spec_ipol = np.interp(np.arange(0, len(magnitude_spec), 1 / NUM_HPS), np.arange(0, len(magnitude_spec)), magnitude_spec)
        mag_spec_ipol = mag_spec_ipol / np.linalg.norm(mag_spec_ipol, ord=2, axis=0)  # normalize
        print("Step 5: Interpolate spectrum to have a fixed number of samples")
        tm.sleep(1)
        hps_spec = copy.deepcopy(mag_spec_ipol)

        # Calculate HPS (Harmonic Product Spectrum)
        print("Step 6: Calculate HPS (Harmonic Product Spectrum)")
        tm.sleep(1)
        for i in range(NUM_HPS):
            tmp_hps_spec = np.multiply(hps_spec[:int(np.ceil(len(mag_spec_ipol) / (i + 1)))], mag_spec_ipol[::(i + 1)])
            if not any(tmp_hps_spec):
                break
            hps_spec = tmp_hps_spec

        max_ind = np.argmax(hps_spec)
        max_freq = max_ind * (SAMPLE_FREQ / WINDOW_SIZE) / NUM_HPS
        print("Step 7: Getting the maximum frequency: " + str(max_freq))
        tm.sleep(1)
        closest_note, closest_pitch = find_closest_note(max_freq)
        max_freq = round(max_freq, 1)
        closest_pitch = round(closest_pitch, 1)
        

        callback.noteBuffer.insert(0, closest_note)
        callback.noteBuffer.pop()

        os.system('cls' if os.name == 'nt' else 'clear')
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
    root.geometry("500x400")
    root.title("Guitar Tuner")

    # Create label for displaying the closest note
    label_text = tk.StringVar()
    label_text.set("Guitar Tuner!")
    label = tk.CTkLabel(root, textvariable=label_text, font=("Helvetica", 16))
    
    label.pack(pady=10)

    # Add the sing.png image to the window
    sing_img = Image.open("sing.png")
    sing_img.thumbnail((300, 300), Image.ANTIALIAS)
    sing_photo = ImageTk.PhotoImage(sing_img)
    sing_label = tk.CTkLabel(root, image=sing_photo)
    sing_label.pack()

    # Create start/stop button
    button_text = tk.StringVar()
    button_text.set("Start Tuner")
    button = tk.CTkButton(root, textvariable=button_text, command=toggle_tuner)
    button.pack(pady=10)

    # Run the main loop
    root.mainloop()

if __name__ == "__main__":
    main()


