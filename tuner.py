import os
import time
import copy
import numpy as np
import scipy.fftpack
import sounddevice as sd
import customtkinter as tk
from PIL import ImageTk, Image  
from typing import Tuple
import threading

# Tuner settings
SAMPLE_FREQ = 48000  # sample frequency in Hz
WINDOW_SIZE = 48000  # DFT window size in samples
WINDOW_STEP = 12000  # window step size
NUM_HPS = 5          # maximum number of Harmonic Product Spectrum (HPS)
POWER_THRESH = 1e-6  # tuner is activated if the signal power is greater than this threshold
CONCERT_PITCH = 440  # defining A4
# remove anything below WHITE_NOISE_THRESH * avg_energy_per_freq
WHITE_NOISE_THRESH = 0.2

OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
ALL_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
HANN_WINDOW = np.hanning(WINDOW_SIZE)
DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE


def find_closest_note(pitch: float) -> Tuple[str, float]:
    # Find the closest note and its pitch to the given pitch
    i = int(np.round(np.log2(pitch / CONCERT_PITCH) * 12))
    closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)
    closest_pitch = CONCERT_PITCH * 2**(i / 12)
    return closest_note, closest_pitch


def callback(indata, frames, time, status, update_label):
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
        callback.window_samples = np.concatenate(
            (callback.window_samples, indata[:, 0]))
        callback.window_samples = callback.window_samples[len(indata[:, 0]):]

        # Calculate signal power and check if it's above the threshold
        signal_power = (np.linalg.norm(callback.window_samples,
                        ord=2, axis=0)**2) / len(callback.window_samples)
        if signal_power < POWER_THRESH:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("Closest note: ...")
            return

        # Apply Hanning window to avoid spectral leakage
        hann_samples = callback.window_samples * HANN_WINDOW
        magnitude_spec = abs(scipy.fftpack.fft(
            hann_samples)[:len(hann_samples) // 2])

        # Remove anything below 62 Hz
        magnitude_spec[:int(62 / DELTA_FREQ)] = 0

        # Calculate average energy for each octave band and remove noise below threshold
        for j in range(len(OCTAVE_BANDS) - 1):
            ind_start = int(OCTAVE_BANDS[j] / DELTA_FREQ)
            ind_end = int(OCTAVE_BANDS[j + 1] / DELTA_FREQ)
            ind_end = ind_end if len(
                magnitude_spec) > ind_end else len(magnitude_spec)

            # compute medium energy on each octave band (average neergy per frequency)
            # and remove noise below threshold WHITE_NOISE_THRESH*avg_energy_per_freq
            avg_energy_per_freq = (np.linalg.norm(
                magnitude_spec[ind_start:ind_end], ord=2, axis=0)**2) / (ind_end - ind_start)
            avg_energy_per_freq = avg_energy_per_freq**0.5  # hann window
            for i in range(ind_start, ind_end):
                magnitude_spec[i] = magnitude_spec[i] if magnitude_spec[i] > WHITE_NOISE_THRESH * \
                    avg_energy_per_freq else 0

        # Interpolate spectrum to have a fixed number of samples
        mag_spec_ipol = np.interp(np.arange(0, len(magnitude_spec), 1 / NUM_HPS), np.arange(0, len(magnitude_spec)),
                                  magnitude_spec)
        mag_spec_ipol = mag_spec_ipol / \
            np.linalg.norm(mag_spec_ipol, ord=2, axis=0)  # normalize

        hps_spec = copy.deepcopy(mag_spec_ipol)

        # Calculate HPS (Harmonic Product Spectrum)
        for i in range(NUM_HPS):
            tmp_hps_spec = np.multiply(
                hps_spec[:int(np.ceil(len(mag_spec_ipol) / (i + 1)))], mag_spec_ipol[::(i + 1)])
            if not any(tmp_hps_spec):
                break
            hps_spec = tmp_hps_spec

        max_ind = np.argmax(hps_spec)
        max_freq = max_ind * (SAMPLE_FREQ / WINDOW_SIZE) / NUM_HPS

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
    # Create a basic Tkinter window for the tuner
    def update_label(note: str, max_freq: float, closest_pitch: float):
        label_text.set(f"Closest note: {note} {max_freq} / {closest_pitch}")
        root.update()

    def start_tuner():
        try:
            print("Starting HPS guitar tuner...")
            print(sd.query_devices())
            with sd.InputStream(channels=2, callback=lambda *args, **kwargs: callback(*args, **kwargs, update_label=update_label), 
                                blocksize=WINDOW_STEP, samplerate=SAMPLE_FREQ):
                while True:
                    time.sleep(0.5)
        except Exception as exc:
            print(str(exc))

    def start_tuner_thread():
        tuner_thread = threading.Thread(target=start_tuner)
        tuner_thread.daemon = True
        tuner_thread.start()

    # Initialize the Tkinter window
    root = tk.CTk()
    root.title("Guitar Tuner")
    root.geometry("500x700")
    tk.set_appearance_mode("dark")
    tk.set_default_color_theme("green")
    # frame = tk.CTkFrame(master=root)
    # frame.pack()

    # Add a label to display the closest note
    label_text = tk.StringVar()
    label_text.set("Closest note: ...")
    label = tk.CTkLabel(root, textvariable=label_text, font=("Roboto", 24))

    # label.pack(pady=10)
    label.grid(row=0, padx=50, pady=0)

    # Create an object of tkinter ImageTk
    img = ImageTk.PhotoImage(Image.open("sing.png"), height=100)

    # Create a Label Widget to display the text or Image
    label1 = tk.CTkLabel(root, image=img, text="")
    label1.grid(row=1,padx=20, pady=0, columnspan=1, rowspan=1)
    
    # Add a button to start the tuner
    start_button = tk.CTkButton(root, text="Start Tuner",
                              command=start_tuner_thread, width=300, height=50)
    # start_button.pack(pady=10)
    start_button.grid(row=2, rowspan=5, columnspan=5, padx=0, pady=10)

   # Configure grid layout weights
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)
    root.grid_rowconfigure(2, weight=1)
    
    root.mainloop()


if __name__ == "__main__":
    main()
