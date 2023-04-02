import copy
import os
import numpy as np
import scipy.fftpack
import sounddevice as sd
import time

# Setari generale ale tunerului
SAMPLE_FREQ = 48000 # frecventa proba in Hz
WINDOW_SIZE = 48000 # marime fereastra DFT in esantioane
WINDOW_STEP = 12000 # marime step 
NUM_HPS = 5         # numar maxim de HPS (Harmonic Product Spectrum)
POWER_THRESH = 1e-6 # tunerul e activat daca puterea semnalului este mai mare decat acest prag
CONCERT_PITCH = 440 # defining a1
WHITE_NOISE_THRESH = 0.2 # orice sub WHITE_NOISE_THRESH*avg_energy_per_freq e scos 

WINDOW_T_LEN = WINDOW_SIZE / SAMPLE_FREQ # lungime fereastra in secunde
SAMPLE_T_LENGTH = 1 / SAMPLE_FREQ        # durata unui esantion in secunde
DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE   # rezolutia spectrului in Hz
OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

ALL_NOTES = ["A","A#","B","C","C#","D","D#","E","F","F#","G","G#"]
HANN_WINDOW = np.hanning(WINDOW_SIZE)


def find_closest_note(pitch):
  i = int(np.round(np.log2(pitch/CONCERT_PITCH)*12))

  closest_note = ALL_NOTES[i%12] + str(4 + (i + 9) // 12)
  closest_pitch = CONCERT_PITCH*2**(i/12)

  return closest_note, closest_pitch


def callback(indata, frames, time, status):
  if not hasattr(callback, "window_samples"):
    callback.window_samples = [0 for _ in range(WINDOW_SIZE)]
  if not hasattr(callback, "noteBuffer"):
    callback.noteBuffer = ["1","2"]

  if status:
    print(status)
    return

  if any(indata):
    callback.window_samples = np.concatenate((callback.window_samples, indata[:, 0])) # adauga esantioane noi
    callback.window_samples = callback.window_samples[len(indata[:, 0]):] # sterge esantioanele vechi

    
    signal_power = (np.linalg.norm(callback.window_samples, ord=2, axis=0)**2) / len(callback.window_samples)
    if signal_power < POWER_THRESH:
      os.system('cls' if os.name=='nt' else 'clear')
      print("Closest note: ...")
      return

    # evita spectral leakage prin inmultirea semnalului cu o fereastra hann
    hann_samples = callback.window_samples * HANN_WINDOW
    magnitude_spec = abs(scipy.fftpack.fft(hann_samples)[:len(hann_samples)//2])

    # sterge orice sub 62 Hz
    for i in range(int(62/DELTA_FREQ)):
      magnitude_spec[i] = 0

    # calculeaza energia medie pe fiecare banda de octave
    # si sterge orice sub WHITE_NOISE_THRESH*avg_energy_per_freq
    for j in range(len(OCTAVE_BANDS)-1):
      ind_start = int(OCTAVE_BANDS[j]/DELTA_FREQ)
      ind_end = int(OCTAVE_BANDS[j+1]/DELTA_FREQ)
      ind_end = ind_end if len(magnitude_spec) > ind_end else len(magnitude_spec)
      avg_energy_per_freq = (np.linalg.norm(magnitude_spec[ind_start:ind_end], ord=2, axis=0)**2) / (ind_end-ind_start)
      avg_energy_per_freq = avg_energy_per_freq**0.5
      for i in range(ind_start, ind_end):
        magnitude_spec[i] = magnitude_spec[i] if magnitude_spec[i] > WHITE_NOISE_THRESH*avg_energy_per_freq else 0

    # interpoleaza spectrul pentru a avea un numar de esantioane fix
    mag_spec_ipol = np.interp(np.arange(0, len(magnitude_spec), 1/NUM_HPS), np.arange(0, len(magnitude_spec)),
                              magnitude_spec)
    mag_spec_ipol = mag_spec_ipol / np.linalg.norm(mag_spec_ipol, ord=2, axis=0) # normalizare

    hps_spec = copy.deepcopy(mag_spec_ipol)

    # calculeaza HPS (Harmonic Product Spectrum)
    for i in range(NUM_HPS):
      tmp_hps_spec = np.multiply(hps_spec[:int(np.ceil(len(mag_spec_ipol)/(i+1)))], mag_spec_ipol[::(i+1)])
      if not any(tmp_hps_spec):
        break
      hps_spec = tmp_hps_spec

    max_ind = np.argmax(hps_spec)
    max_freq = max_ind * (SAMPLE_FREQ/WINDOW_SIZE) / NUM_HPS

    closest_note, closest_pitch = find_closest_note(max_freq)
    max_freq = round(max_freq, 1)
    closest_pitch = round(closest_pitch, 1)

    callback.noteBuffer.insert(0, closest_note) # note that this is a ringbuffer
    callback.noteBuffer.pop()

    os.system('cls' if os.name=='nt' else 'clear')
    if callback.noteBuffer.count(callback.noteBuffer[0]) == len(callback.noteBuffer):
      print(f"Closest note: {closest_note} {max_freq}/{closest_pitch}")
    else:
      print(f"Closest note: ...")

  else:
    print('no input detected')

try:
  print("Starting HPS guitar tuner...")
  print(sd.query_devices())
  with sd.InputStream(channels=1, callback=callback, blocksize=WINDOW_STEP, samplerate=SAMPLE_FREQ, device=4):
    while True:
      time.sleep(0.5)
except Exception as exc:
  print(str(exc))