
import soundfile as sf
import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt

def background_noise(audiofile):
    """
    Add a noise.
    The noise is proportional to the amplitude of the sound signal

    :param audiofile: path of the audio file
    :return: y,sr the noise with noise (format wav)
    """
    y, sr = librosa.load(audiofile, res_type='kaiser_fast',  duration=20)


    # noise coefficient
    max = y.mean()
    min = y.min()
    coef_noise = ((max-min) / 2) * 1/20

    # gaussian random noise
    noise = np.random.normal(0, coef_noise, len(y))

    return y + noise, sr

if __name__ == '__main__' :
    data_path = '../data/audio_and_txt_files/'
    filenames = [f for f in os.listdir(data_path) if
                 (os.path.isfile(os.path.join(data_path, f)) and f.endswith('.wav'))]
    filepaths = [os.path.join(data_path, f) for f in filenames]

    for filepath in filepaths:

        print("Add noise to the file : " + filepath)

        # charging the file and add the noise
        y, sr = background_noise(filepath)
        # construct the new path
        new_path = filepath.replace(".wav", "") + "With_Noise.wav"
        # writte the new soundfile
        sf.write(new_path, y, sr)

