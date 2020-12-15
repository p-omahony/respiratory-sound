import numpy as np
import librosa
import soundfile as sf
import os

def time_stretch(audiofile, rate) :
    y, sr = librosa.load(audiofile)
    return librosa.effects.time_stretch(y, rate), sr

if __name__ == '__main__' :
    data_path = '../data/audio_and_txt_files/'
    filenames = [f for f in os.listdir(data_path) if (os.path.isfile(os.path.join(data_path, f)) and f.endswith('.wav'))]
    filepaths = [os.path.join(data_path, f) for f in filenames]

    p_id_in_file = [int(name[:3]) for name in filenames]

    features, c, max_pad_len, n_files = [], 1, 862, len(filepaths)
    rates = [0.8, 0.9, 1.1, 1.2]
    for file_name in filepaths:
        print('Stretching : ' + file_name + ' (' + str(c) + '/' + str(n_files) + ')')
        for rate in rates :
            y, sr = time_stretch(file_name, rate)
            sf.write(data_path+'stretched/'+file_name.split('/')[3].replace('.wav','')+'_stretched_'+str(rate)+'.wav', y, sr)
        c+=1
