import numpy as np
import librosa
import soundfile as sf
import pandas as pd
import os

def time_stretch(audiofile, rate) :
    y, sr = librosa.load(audiofile, res_type='kaiser_fast', duration=20)
    return librosa.effects.time_stretch(y, rate), sr

if __name__ == '__main__' :
    data_path = '../data/audio_and_txt_files/'
    filenames = [f for f in os.listdir(data_path) if (os.path.isfile(os.path.join(data_path, f)) and f.endswith('.wav'))]
    filepaths = [os.path.join(data_path, f) for f in filenames]

    p_id_in_file = [int(name[:3]) for name in filenames]

    p_diag = pd.read_csv('../data/patient_diagnosis.csv',header=None)
    labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])

    features, c, max_pad_len, n_files = [], 1, 862, len(filepaths)
    rates = [0.8, 0.9, 1.1, 1.2]
    for k in range(len(filepaths)):
        if labels[k] != 'COPD' :
            print('Stretching : ' + filepaths[k])
            for rate in rates :
                y, sr = time_stretch(filepaths[k], rate)
                sf.write(data_path+'stretched/'+filepaths[k].split('/')[3].replace('.wav','')+'_stretched1_'+str(int(rate*100))+'.wav', y, sr)
                y, sr = librosa.load(data_path+'stretched/'+filepaths[k].split('/')[3].replace('.wav','')+'_stretched1_'+str(int(rate*100))+'.wav', res_type='kaiser_fast', duration=20)
                sf.write(data_path+'stretched/'+filepaths[k].split('/')[3].replace('.wav','')+'_stretched_'+str(int(rate*100))+'.wav', y, sr)
                os.remove(data_path+'stretched/'+filepaths[k].split('/')[3].replace('.wav','')+'_stretched1_'+str(int(rate*100))+'.wav')
            c+=1
