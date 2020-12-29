import numpy as np
import librosa
import soundfile as sf
import pandas as pd
import os

def pitch_shifting(audiofile, n_steps) :
    y, sr = librosa.load(audiofile, res_type='kaiser_fast', duration=16)
    return librosa.effects.pitch_shift(y, sr, n_steps), sr

if __name__ == '__main__' :
    data_path = '../data/audio_and_txt_files/'
    filenames = [f for f in os.listdir(data_path) if (os.path.isfile(os.path.join(data_path, f)) and f.endswith('.wav'))]
    filepaths = [os.path.join(data_path, f) for f in filenames]

    p_id_in_file = [int(name[:3]) for name in filenames]

    p_diag = pd.read_csv('../data/patient_diagnosis.csv',header=None)
    labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])

    features, c, max_pad_len, n_files = [], 1, 862, len(filepaths)
    rates = [-3.5,-2.5,-2,-1,1,2,2.5,3.5]
    for k in range(len(filepaths)):
        if labels[k] != 'COPD' :
            print('Pitch Shifting : ' + filepaths[k] + ' (' + str(c) + '/' + str(n_files) + ')')
            for rate in rates :
                y, sr = pitch_shifting(filepaths[k], rate)
                sf.write(data_path+'pitch_shifted/'+filepaths[k].split('/')[3].replace('.wav','')+'_pitchshifted1_'+str(int(rate*100))+'.wav', y, sr)
                y, sr = librosa.load(data_path+'pitch_shifted/'+filepaths[k].split('/')[3].replace('.wav','')+'_pitchshifted1_'+str(int(rate*100))+'.wav', res_type='kaiser_fast', duration=20)
                sf.write(data_path+'pitch_shifted/'+filepaths[k].split('/')[3].replace('.wav','')+'_pitchshifted_'+str(int(rate*100))+'.wav', y, sr)
                os.remove(data_path+'pitch_shifted/'+filepaths[k].split('/')[3].replace('.wav','')+'_pitchshifted1_'+str(int(rate*100))+'.wav')
            c+=1
