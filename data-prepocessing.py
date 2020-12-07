import os
import pandas as pd
import librosa
import numpy as np

def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=20)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs


if __name__ == '__main__' :

    data_path = './data/audio_and_txt_files/'
    filenames = [f for f in os.listdir(data_path) if (os.path.isfile(os.path.join(data_path, f)) and f.endswith('.wav'))]
    filepaths = [os.path.join(data_path, f) for f in filenames]

    p_id_in_file = [int(name[:3]) for name in filenames]

    p_diag = pd.read_csv('./data/patient_diagnosis.csv',header=None)
    labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])
    np.save('./preprocessed_data/labels.npy', labels)


    features, c, max_pad_len, n_files = [], 1, 862, len(filepaths)
    for file_name in filepaths:
        print('Extracting features from : ' + file_name + ' (' + str(c) + '/' + str(n_files) + ')')
        data = extract_features(file_name)
        features.append(data)
        c+=1

    print('Finished feature extraction from ', len(features), ' files')
    features = np.array(features)
    print('Saving features in ./data/extracted_features.npy ...')
    np.save('./preprocessed_data/extracted_features.npy', features)
    print('Features saved !')
