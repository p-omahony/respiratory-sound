import os
import pandas as pd
import librosa
import numpy as np

def get_data(data_path):
    filenames = [f for f in os.listdir(data_path) if (os.path.isfile(os.path.join(data_path, f)) and f.endswith('.wav'))]
    filepaths = [os.path.join(data_path, f) for f in filenames]

    p_id_in_file = [int(name[:3]) for name in filenames]

    p_diag = pd.read_csv('../data/patient_diagnosis.csv',header=None)
    labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])

    return filepaths, labels

def extract_mfcc(file_name):
    max_pad_len = 862
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=16)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs

def extract_features(filepaths):
    features, c, max_pad_len, n_files = [], 1, 862, len(filepaths)
    for file_name in filepaths:
        print('Extracting features from : ' + file_name + ' (' + str(c) + '/' + str(n_files) + ')')
        data = extract_mfcc(file_name)
        features.append(data)
        c+=1
    return features

def main(data_path, dest_filename_labels, dest_filename_features):
    filepaths, labels = get_data(data_path)
    np.save('../preprocessed_data/' + dest_filename_labels, labels)
    features = extract_features(filepaths)
    print('Finished feature extraction from ', len(features), ' files')
    features = np.array(features)
    print('Saving features in ../data/extracted_features.npy ...')
    np.save('../preprocessed_data/'+dest_filename_features, features)
    print('Features saved ! \n')

if __name__ == '__main__' :

    data_path = '../data/audio_and_txt_files/'
    data_stretched = '../data/audio_and_txt_files/stretched/'

    main(data_path, 'labels.npy', 'features.npy')
    main(data_stretched, 'labels_stretched.npy', 'features_stretched.npy')
