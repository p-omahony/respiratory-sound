# respiratory-sound

## Installation

Firstly clone the project : ```git clone https://github.com/p-omahony/respiratory-sound.git``` \

Go on the root of the project : ```cd respiratory-sound``` \

Install the required python libraries : ```pip install -r requirements.txt``` \

Download the data : ```wget "http://164.68.116.174:5000/download/files/isep/a3/deep-learning/respiratory-sound/data.zip"``` and then unzip it \

Dowload the pre-processed data : \
```mkdir preprocessed_data && cd preprocessed_data``` \
```wget "http://164.68.116.174:5000/download/files/isep/a3/deep-learning/respiratory-sound/extracted_features.npy" && wget "http://164.68.116.174:5000/download/files/isep/a3/deep-learning/respiratory-sound/labels.npy"```
