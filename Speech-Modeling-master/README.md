Accent Screech
==============================

# Accent Preserving Voice Cloning using SV2TTS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Dataset Source
===================
GMU Speech Accent Archive [https://www.kaggle.com/rtatman/speech-accent-archive/download]
TIMIT corpus [https://www.kaggle.com/mfekadu/darpa-timit-acousticphonetic-continuous-speech/download]

Data Preparation
===================
* Run python [src/data/data_Create.py](src/data/data_create.py)
    * Downloads the data into [data/raw](data/raw)
    * Creates interim data formats [data/interim](data/interim)
    * Creates Processed data [data/processed](data/processed/)
    * Does other structural setups 

Running Models
===================
Run any of the model file directly, each has info regarding how to run it.
* Accent Encoder LSTM Model at [src/models/encoder.py](src/models/encoder.py)
* Accent Encoder 1dConv Model at [src/models/encoder_conv.py](src/models/encoder_conv.py)
* Accent Encoder LSTM Model wiht triplet loss at [src/models/encoder_trip.py](src/models/encoder_trip.py)
* Accent Encoder autoencoder 1DConv Model at [src/models/encoder_ae.py](src/models/encoder_ae.py)
* Accent Encoder autoencoder LSTM Model at [src/models/encoder_ae_rec.py](src/models/encoder_ae_rec.py)
* Accent Encoder Conv VAE Model at [src/models/encoder_vae_conv.py](src/models/encoder_vae_conv.py)

#### Training metrics are saved in [runs](runs) directory, can be visualized by pointing tensorboard at the driectory


#### Models are saved in [models_](models_) directory


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
