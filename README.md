# MusicGenreClassification

<h1 align="center">
  <br>
Technion ECE 046211 - Deep Learning
  <br>
  <img src="https://raw.githubusercontent.com/taldatech/ee046211-deep-learning/main/assets/nn_gumgum.gif" height="200">
</h1>
  <p align="center">
    <a href="https://github.com/ItaiAllouche">Itai Allouche</a> •
    <a href="https://github.com/adamkatav">Adam katav</a>
  </p>

- [MusicGenreClassification](#MusicGenreClassification)
  * [Background](#background)
  * [The Model](#the-model)
  * [Dataset](#dataset)
  * [Agenda](#agenda)
  * [Results](#Results)
    + [30s model](#30s-model)
    + [15s model](#15s-model)
    + [10s model](#10s-model)
  * [Docker](#Docker)
  * [Training](#Training)
    + [Train 30s model](#train-30s-model)
  * [Run the model](#Run-the-model)
    + [Run the model - from huggingface 🤗](#run-the-model---from-huggingface-)
    + [Run the model - using python](#Run-the-model---using-python)

## Background
For our final project in the Technion DL course (046211), we chose to classify music genres over the GTZAN dataset.
<br>
The approach to the problem is using a pre-trained Wav2Vec2 transformer model.
<br>
As appose to most existing models, a transformer will use the raw time series data which is the reason we predicted an improvement over existing methods
  

## The Model
We used the model <a href="https://huggingface.co/facebook/wav2vec2-large-100k-voxpopuli">facebook/wav2vec2-large-100k-voxpopuli</a> from huggingface,
Facebooks Wav2Vec2 model pre-trained on 100k unlabeled subset of speech data.
<br>


## Dataset
We used the femiliar <a href="https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification">GTZAN</a> dataset.
<br>
The dataset consists of 1000 audio tracks each 30 seconds long.
<br>
It contains 10 genres, each represented by 100 tracks:
<br>
The genres are: `blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock`
<br>
The tracks are all 22050Hz Mono 16-bit audio files in .wav format.


## Agenda

|File       | Purpsoe |
|----------------|---------|
|`img`| Contains images for README.md file  |
|`train_30s_model.py`| train the model on 30s tracks |
|`train_15s_model.py`| train the model on 15s tracks  |
|`train_10s_model.py`| train the model on 10s tracks  |
|`eval_model.py`| evaluate the model|
|`rolling_stones.wav`| example audio file|
## Results
### 30s model
The model was trained on 30s tracks.
<br>
performance:
<br>
87% accuracy on validation set
<br>

<img src="/img/30sec_valid.jpeg">
<br>

77% accuracy on test set
<br>

<img src="/img/30sec_test.jpeg">

### 15s model
<br>
The model was trained on 15s long tracks.
Each 30s track was divided into 2 sub-tracks 15s long
<br>
performance:
<br>
78.85% accuracy on validation set
<br>

<img src="/img/15sec_valid.jpeg">
<br>

75.5% accuracy on test set
<br>

<img src="/img/15sec_test.jpeg">

### 10s model
The model was trained on 10s tracks.
Each 30s track was divided into 3 sub-tracks 10s long
<br>
performance:

<br>
78% accuracy on validation set
<br>

<img src="/img/10sec_valid.jpeg">
<br>

74.5% accuracy on test set
<br>
<img src="/img/10sec_test.jpeg">
<br>
## Docker
The project is intended to run in huggingface docker image
<br>
For instructions on how to install docker:
<br>
<a href="https://docs.docker.com/engine/install/">https://docs.docker.com/engine/install/</a>
## Training
### Train 30s model
Replace `train_30s_model.py` with your chosen model
```bash
docker run --name gtzan --rm -it --ipc=host --gpus=all -v $PWD:/home huggingface/transformers-pytorch-gpu python3 /home/train_30s_model.py
```
This command spins up a docker container from the official huggingface image, mounts the repo directory and run the training script
## Running
### Run the model - from huggingface 🤗
Open the <a href="https://huggingface.co/adamkatav/wav2vec2_100k_gtzan_30s_model">Model</a> in hugging face.
<br>
<img src="/img/run_in_hugging_face.jpeg">
<br>
*Note that hugging face server supports tracks up to 2-3 minutes*
### Run the model - using python
#### On GPU:
```bash
docker run --name gtzan --rm -it --ipc=host --gpus=all -v $PWD:/home huggingface/transformers-pytorch-gpu
```
#### On CPU:
```bash
docker run --name gtzan --rm -it -v $PWD:/home huggingface/transformers-pytorch-gpu
```
In the container either use a python script file or via the interactive interpreter:
```python
from transformers import pipeline
import torchaudio
import sys
MODEL_NAME = 'adamkatav/wav2vec2_100k_gtzan_30s_model'
SONG_IN_REPO_DIR_PATH = '/home/rolling_stones.wav'

pipe = pipeline(model=MODEL_NAME)
audio_array,sample_freq = torchaudio.load(SONG_IN_REPO_DIR_PATH)
resample = torchaudio.transforms.Resample(orig_freq=sample_freq)
audio_array = audio_array.mean(axis=0).squeeze().numpy()
output = pipe(audio_array)
print(output)
```
