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
  * * [Dataset](#dataset)
  * [Agenda](#agenda)
  * [30_sec model](#30_sec-model)
  * [15_sec model](#15_sec-model)
  * [10_sec model](#10_sec-model)
    + [Train the 30s model](#train-the-30s-model)
    + [Run the model](#run-the-model)

## Background
As our final project in Deep learning course, we chose a problem of genre classification of a given 30-sec track.
<br>
We chose to solve this problem using Wav2Vec2 transformer architecture.
<br>
the data is time series, therefore we assume a transformer architecture will suit the task.
  

## The Model
We used the model facebook/wav2vec2-large-100k-voxpopuli,
<br>
a Facebook's Wav2Vec2 large model pre-trained on the 100k unlabeled subset of VoxPopuli corpus.
<br>
https://huggingface.co/facebook/wav2vec2-large-100k-voxpopuli

## Dataset
We used the femiliar <a href="https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification">GTZAN</a> dataset.
<br>
The dataset consists of 1000 audio tracks each 30 seconds long.
<br>
It contains 10 genres, each represented by 100 tracks:
<br>
The genrs are: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.
<br>
The tracks are all 22050Hz Mono 16-bit audio files in .wav format.


## Agenda

|File       | Purpsoe |
|----------------|---------|
|`img`| Contains images for README.md file  |
|`train_30s_model.py`| train the model on 30 sec-long tracks |
|`train_15s_model.py`| train the model on 15 sec-long tracks  |
|`train_10s_model.py`| train the model on 10 sec-long tracks  |
|`eval_model.py`| evaluate the model.|
|`rolling_stones.wav`| audio file for evaluate the model.|

## 30_sec model
This model was trained on 30 sec long tracks.
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

## 15_sec model
<br>
This model was trained on 15 sec long tracks.
Each 30-sec track was divided into 2 sub-track on 15 sec long
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

## 10_sec model
This model was trained on 10 sec long tracks.
Each 30-sec track was divided into 3 sub-track on 10 sec long
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

### Train the 30s model
```bash
docker run --name gtzan --rm -it --ipc=host --gpus=all -v $PWD:/home huggingface/transformers-pytorch-gpu python3 /home/train_30s_model.py
```
### Run the model
Open to the <a href="https://huggingface.co/adamkatav/wav2vec2_100k_gtzan_30s_model">Model</a> in hugging face.
<br>
<img src="/img/run_in_hugging_face.jpeg">
<br>
Choose your wishful song to be genre classifed.
<br>
*Note that hugging face server supports tracks up to 2-3 minutes*











































