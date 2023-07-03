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
    + [Running Online](#running-online)
    + [Running Locally](#running-locally)
  * [Installation Instructions](#installation-instructions)
    + [Libraries to Install](#libraries-to-install)

## Background
As our final project in Deep learning course, we chose a problem of genre classification of a given 30-sec track.
  We chose to solve this problem using transformer architecture.

## The Model
We used the model facebook/wav2vec2-large-100k-voxpopuli, a Facebook's Wav2Vec2 large model pre-trained on the 100k unlabeled subset of VoxPopuli corpus.
      https://huggingface.co/facebook/wav2vec2-large-100k-voxpopuli

## Dataset
We used the femiliar <a href="https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification">GTZAN</a> dataset.
The dataset consists of 1000 audio tracks each 30 seconds long.
It contains 10 genres, each represented by 100 tracks:
The genrs are: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.
The tracks are all 22050Hz Mono 16-bit audio files in .wav format.


## Agenda

|File       | Purpsoe |
|----------------|---------|
|`img`| Contains images for README.md file  |
|`train_30s_model.ipynb`| train the model on 30 sec-long tracks |
|`train_15s_model.ipynb`| train the model on 15 sec-long tracks  |
|`eval_model.ipynb`| evaluate the model.|


## 30_sec model
This model was trained on 30 sec long tracks.
performance:

87% accuracy on validation set

<img src="/img/30sec_vaild.png">

77% accuracy on test set

<img src="/img/30sec_test.jpeg">

## 15_sec model
This model was trained on 15 sec long tracks.
Each 30-sec track was divided into 2 sub-track on 15 sec long
performance:

78.85% accuracy on validation set

<img src="/img/15sec_vaild.jpeg">

75.5% accuracy on test set

<img src="/img/15sec_test.jpeg">

## 10_sec model
This model was trained on 10 sec long tracks.
Each 30-sec track was divided into 3 sub-track on 10 sec long
performance:

% accuracy on validation set


% accuracy on test set





### Running Online

|Service      | Usage |
|-------------|---------|
|Jupyter Nbviewer| Render and view the notebooks (can not edit) |
|Binder| Render, view and edit the notebooks (limited time) |
|Google Colab| Render, view, edit and save the notebooks to Google Drive (limited time) |


Jupyter Nbviewer:

[![nbviewer](https://raw.githubusercontent.com/taldatech/ee046211-deep-learning/main/assets/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/taldatech/ee046202-unsupervised-learning-data-analysis/tree/master/)


Press on the "Open in Colab" button below to use Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taldatech/ee046202-unsupervised-learning-data-analysis)

Or press on the "launch binder" button below to launch in Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/taldatech/ee046202-unsupervised-learning-data-analysis/master)

Note: creating the Binder instance takes about ~5-10 minutes, so be patient

### Running Locally

Press "Download ZIP" under the green button `Clone or download` or use `git` to clone the repository using the 
following command: `git clone https://github.com/taldatech/ee046211-deep-learning.git` (in cmd/PowerShell in Windows or in the Terminal in Linux/Mac)

Open the folder in Jupyter Notebook (it is recommended to use Anaconda). Installation instructions can be found in `Setting Up The Working Environment.pdf`.


## Installation Instructions

For the complete guide, with step-by-step images, please consult `Setting Up The Working Environment.pdf`

1. Get Anaconda with Python 3, follow the instructions according to your OS (Windows/Mac/Linux) at: https://www.anaconda.com/products/individual
2. Install the basic packages using the provided `environment.yml` file by running: `conda env create -f environment.yml` which will create a new conda environment named `deep_learn`. If you did this, you will only need to install PyTorch, see the table below.
3. Alternatively, you can create a new environment for the course and install packages from scratch:
In Windows open `Anaconda Prompt` from the start menu, in Mac/Linux open the terminal and run `conda create --name deep_learn`. Full guide at https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands
4. To activate the environment, open the terminal (or `Anaconda Prompt` in Windows) and run `conda activate deep_learn`
5. Install the required libraries according to the table below (to search for a specific library and the corresponding command you can also look at https://anaconda.org/)

### Libraries to Install

|Library         | Command to Run |
|----------------|---------|
|`Jupyter Notebook`|  `conda install -c conda-forge notebook`|
|`numpy`|  `conda install -c conda-forge numpy`|
|`matplotlib`|  `conda install -c conda-forge matplotlib`|
|`pandas`|  `conda install -c conda-forge pandas`|
|`scipy`| `conda install -c anaconda scipy `|
|`scikit-learn`|  `conda install -c conda-forge scikit-learn`|
|`seaborn`|  `conda install -c conda-forge seaborn`|
|`tqdm`| `conda install -c conda-forge tqdm`|
|`opencv`| `conda install -c conda-forge opencv`|
|`optuna`| `pip install optuna`|
|`pytorch` (cpu)| `conda install pytorch torchvision torchaudio cpuonly -c pytorch` |
|`pytorch` (gpu)| `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch` |
|`torchtext`| `conda install -c pytorch torchtext`|


5. To open the notebooks, open Ananconda Navigator or run `jupyter notebook` in the terminal (or `Anaconda Prompt` in Windows) while the `deep_learn` environment is activated.











































