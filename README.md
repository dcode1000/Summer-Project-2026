<<<<<<< HEAD
﻿# Summer-Project-2026

This repository contains all configs and selected other files used in this project. This project had the primary goal of training a truth surrogate version of the GN2 flavour tagging model used by ATLAS. A secondary objective was to observe scaling law behaviour in trained models. All documents related to this objective are in the scaling_laws_test directory, with the rest of the repository relating to the primary objective. All information regarding the scaling laws objective are in a separate README in that folder.

## Software

In this project 2 key python packages were used:
- umami-preprocessing: Used to prepare data for input into model training. The docs for this package can be found here: https://umami-hep.github.io/umami-preprocessing/  
- salt-ml: Used to train models using the salt framework. The docs can be found at: https://ftag-salt.docs.cern.ch/

In order to have the appropriate data for training it was necessary to use the Ftag-dumpster package (docs: https://training-dataset-dumper.docs.cern.ch/), which creates hdf5 files from simulation data.

### Versions

Salt-ml versions 0.9.0 and 0.13.0 were used in this project. All finalised configs and attached documents were created using salt-ml 0.13.0. The other packages used versions:
- ftag-dumpster: uses AthAnalysis 25.2.97
- umami-preprocessing 0.3.1

Python version 3.11.15 was used throughout. 

#### Creating a Virtual Environment
To ensure proper versioning, uv was used to create virtual environments which allowed controlling which versions of each software were used. Instructions for installing uv can be found here: https://docs.astral.sh/uv/getting-started/installation/, creating a virtual environment here: https://docs.astral.sh/uv/pip/environments/ and using that environment in jupyterlab here: https://discourse.jupyter.org/t/how-to-manage-multiple-python-versions-and-environments/38072/3.

## Truth Model Training Pipeline

All 3 packages are controlled by config files, (umami, salt use .yaml files and ftag-dumpster uses .json). These config files have been saved in this repository. All configs used in the final truth model training pipeline are in the finalised_pipeline folder. 

### Stage 1: Ftag-dumpster
The first stage of the pipeline is dumping the appropriate data into the hdf5 format required for the rest of the software in the pipeline. This is necessary because the truth hadrons in the original open dataset available were not "non-Geant Truth Hadrons". This means that the truth hadrons contained within the original dataset were those which had been processed by simulation software which simulated their impact with the detector etc. The aim of this project was for a truth level trained model so truth hadrons which hadn't interacted with the detector were required.

#### Running the Dumper

Using appropriate files of the DAOD format and the config in this repository, hdf5 files were generated using the command:\
`dump-single-btag -c $CONFIG $FILENAME -o $OUTPUT_FILENAME`

### Stage 2: Umami-preprocessing
After hdf5 files containing the appropriate data have been created, it must be prepared for use in model training. This can be done using umami-preprocessing. Umami can preprocess data using the command: \
`preprocess --config $CONFIG --prep --resample --merge --norm --plot --split=all`\
This performs all steps of preprocessing and creates training, validation and testing datasets as well as norm_dict.yaml and class_dict.yaml files necessary for salt training. The norm_dict and class_dict file have been included in the UPP folder of the finalised_pipeline folder. 

### Stage 3: Salt Training
Once the data has been preprocessed, training can be performed. This is done using salt-ml. Salt training is again configured using a config, three configs for various different models have been included in the salt folder of the finalised_configs folder. In the config the training variables must be specified. To use multiple different kinds of variable in the same training (say "tracks" and "truth_hadrons") separate init_nets must be setup in the appropriate place, which can be copied from the setup in the configs in this repository. Training without using GPUs is not recommended. Various GPU settings can be configured from the config.yaml file.

#### Running Salt Training

Salt training can be performed using the command:\
`salt fit --config $CONFIG`\
To target specific GPUs can use the `--trainer.devices` flag and enclose specific device numbers in square brackets. A slurm submission script is also included in the docs, although will not work with uv virtual environments. Manual alteration of the produced sbatch script can be used to make it work with uv venvs.

## Analysis

Analysing trained models can be done using the salt test command (see docs). For more in depth analysis the 'Salt Inference Pipeline' (SIP) which is in a linked repository, can be used. (SOME FEATURES STILL WORK IN PROGRESS). This contains some python scripts for producing graphs and comparing models.
=======
# SIP Analysis Pipeline

## Overview 

This is a simple project to provide some tools for running inference on a checkpoint produced using the ATLAS salt-ml machine learning package. This pipeline can:

- Take in test files in the hdf5 format produced by umami-preprocessing (or any other similarly structured hdf5 files)
- Load a torch lightning .ckpt file
- Identify relevant variables for running inference on from the hdf5 file
- Run inference on the loaded model
- Produce a CSV file containing the output probabilities and other comparison parameters (pt, eta etc)
- Create an variety of graphs analysing the output of the model, breaking it down by pt, eta and measuring how the predictions change with classification threshold

## Project Structure

- `SIP_pipeline.py` - Main pipeline script, runs the pipeline, with various options
- `SIP_inference_functions.py` - Inference functions, ie those used to extract the model and prep the test dataset and then run inference using the loaded model
- `SIP_analysis_functions.py` - Analysis functions, used to plot graphs and performance analysis and comparison on inference produced predictions
- `SIP_config.txt` - Configuration settings for the pipeline, controlled from a .txt file

## Modes

The Salt Inference Pipeline (SIP) has the following modes:

- save_mode: saves the model predicted probabilities to a csv file along with some additional variables such as the truth flavour of the jets, and their pt and eta also saves a .txt file containing information on the configuration of the pipeline. Default: True
- analysis_mode: produces a series of graphs for determining the performance of the model under analysis. Default: False
- comp_mode: takes in a csv file containing inference data from another model and then compares this to the model output being analysed in this training mode, allowing visual comparison on the produced graphs (need analysis mode on for comp mode to be useful). Default: False

## Configuring the Pipeline

Configuration of the SIP pipeline is done via a .txt file. All required formats are given in `SIP_config.txt`. 

## Running the Pipeline

To run the python from the command line simply enter:\
`python SIP_pipeline.py --config $CONFIG --ckpt $CKPT`\
Modes can be activated using `--mode`. Note that save_mode is by default turned on.

## Requirements

This software was written in python 3.11.15. The following packages are required:
- NumPy
- SciPy
- Pandas
- Matplotlib
- h5py
- hdf5plugin
- salt (salt-ml from pypi)

Salt is based on pytorch lightning. Ensure compatibility between the version of lightning the checkpoint was created in and the version of lightning you are running to load the checkpoint. This also goes for the version of salt being used. 

## Installation

Clone the repository:

```bash
git clone https://github.com/dcode1000/SaltInferencePipeline.git
>>>>>>> SIP/main
