
# Summer-Project-2026

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
