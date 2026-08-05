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
