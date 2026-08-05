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

## Configuring the Pipeline

## Running the Pipeline

## Requirements

List the software and Python packages needed to run the project.

Example:

- Python 3.x
- NumPy
- SciPy
- Pandas
- Matplotlib

## Installation

Clone the repository:

```bash
git clone https://github.com/dcode1000/SaltInferencePipeline.git
