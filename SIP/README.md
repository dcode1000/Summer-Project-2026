# SIP Analysis Pipeline

## Overview 

This is a simple project to provide some tools for running inference on a checkpoint produced using the ATLAS salt-ml machine learning package. This pipeline can:

- Take in test files in the hdf5 format produced by umami-preprocessing (or any other similarly structured hdf5 files)
- Load torch lightning .ckpt files
- Prepare data for running inference on the models from the test file
- Run inference on the loaded model
- Produce a CSV file containing the output probabilities and other comparison parameters (pt, eta etc)
- Create a variety of graphs comparing different loaded models and break down their output by pt, eta and look at the actual jet flavour the model predicts each jet should be

## Project Structure

- `SIP_pipeline.py` - Main pipeline script, runs the pipeline, with various options
- `SIP_inference_functions.py` - Inference functions, ie those used to extract the model and prep the test dataset and then run inference using the loaded model
- `SIP_analysis_functions.py` - Analysis functions, used to plot graphs and performance analysis and comparison on inference produced predictions
- `SIP_config.yaml` - configures the pipeline 

## Modes

The Salt Inference Pipeline (SIP) has the following modes:

- save_mode: saves the model predicted probabilities to a csv file along with some additional variables such as the truth flavour of the jets, and their pt and eta also saves a .txt file containing information on the configuration of the pipeline. Default: True
- analysis_mode: produces a series of graphs for determining the performance of the models under analysis. Default: True

## Configuring the Pipeline

Configuration of the SIP pipeline is done via a .yaml file. All required formats are given in `SIP_config.yaml`. The pipeline can analyse any number of models, provided that all training variables are included in the test dataset. For comparison between models requiring different training variables, all possible variables can be added during umami preprocessing, and then the training variables for each model can be selected in their respective salt configs. 

### Adding Models

Models can be added to the pipeline by using the structure found in the SIP_config.yaml file. Simply indent from the models section of the config and add the name of the model you wish to analyse. YAML files are automatically read as dictionaries. To be able to perform analysis the dictionary for the model needs the following information (indented again from the model name):
- model_checkpoint: path to saved checkpoint file containing a .ckpt file containing the trained model under analysis
- inference_task: the name of the task you want to run inference on in the salt model config
- inference_output: either probs or scores, for classification models you want the softmaxed scores - ie the probabilities but for some regression models you may wish to use the scores
- inference_variables: the training variables required for the model, organised by type (ie jets, truth_hadrons etc)
- plot_colour: the colour of the output of this model during graphical analysis
To add plots you can simply add them under the inference_report section. The supertitle of the report can be added in the report_title entry. Each plot can be added separately, you only need to specify the type of plot and the x data you wish to use (if a profile histogram, otherwise do not specify).

## Running the Pipeline

To run the python from the command line simply enter:\
`python SIP_pipeline.py --config $CONFIG --ckpt $CKPT`\
Modes can be activated using `--mode`. Note that save_mode is by default turned on.

## Analysis Graphs

This pipeline currently has 3 types of graphs available for analysis:
- profile_histogram: plots a profile histogram (using matplotlib) for each jet truth flavour. The x axis can either be set to 'pt' or 'eta'. The y axis shows the model flavour tag probability.
- profile_histogram_truth: for each model produced favour probability (pb , pc, etc) produces a profile histogram of all the probabilities arranged by the truth flavour of jet. Again can specify whether to histogram the data by pt or eta
- prediction_plot: for all probabilities above a certain threshold, classifies jets based on the highest flavour tag probability of the four produced for each jet. Plots a graph for each jet flavour classified (all classified b jets), with x axis being the classification threshold and the y axis showing the fraction of classified jets. Arranges all jets in each plot by the truth flavour of the jet 

## Requirements

This software was written in python 3.11.15. The following packages are required:
- NumPy
- SciPy
- Pandas
- Matplotlib
- h5py
- hdf5plugin
- salt (salt-ml from pypi)
- yaml
- warnings

Salt is based on pytorch lightning. Ensure compatibility between the version of lightning the checkpoint was created in and the version of lightning you are running to load the checkpoint. This also goes for the version of salt being used. 

