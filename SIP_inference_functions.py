#!/usr/bin/env python
# coding: utf-8


### Salt Inference Pipeline - Inference Functions
#Repository for all data processing, model loading and inference running functions used in Salt Inference Pipeline

import h5py, hdf5plugin, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from salt.modelwrapper import ModelWrapper
import scipy.stats as stats
from matplotlib.lines import Line2D

### Data Preparation
def analysis_config_parser(config_file):
    """
    reads in a config file (.txt) which tells it which datasets and the columns of those datasets to select from an h5 file for running inference/input
    into a trained model. Returns a a list for each dataset of relevant columns
    inputs:
    - config_file: contains heading "variables" under this are further headings which correspond to datasets of the h5 file. Under these are bulletpoints
    pertaining to the relevant columns of the dataset to be selected
    returns:
    - NORM: path to normalisation dictionary for the trained model
    - N: sample size for analysis
    - STORED: name of a model for comparison with the returns of the model with which inference is being run on
    - low_pt_cutoff: the minimum pt value a jet needs to have
    - high_pt_cutoff: the maximum pt value a jet can have
    - list_dict: dictionary with each key corresponding to a dataset name and the value being a list of columns to be taken from that dataset
    """
    with open(config_file, "r") as file:
        # Skip everything until "variables:"
        for line in file:
            if line.startswith("test_file: "):
                TEST_FILE = line.split("test_file: ",1)[1] # is the h5 dataset for running inference on
                TEST_FILE = TEST_FILE.strip()
            if line.startswith("comp_baseline_data: "):
                COMP_BASELINE_DATA = line.split("comp_baseline_data: ",1)[1] # csv file containing comparison jet flavour
                                                    # probabilities and corresponding jet pt, eta vals and truth flavours
                COMP_BASELINE_DATA = COMP_BASELINE_DATA.strip()
            if line.startswith("model_name: "):
                MODEL_NAME = line.split("model_name: ",1)[1]
                MODEL_NAME = MODEL_NAME.strip()
            if line.startswith("norm_dict: "):
                NORM = line.split("norm_dict: ",1)[1]
                NORM = NORM.strip()
            if line.startswith("sample_size: "):
                N = int(line.split("sample_size: ",1)[1])
            if line.startswith("stored: "):
                STORED = line.split("stored: ",1)[1]
                STORED = STORED.strip()
            if line.startswith("low_pt_cutoff: "):
                low_pt_cutoff = int(line.split("low_pt_cutoff: ",1)[1])
            if line.startswith("high_pt_cutoff: "):
                high_pt_cutoff = int(line.split("high_pt_cutoff: ",1)[1])
            if line.strip() == "variables:":
                break

        list_dict = {}
        current_key = None

        for line in file:
            line = line.strip()

            if not line:  # Skip blank lines
                continue

            if line.endswith(":"):
                current_key = line[:-1]  # Remove the colon
                list_dict[current_key] = []
            elif current_key is not None:
                list_dict[current_key].append(line.strip()[2:])
        return TEST_FILE,COMP_BASELINE_DATA,MODEL_NAME,NORM,N,STORED,low_pt_cutoff,high_pt_cutoff,list_dict

def h5_test_datafile_prepper(h5_file,list_dict,sample_size,STORED,lower_pt_cutoff,upper_pt_cutoff):
    """
    Takes in an h5 file of data to run inference on, and a dictionary of lists of variables that the machine model has been trained on. Returns a
    dictionary of datasets, keys being the keys of the input dictionary and values being datasets containing the relevant columns for input into inference
    Inputs:
    - h5_file: test dataset to run inference on
    - list_dict: dictionary containing lists of all variables used in training the model
    - sample_size: number of entries to be selected from the h5 file
    - STORED: name of a stored model in the h5 file for comparison
    - lower cutoff: lower bound of pt cutoff
    - upper cutoff: upper bound of pt cutoff
    Returns:
    - data_dict: dictionary containing tensor datasets for relevant keys and columns, filtered by pt and sample size
    - pad_dict: a dictionary which shows where a pad has been used - only applicable if using track information
    - comp_probs: a pandas dataset of flavour probabilities returned from the STORED model
    - truth_flavours: true flavours of all sampled jets
    Requirements:
    - h5py, pandas, torch, numpy
    """
    data_dict = {key: 0 for key in list_dict.keys()} # creating a dictionary with the same shape as the input list dictionary
    pad_dict = {}

    with h5py.File(h5_file, "r") as f:

        pt = f["jets"]["pt_btagJes"][:] # only sampling jets from the desired pt range

        indices = np.where((pt >= lower_pt_cutoff) & (pt <= upper_pt_cutoff))[0][:sample_size] # identifying the indices of the first sample size number of jets

        ### extracting other necessary variables from h5 dataset

        jets = f["jets"]
        comp_vals = np.stack([jets[f"{STORED}_pb"],jets[f"{STORED}_pc"],jets[f"{STORED}_pu"],jets[f"{STORED}_ptau"],], axis=1)[indices] 
                            # output from another model, for the same jets
        comp_probs = pd.DataFrame(comp_vals, columns=[f"{STORED}_pb", f"{STORED}_pc", f"{STORED}_pu", f"{STORED}_ptau"]) 
                        # converting probabilities returned from the STORED model to pd dataframe 
        truth_flavours = np.vectorize({0: "b", 1: "c", 2: "light", 3: "tau"}.get)(jets["flavour_label"][indices])
                    # truth flavours of the jets - mapped to their names

        pt_vals = jets["pt_btagJes"][indices]/1000 # extracting corresponding eta and pt values for the jets sampled (in GeV)
        eta_vals = jets["eta_btagJes"][indices]

        for key in data_dict.keys():
            data_dict[key] = f[key][indices] # putting the correctly sized datasets for jets within the desired pt range onto their corresponding key


    for key in data_dict.keys():

        if key == "tracks" or key == "truth_hadrons": # tracks requires masking due to the "valid" column

            dset = data_dict[key] # the data_dict["tracks"] currently contains all track data columns

            pad = ~dset["valid"].astype(bool)  # identifying where tracks not valid
            pad_dict[key] = torch.from_numpy(pad) # creating a mask for these

            x = np.stack([dset[v] for v in list_dict[key]],axis=-1).astype(np.float32) # stacking all required columns from the input list_dict
            x = np.nan_to_num(x,nan=0.0)
            x = torch.from_numpy(x) # creating a tensor of the same datatype as the numpy data array

            x[pad_dict[key]] = 0.0 # setting the invalid track params to 0

            data_dict[key] = x  # setting the tensor to be the value of the "tracks" key

        else:

            x = np.stack([data_dict[key][v] for v in list_dict[key]],axis=-1).astype(np.float32) # stacking the appropriate parameters
            data_dict[key] = torch.from_numpy(x) # setting the value of the key to be a tensor containing the appropriate parameters

    return data_dict,pad_dict,comp_probs,truth_flavours,pt_vals,eta_vals

### Loading Model
def model_loader(CKPT,NORM):
    """
    Instantiates a model from a lightning wrapper and a lightning checkpoint. Requires no inputs except previously defined parameters.
    Returns:
    - model: Is a loaded salt model of type ModelWrapper from salt-ml package
    Requires:
    - torch, torch lightning, salt-ml (version 0.9.0)
    """
    norm_config = dict(torch.load(CKPT, map_location="cpu", weights_only=False)["hyper_parameters"]["norm_config"]) # extracting norm_config from the checkpoint

    norm_config["norm_dict"] = NORM # attaching the norm dictionary into the norm config

    model = ModelWrapper.load_from_checkpoint(CKPT, map_location="cpu", norm_config=norm_config,weights_only=False).eval() # loading the model using the lightning method
                    # to load directly from the checkpoint, also setting the model to eval mode
    return model

### Running Inference
def inference_run(data_dict,pad_dict,model,return_scores=False):
    """
    runs inference for a trained model (called model) and returns the probabilities applied by softmax. Can also return scores if scores==True
    inputs:
    - data_dict: dictionary of tensors containing samples and processed datasets with columns corresponding to the training inputs of the model
    - pad_dict: dictionary telling model where padding was applied
    returns:
    - probs: array containing 4 probabilities for each jet with the probabilities corresponding to the probability that the jet is each of the 4 flavours
            in order (b, c , light, tau)
    - scores: original 4 scores for each jet returned by model before softmax applied
    """
    with torch.no_grad():  # running inference using the model

        scores = model(data_dict,pad_dict) # getting scores


    probs = torch.softmax(scores[0]["jets"]["jets_classification"], -1).numpy() #softmax converts to probability

    if return_scores == True:
        return probs, scores
    else:
        return probs

def save_output(probs,pt_vals,eta_vals,truth_flavours, MODEL_NAME, CKPT, FILE, CONFIG, N):
    """
    Saves a pandas dataframe of model inferenced jet flavour probabilities, organised by flavour as well as the pt, eta and truth flavours of the jets. 
    Also saves a .txt file containing info on the model name, checkpoint filename, inference dataset filename, SIP config .txt file and inference sample size.
    Inputs:
    - probs: model returned jet flavour probabilities from running inference on the input dataset
    - pt_vals: pt of classified jets (in GeV)
    - eta_vals: eta of classified jets
    - truth_flavours: truth flavours of classified jets
    - MODEL_NAME: name of model inference is being run on
    - CKPT: filename of .ckpt file from which model has been loaded
    - FILE: filename of .h5 file containing the datasets used to run inference
    - CONFIG: filename of SIP .txt config file
    - N: sample size of data used to run inference
    Returns:
    - model_name_inference_output.csv: csv file containing pandas dataframe organised data
    - model_name_inference_run_data.txt: .txt file containing information on the inference run, specifying files used to run it
    Requires:
    - pandas
    """
    ### creating datafram
    data = pd.DataFrame({f'{MODEL_NAME}_pb':probs[:,0]
                        ,f'{MODEL_NAME}_pc':probs[:,1]
                        ,f'{MODEL_NAME}_pu':probs[:,2],
                        f'{MODEL_NAME}_ptau':probs[:,3]})
    data["pt"] = pt_vals
    data["eta"] = eta_vals
    data["truth_flavour"] = truth_flavours
    data.to_csv(f'{MODEL_NAME}_inference_output.csv',index=False)

    ### creating inference_run_data
    with open(f'{MODEL_NAME}_inference_run_data.txt', "w") as f:
        f.write(f'Model Name: {MODEL_NAME}\n')
        f.write(f'Checkpoint Filename: {CKPT}\n')
        f.write(f'Inference Data Filename: {FILE}\n')
        f.write(f'Inference Pipeline Config: {CONFIG}\n')
        f.write(f'Inference Run Sample Size: {N}')

def load_comp_data(COMP_BASELINE_DATA,N):
    """
    Loads in comparison data csv file created using the save_output function.
    Inputs:
    - COMP_BASELINE_DATA: csv file with data for comparing to inference run model
    - N: sample size for analysis
    Returns:
    - comp_probs: jet flavour probabilities comparison model inference data
    - comp_pt_vals: jet pt values from inference comparison data
    - comp_eta_vals: jet eta values from inference comparison data
    - comp_truth_flavours: jet flavours from inference comparison data
    """
    comp_data = pd.read_csv(COMP_BASELINE_DATA)

    comp_data = comp_data.iloc[:N]

    comp_probs = comp_data.iloc[:,:4].to_numpy()
    comp_pt_vals = comp_data["pt"].to_numpy()
    comp_eta_vals = comp_data["eta"].to_numpy()
    comp_truth_flavours = comp_data["truth_flavour"].to_numpy()

    return comp_probs,comp_pt_vals,comp_eta_vals,comp_truth_flavours


