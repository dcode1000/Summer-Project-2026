#!/usr/bin/env python
# coding: utf-8


### Salt Inference Pipeline - Inference Functions
#Repository for all data processing, model loading and inference running functions used in Salt Inference Pipeline

import h5py, hdf5plugin, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from salt.modelwrapper import ModelWrapper
import scipy.stats as stats
import yaml
from matplotlib.lines import Line2D

### Data Preparation
def config_parser(config_file, analysis_mode):
    """
    Config parser that takes in a .yaml config file and extracts appropriate information. For comparing multiple models, a separate list_dict is created for the
    variables of each model
    Inputs:
    - config_file: .yaml file containing SIP config information
    - analysis_mode: True if want to perform plotting based on the data produced during the inference run
    returns:
    - test_file: file used to run inference on the models
    - sample_size: sample size for analysis
    - stored: name of a model for comparison with the returns of the model with which inference is being run on
    - low_pt_cutoff: the minimum pt value a jet needs to have
    - high_pt_cutoff: the maximum pt value a jet can have
    - model_dict: dictionary of model dictionaries. Each dictionary contains model name, location of a checkpoint file of the trained model and list of variables used in training the
    model,  also contains a norm_dict of relevant variables for running inference on
    - plot_dict: dictionary containing configuration information for inference report
    """
    with open(config_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        test_file = cfg["test_file"]
        sample_size = cfg["sample_size"]
        stored = cfg["stored"]
        low_pt_cutoff = cfg["low_pt_cutoff"]
        high_pt_cutoff = cfg["high_pt_cutoff"]
        model_dict = cfg["models"]

        if analysis_mode:
            plot_dict = cfg["inference_report"]
            
    with open(cfg["norm_dict"]) as g:
        norm = yaml.load(g, Loader=yaml.FullLoader)
    
        for model_name, model in model_dict.items():
            
            norm_dict_subset = {} # creates a subset of the norm dictionary with only the training parameters
    
            for var_type, variables in model["inference_variables"].items():
    
                if var_type not in norm:
                    raise KeyError(
                        f"Variable type '{var_type}' for model '{model_name}' "
                        f"is missing from the normalization dictionary."
                    )
    
                norm_dict_subset[var_type] = {}
    
                for variable in variables:
    
                    if variable not in norm[var_type]:
                        raise KeyError(
                            f"Variable '{variable}' in variable type '{var_type}' "
                            f"for model '{model_name}' "
                            f"is missing from the normalization dictionary."
                        )
    
                    norm_dict_subset[var_type][variable] = norm[var_type][variable]

            subset_norm_file = f"{model_name}_norm.yaml"

            with open(subset_norm_file, "w") as f:
                yaml.safe_dump(norm_dict_subset, f)

            # Store the subset norm config file path rather than the dictionary in the model dictionary
            model["norm_dict"] = subset_norm_file
    if analysis_mode:
        return test_file, sample_size, stored, low_pt_cutoff, high_pt_cutoff, model_dict, plot_dict
    else:
        return test_file, sample_size, stored, low_pt_cutoff, high_pt_cutoff, model_dict


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
def model_loader(ckpt, norm_file):
    """
    Instantiates a model from a lightning wrapper and a lightning checkpoint. Requires no inputs except previously defined parameters.
    Inputs:
    - ckpt: .ckpt file containing trained model
    - norm_dict: dictionary containing normalisation parameters of all training variables
    Returns:
    - model: Is a loaded salt model of type ModelWrapper from salt-ml package
    Requires:
    - torch, torch lightning, salt-ml 
    """
    checkpoint = torch.load(ckpt,map_location="cpu",weights_only=False)
    
    hparams = checkpoint["hyper_parameters"]
    
    # Remove class weights from the loss configuration.
    # They are not needed for inference and cause jsonargparse
    # to reject the checkpoint because they are stored as lists.
    for task in hparams["model"]["init_args"]["tasks"]["init_args"]["modules"]:
        loss = task["init_args"].get("loss")
    
        if (
        loss is not None
        and loss.get("class_path") == "torch.nn.CrossEntropyLoss"):
            loss["init_args"]["weight"] = None
    
    # Get the normalization configuration
    norm_config = dict(hparams["norm_config"])
    
    # Replace the checkpoint's norm dictionary with the one selected by the user/configuration.
    norm_config["norm_dict"] = norm_file

    hparams["norm_config"] = norm_config

    # Remove the corresponding loss-weight buffers
    # from the checkpoint state_dict.
    state_dict = checkpoint["state_dict"]

    keys_to_remove = [
        key for key in state_dict
        if key.endswith(".loss.weight")
    ]

    for key in keys_to_remove:
        del state_dict[key]
    
    checkpoint["hyper_parameters"] = hparams
    checkpoint["state_dict"] = state_dict
    
    torch.save(checkpoint, "inference_checkpoint.ckpt")

    model = ModelWrapper.load_from_checkpoint("inference_checkpoint.ckpt", map_location="cpu", norm_config=norm_config).eval() # loading the model using the lightning method
                    # to load directly from the checkpoint, also setting the model to eval mode
    return model

### Running Inference
def inference_run(data_dict,pad_dict,loaded_model, model_dict):
    """
    runs inference for a trained model (called model) and returns the probabilities applied by softmax. Can also return scores if scores==True
    inputs:
    - data_dict: dictionary of tensors containing samples and processed datasets with columns corresponding to the training inputs of the model
    - pad_dict: dictionary telling model where padding was applied
    - loaded_model: model loaded from checkpoint for running inference on
    - model_dict: dictionary of parameters relating to the model
    returns:
    - probs: array containing 4 probabilities for each jet with the probabilities corresponding to the probability that the jet is each of the 4 flavours
            in order (b, c , light, tau)
    - scores: original 4 scores for each jet returned by model before softmax applied
    """
    with torch.no_grad():  # running inference using the model
        
        scores = loaded_model(data_dict,pad_dict) # getting scores

    probs = torch.softmax(scores[0]["jets"][model_dict["inference_task"]], -1).numpy() #softmax converts to probability
    scores = scores[0]["jets"][model_dict["inference_task"]].numpy()

    return probs, scores

def scores_unnormaliser(model_dict):
    """
    for models regressed directly onto GN2 probabilities, will unnormalise the data so that it will be comparable to softmaxed probabilities
    inputs:
    - model_dict: dictionary of information related to model of interest
    returns:
    - model_dict: updates model dict probs so that they have had their normalisation reversed - for comparison with models that output softmaxed probabilities
    """
    for n in range(len(model_dict["scores"][0,:])):
        model_dict["probs"][:,n] = (model_dict["scores"][:,n])*model_dict["scores_stds"][n] + model_dict["scores_means"][n]
    print(model_dict["probs"])

    return model_dict

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
    data["pt (GeV)"] = pt_vals
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


