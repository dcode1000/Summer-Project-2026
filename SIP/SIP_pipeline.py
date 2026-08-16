#!/usr/bin/env python
# coding: utf-8

### SIP Inference Pipeline

### Required Packages
import h5py, hdf5plugin, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from salt.modelwrapper import ModelWrapper
import scipy.stats as stats
from matplotlib.lines import Line2D

### importing functions
import SIP_inference_functions as SIP_if
import SIP_analysis_functions as SIP_af

### overriding warning messages
import warnings 
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")  
warnings.filterwarnings("ignore", message="Cannot use flash-varlen backend. No GPU available. Reverting to torch-math.")


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--checkpoint", required=True,
                    help="Checkpoint file for model being loaded and analysed.")

parser.add_argument("--config", required=True,
                    help="Inference pipeline config .txt file.")

parser.add_argument("--analysis_mode", action="store_true")
parser.add_argument("--comp_mode", action="store_true")
parser.add_argument("--save_mode", action="store_false")

args = parser.parse_args()

### pipeline input
CONFIG = args.config # Text file for configuring h5 test dataset

### pipeline configs
analysis_mode = args.analysis_mode # plots graphs and histograms, make false and turn on save_mode to just save the model
save_mode = args.save_mode # saves output jet flavour probabilities, pt, eta and truth flavours for comparison to another model

### Running Inference
if analysis_mode:
    test_file, sample_size, stored, low_pt_cutoff, high_pt_cutoff, model_dict, plot_dict = config_parser(CONFIG, analysis_mode)
else:
    test_file, sample_size, stored, low_pt_cutoff, high_pt_cutoff, model_dict = config_parser(CONFIG, analysis_mode)

print(f'Config file: {CONFIG} Parsed')
for model_name,model in model_dict.items():
    print("")
    print(f'Running Inference On Model: {model_name}')
    data_dict,pad_dict,comp_probs,truth_flavours,pt_vals,eta_vals = h5_test_datafile_prepper(test_file,model["inference_variables"]
                                                                                             ,sample_size,stored,low_pt_cutoff,high_pt_cutoff)
    model["data_dict"] = data_dict
    model["pad_dict"] = pad_dict
    model["comp_probs"] = comp_probs
    model["truth_flavour"] = truth_flavours
    model["pt_vals"] = pt_vals
    model["eta_vals"] = eta_vals
    
    print(f'Test File: {test_file} Prepared')
    ckpt, norm = model["model_checkpoint"], model["norm_dict"]
    loaded_model = model_loader(ckpt, norm)
    print(f'Checkpoint Model: {model["model_checkpoint"]} Loaded')
    
    probs,scores = inference_run(data_dict,pad_dict,loaded_model,return_scores=True)
    print(f'Inference Run')
    
    model["probs"] = probs
    model["scores"] = scores
    del model["pad_dict"]["REGISTERS"]
    
    if save_mode:
        save_output(probs,pt_vals,eta_vals,truth_flavours,model_name,ckpt,test_file,CONFIG,sample_size)
        print(f'Data saved as: {model_name}_inference_output.csv\nMetadata saved as: {model_name}_inference_run_data.txt')

print("")
print(f'Inference Complete ')

if analysis_mode:
    n_plots = len(plot_dict["plots"])
    fig = plt.figure(figsize=(10,5*n_plots), constrained_layout=True)
    gs = fig.add_gridspec(5,1)
    for model_name,model in model_dict.items():
        model["counts_dict"],model["confidences"] = SIP_af.jet_class_confidence_counter(model["probs"],model["truth_flavour"],0.3,0.01)
    i = 0
    for plot in plot_dict["plots"].values():
        if plot["plot_type"] == "profile_histogram":
            SIP_af.profile_histogram(fig,gs[i,0],model_dict,plot["x_data"],"probability",10)
        elif plot["plot_type"] == "profile_histogram_truth":
            SIP_af.profile_histogram_truth(fig,gs[i,0],model_dict,plot["x_data"],"probability",10)
        elif plot["plot_type"] == "prediction_plot":
            SIP_af.prediction_plot(fig,gs[i,0],model_dict)
        i += 1
    fig.suptitle(plot_dict["report_title"],fontsize=16)
    fig.savefig(f'{model_name}_inference_report.png')
    plt.show()




