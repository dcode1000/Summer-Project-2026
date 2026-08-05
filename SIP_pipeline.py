#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
CKPT   = args.checkpoint # model checkpoint
CONFIG = args.config # Text file for configuring h5 test dataset

### pipeline configs
analysis_mode = args.analysis_mode # plots graphs and histograms, make false and turn on save_mode to just save the model
comp_mode = args.comp_mode # enables comparison with data input from a comparison pandas dataset
save_mode = args.save_mode # saves output jet flavour probabilities, pt, eta and truth flavours for comparison to another model


# In[5]:


### Running Inference
TEST_FILE,COMP_BASELINE_DATA,MODEL_NAME,NORM,N,STORED,low_pt_cutoff,high_pt_cutoff,config_dict = SIP_if.analysis_config_parser(CONFIG)
print(f'Config file: {CONFIG} Parsed')

data_dict,pad_dict,comp_probs,truth_flavours,pt_vals,eta_vals = SIP_if.h5_test_datafile_prepper(TEST_FILE,config_dict,N,STORED,low_pt_cutoff,high_pt_cutoff)
print(f'Test File: {TEST_FILE} Prepared')

model = SIP_if.model_loader(CKPT,NORM)
print(f'Checkpoint Model: {CKPT} Loaded')

probs = SIP_if.inference_run(data_dict,pad_dict,model)
print(f'Inference Run')

if save_mode:
    SIP_if.save_output(probs,pt_vals,eta_vals,truth_flavours,MODEL_NAME,CKPT,TEST_FILE,CONFIG,N)
    print(f'Data saved as: {MODEL_NAME}_inference_output.csv\nMetadata saved as: {MODEL_NAME}_inference_run_data.txt')


# In[6]:


if analysis_mode:
    counts_dict,confidences = SIP_af.jet_class_confidence_counter(probs,truth_flavours,0.3,0.01)
    if comp_mode ==True:
        comp_probs,comp_pt,comp_eta,comp_truth_flavours = SIP_if.load_comp_data(COMP_BASELINE_DATA, N)
        comp_counts_dict,comp_confidences = SIP_af.jet_class_confidence_counter(comp_probs,comp_truth_flavours,0.3,0.01)

    flav_classes = ["b","c","light","tau"]

    fig = plt.figure(figsize=(10,25), constrained_layout=True)
    gs = fig.add_gridspec(5,1)
    if comp_mode == True:
        SIP_af.profile_hist_plotter(fig,gs[0,0],pt_vals,probs,10,"$p_T$","Probability",MODEL_NAME,"GeV",
                             comp_mode=True,comp_x_data=comp_pt,comp_y_data=comp_probs)
        SIP_af.profile_hist_plotter(fig,gs[1,0],eta_vals,probs,10,rf"$\eta$","Probability",MODEL_NAME,None,
                             comp_mode=True,comp_x_data=comp_eta,comp_y_data=comp_probs)
        SIP_af.profile_hist_plotter_truth(fig,gs[2,0],pt_vals,probs,truth_flavours,10,"$p_T$","Probability",MODEL_NAME,"GeV",
                                   comp_mode=True,comp_x_data=comp_pt,comp_y_data=comp_probs,comp_truth_flavour=comp_truth_flavours)
        SIP_af.profile_hist_plotter_truth(fig,gs[3,0],eta_vals,probs,truth_flavours,10,rf"$\eta$","Probability",MODEL_NAME,None,
                                  comp_mode=True,comp_x_data=comp_eta,comp_y_data=comp_probs,comp_truth_flavour=comp_truth_flavours)
        SIP_af.predict_count_plotter(fig,gs[4,0],counts_dict,comp_confidences,MODEL_NAME,
                              comp_mode=True,comp_counts_dict=comp_counts_dict)
    else:
        SIP_af.profile_hist_plotter(fig,gs[0,0],pt_vals,probs,10,"$p_T$","Probability",MODEL_NAME,"GeV",comp_mode=False)
        SIP_af.profile_hist_plotter(fig,gs[1,0],eta_vals,probs,10,rf"$\eta$","Probability",MODEL_NAME,None,comp_mode=False)
        SIP_af.profile_hist_plotter_truth(fig,gs[2,0],pt_vals,probs,truth_flavours,10,"$p_T$","Probability",MODEL_NAME,"GeV",comp_mode=False)
        SIP_af.profile_hist_plotter_truth(fig,gs[3,0],eta_vals,probs,truth_flavours,10,rf"$\eta$","Probability",MODEL_NAME,None,comp_mode=False)
        SIP_af.predict_count_plotter(fig,gs[4,0],counts_dict,confidences,MODEL_NAME,comp_mode=False)
    fig.suptitle(f'{MODEL_NAME} Inference Run Analysis',fontsize=16)
    fig.savefig(f'{MODEL_NAME}_inference_report.png')
    plt.show()


# In[ ]:




