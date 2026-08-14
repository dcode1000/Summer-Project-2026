#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Salt Inference Pipeline - Analysis Functions
# All functions used to analyse and plot and compare model outputs
import h5py, hdf5plugin, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from salt.modelwrapper import ModelWrapper
import scipy.stats as stats
from matplotlib.lines import Line2D
### Jet Classification
flav_classes = ["b","c","light","tau"]
# classify jets from model probabilities

def jet_classifier(probs_array,confidence_threshold):
    """
    is a function which returns a list of classified jets from their model predicted probabilities if the probability for a given jet is above 
    the confidence threshold
    inputs:
    probs_array: array with each column representing model predicted probability of a given jet being a given flavour
    confidence_threshold: a given flavour probability for a jet has to be above this threshold to be classified as that flavour - if no probabilities are
    above the threshold it is unclassified
    returns:
    classed_jets: is a list of jets which had a probability which allowed classification
    classed_jets_indices: is the row indices of the original probability array corresponding to each classified jet - allowing for comparison with eg
    truth values
    """

    max_probabilities = np.max(probs_array, axis=1)     #  Get the highest probability value for each row
    classed_jets_indices = (np.where(max_probabilities >= confidence_threshold))[0] #Apply threshold: Keep index if >= threshold, else assign fallback (-1)
    confident_predictions = np.argmax(probs_array[classed_jets_indices],axis=1) # flavour indices of probabilities which are above the threshold ie 0,1,2,3

    labels = {0:"b",1:"c",2:"light",3:"tau"} # for the returned flavour index - relationship for appropriate flavour
    classed_jets = np.array([labels[x] for x in confident_predictions]) # assigns flavour based on index number using labels

    return classed_jets, classed_jets_indices

# organising model classed jets by their truth flavour

def jet_class_confidence_counter(probs_array,truth_vals,low_threshold,interval):
    """
    Over a range of confidence thresholds counts the number of classified.
    Takes in a low threshold and an interval and works up to a confidence of 1. The model produces a probability of each jet corresponding to each
    jet flavour, these jets are only flavour-classified as such if the probability of them being that flavour is above a given confidence threshold.
    This graph compares all such flavour classified jets at each confidence threshold organised by the truth flavour of that jet.
    Inputs:
    probs_array: array of model produced probabilities - 4 probabilities for each jet
    truth_vals: array of true flavours of jets
    low_threshold: lowest confidence threshold wish to test (>0.25 makes sense)
    interval: interval between confidences to be tested
    Returns:
    counts_dict: a dictionary organised by predicted flavour of counts of jets of different truth flavours 
    confidences: list of confidence thresholds used to classify jets
    """
    confidences = np.arange(low_threshold,1,interval) # creating an array of confidence values

    jet_flavours = ["b", "c", "light", "tau"] # flavour labels in order

    counts_dict = {flavour: [] for flavour in jet_flavours} # creating an empty dictionary which will have four keys corresponding to the 4 jet flavours

    for confidence in confidences: # looping over each confidence threshold

        classed_jets, classed_jets_indices = jet_classifier(probs_array, confidence) # classifies jets by highest probability for a given 
                                                                # confidence interval, returns indices of those classified and their classification
        corresponding_truths = truth_vals[classed_jets_indices] # are truth flavours corresponding to the classified jets


        for classified_flavour in jet_flavours:
            # Keep only jets classified as this flavour
            flavour_mask = (classed_jets == classified_flavour)
            truths = corresponding_truths[flavour_mask]
            jet_count = len(truths)
            confidence_counts = []

            # Count each truth flavour
            for truth_flavour in jet_flavours:
                confidence_counts.append(np.sum(truths == truth_flavour) / jet_count)

            counts_dict[classified_flavour].append(confidence_counts)
    return counts_dict, confidences

### histogramming functions

def binner(x_data, bin_number):
    """
    Determines bin widths required to ensure equal density of profile histogram
    Inputs:
    - x_data: data which determines binning
    - bin_number: number of bins to bin x_data into
    Returns:
    - bin_centers: centres of all bins
    - bin_edges: edges of all bins
    - bin_hwidth: half width of the bins
    """
    bin_edges = np.quantile(x_data,np.linspace(0,1,bin_number+1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
    bin_hwidth = (bin_edges[1:] - bin_edges[:-1]) / 2.  # half-width of each bin

    return bin_edges, bin_centers, bin_hwidth

def data_histogrammer(x_data,y_data,bins):
    """
    Separates data into bins, determined by binner function, for use in a profile histogram
    Inputs:
    - x_data: data on x axis which determines the binning
    - y_data: data on a profile histogram
    - bins: the edges of the bins the data is to be binned into
    Returns:
    - means: mean values in each bin
    - counts: counts in each bin
    - stds: standard deviations in each bin
    - sems: standard errors on each bin
    """
    means = stats.binned_statistic(x_data, y_data, bins=bins, statistic='mean').statistic
    counts = stats.binned_statistic(x_data, y_data, bins=bins, statistic='count').statistic
    stds = stats.binned_statistic(x_data, y_data, bins=bins, statistic='std').statistic
    sems = stds/(np.sqrt(counts))

    return means, sems

### Profile Histogram

def profile_hist_plotter(fig,gs_cell,model_dict,x_data,y_data,bin_number):
    """
    creates an approximate profile histogram by plotting the mean and sem of the y data at the midpoint of the binned x data
    Inputs
    - model_dict: dictionary containing all loaded models and relevant information
    - x_data: specifies plotting a profile histogram with either pt or eta
    - y_data: specifies whether to plot a profile histogram with either probabilities or scores
    - bin_number: number of bins to separate the data into
    """
    inner = gs_cell.subgridspec(3,2,height_ratios=[0.15, 1, 1])
    title_ax = fig.add_subplot(inner[0, :])
    title_ax.axis("off")

    flav_classes = ["b","c","light","tau"]
    
    plot_config = {
        "pt": {
            "label": r"$p_T$",
            "units": "(GeV)",
            "values": lambda model: model["pt_vals"],
        },
        "eta": {
            "label": r"$\eta$",
            "units": "",
            "values": lambda model: model["eta_vals"],
        },
        "probability": {
            "label": "Tag Probability",
            "values": lambda model: model["probs"],
        },
        "scores": {
            "label": "Tag Probability",
            "values": lambda model: model["scores"],
        }
    }
    

    x_config = plot_config[x_data]
    x_name = x_config["label"]
    x_units = x_config["units"]
    
    y_config = plot_config[y_data]
    y_name = y_config["label"]

    source_handles = [Line2D([0], [0],color=model_dict[model_name]["plot_colour"],lw=2,marker="^",markersize=5,label=f"{model_name}",)
                      for model_name in model_dict]

    title_ax.set_title(rf'Model Tagging Probability For Each Flavour Against {x_name}')
    
    for n in range(4):
        ax = fig.add_subplot(inner[1+n // 2, n % 2])
        for model_name, model in model_dict.items():
            x_vals = x_config["values"](model)
            y_vals = y_config["values"](model)
    
            y = y_vals[:,n]
            bin_edges, bin_centers, bin_hwidth = binner(x_vals,bin_number)
            means,sems = data_histogrammer(x_vals,y,bin_edges)
            
            ax.errorbar(x=bin_centers, y=means, xerr=bin_hwidth, yerr=sems, linestyle='none', 
                        marker='^',ms=5,color=model["plot_colour"],capsize=2)
            if n == 0:
                ax.legend(
                    handles=source_handles,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    fontsize=8,)

        ax.set_ylim(0,1)
        ax.set_title(rf'{flav_classes[n]}-jet')
        ax.set_xlabel(rf"{x_name} {x_units}")
        ax.set_ylabel(rf"{y_name}")
        ax.grid()

### Truth Sorted Profile Histogram

def profile_hist_plotter_truth(fig,gs_cell,model_dict,x_data,y_data,bin_number):
    """
    creates an approximate profile histogram by plotting the mean and sem of the y data at the midpoint of the binned x data, sorted by truth value of jets
    Inputs
    - model_dict: dictionary containing all loaded models and relevant information
    - x_data: specifies plotting a profile histogram with either pt or eta
    - y_data: specifies whether to plot a profile histogram with either probabilities or scores
    - bin_number: number of bins to separate the data into
    Returns:
    Subplots organised by model predicted flavour (is predicted flavour of training model if in comp mode)
    Requires:
    matplotlib, matplotlib.lines import Line2D, numpy, scipy
    """
    inner = gs_cell.subgridspec(3,2,height_ratios=[0.15, 1, 1])

    marker_dict = {"b":"o","c":"^","light":"s","tau":"x"}
    flav_classes = ["b","c","light","tau"]
    
    plot_config = {
        "pt": {
            "label": r"$p_T$",
            "units": "(GeV)",
            "values": lambda model: model["pt_vals"],
        },
        "eta": {
            "label": r"$\eta$",
            "units": "",
            "values": lambda model: model["eta_vals"],
        },
        "probability": {
            "label": "Tag Probability",
            "values": lambda model: model["probs"],
        },
        "scores": {
            "label": "Tag Probability",
            "values": lambda model: model["scores"],
        }
    }
    x_config = plot_config[x_data]
    x_name = x_config["label"]
    x_units = x_config["units"]
    
    y_config = plot_config[y_data]
    y_name = y_config["label"]

    title_ax = fig.add_subplot(inner[0, :])
    title_ax.axis("off")
    title_ax.set_title(rf'Tagging Probability By Truth Flavour Against {x_name}')
    
    for n in range(4):
        ax = fig.add_subplot(inner[1+n // 2, n % 2])
        for model_name, model in model_dict.items():
            i = 0
            for flavour in np.unique(model["truth_flavour"]):
                x_vals = x_config["values"](model)
                y_vals = y_config["values"](model)
                bin_edges, bin_centers, bin_hwidth = binner(x_vals,bin_number)
                truth_mask = (model["truth_flavour"] == flavour)
                x_j = x_vals[truth_mask]
                y_j = y_vals[truth_mask,n]
                
                means,sems = data_histogrammer(x_j,y_j,bin_edges)
    
                ax.errorbar(x=bin_centers, y=means,xerr=bin_hwidth, yerr=sems, linestyle=" ", marker=marker_dict[flavour],
                            ms=6,color=model["plot_colour"],capsize=2)

            ax.set_xlabel(rf'{x_name} {x_units}')
            ax.set_ylabel(rf'{y_name}')
            ax.set_ylim(0,1)
            ax.set_title(rf'Model Predicted {flav_classes[n]}-jet')

            i +=1
        if n == 0:
        
            # Legend for truth flavour (marker style)
            truth_handles = [Line2D([0], [0],color="tab:blue",marker=marker_dict[flavour],linestyle="None",markersize=6,label=flavour,)
                for i, flavour in enumerate(np.unique(model["truth_flavour"]))]
        
            truth_legend = ax.legend(
                handles=truth_handles,
                title="Jet Truth Flavour",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=8,
            )
        
            # Legend for model (colour)
            source_handles = [Line2D([0], [0],color=model_dict[model_name]["plot_colour"],marker="o",linestyle="None",markersize=5,label=model_name,)
                for model_name in model_dict]
        
            source_legend = ax.legend(handles=source_handles,title="Model",loc="center left",bbox_to_anchor=(1.02, 0.0),fontsize=8,)
            # Keep both legends
            ax.add_artist(truth_legend)
        ax.grid()

### plotting classed jets by truth flavour by confidence threshold

def predict_count_plotter(fig,gs_cell,model_dict):
    """
    for an input dictionary of flavour counts of various labels, plots a graph for each predicted flavour with a line for the counts at each confidence
    organised by truth label
    inputs:
    model_dict: dictionary of trained models and associated data
    returns:
    plot with 4 subplots
    Requires:
    matplotlib, matplotlib.lines import Line2D, numpy, scipy
    """
    inner = gs_cell.subgridspec(3,2,height_ratios=[0.10, 1, 1])
    line_dict = {"b":"solid","c":"dotted","light":"dashed","tau":"dashdot"}

    title_ax = fig.add_subplot(inner[0, :])
    title_ax.axis("off")
    title_ax.set_title("Model Predicted Jet Classification By Jet Truth Flavour\n""Against Classification Threshold")
    
    jet_flavours = ["b", "c", "light", "tau"]
    for n in range(4):
        ax = fig.add_subplot(inner[1+n// 2, n % 2])
        for model_name, model in model_dict.items():
            for i in range(len(jet_flavours)):
                flav_counts = [row[i] for row in model["counts_dict"][jet_flavours[n]]]
                ax.plot(model["confidences"],flav_counts,label=jet_flavours[i],color=model["plot_colour"],linestyle=line_dict[jet_flavours[i]])
            ax.set_title(rf'Model Predicted {jet_flavours[n]}-jet')
            ax.set_xlabel("Classification Threshold")
            ax.set_ylabel("Fraction of Classified Jets")
            ax.set_ylim(-0.01,1.01)
        if n == 0:
            truth_handles = [Line2D([0], [0],color="tab:blue",linestyle=line_dict[flavour],markersize=6,label=flavour,)
                for i, flavour in enumerate(np.unique(model["truth_flavour"]))]
        
            truth_legend = ax.legend(
                handles=truth_handles,
                title="Jet Truth Flavour",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=8,
            )
        
            # Legend for model (colour)
            source_handles = [Line2D([0], [0],color=model_dict[model_name]["plot_colour"],marker="o",linestyle="None",markersize=5,label=model_name,)
                for model_name in model_dict]
        
            source_legend = ax.legend(handles=source_handles,title="Model",loc="center left",bbox_to_anchor=(1.02, 0.0),fontsize=8,)
            # Keep both legends
            ax.add_artist(truth_legend)
        ax.grid()

