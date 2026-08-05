#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Salt Inference Pipeline - Analysis Functions
# All functions used to analyse and plot and compare model outputs
import h5py, numpy as np, pandas as pd, torch
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
def profile_hist_plotter(fig,gs_cell,x_data,y_data,bin_number,x_name,y_name,model_name,x_units,comp_mode=False,comp_x_data=None,comp_y_data=None):
    """
    creates an approximate profile histogram by plotting the mean and sem of the y data at the midpoint of the binned x data
    Inputs
    - x_data: data that will be histogrammed along the x axis
    - y_data: data which will provide the mean and sem values for the y axis
    - bin_number: number of bins to separate the data into
    - x_name: name of x data to be plotted
    - y_name: name of y data to be plotted
    - model_name: name of inference run model
    - x_units: units of x_data
    - comp_mode: determines whether there is a baseline model output to be plotted for comparison against the training model output
    - comp_x_data: comparison x data, of the same type as x_data
    - comp_y_data: comparison y data of the same type as y_data
    """
    inner = gs_cell.subgridspec(3,2,height_ratios=[0.15, 1, 1])
    title_ax = fig.add_subplot(inner[0, :])
    title_ax.axis("off")
    title_ax.set_title(rf'{model_name} Predicted Jet Flavour Probability Against {x_name}')

    for n in range(4):
        y = y_data[:,n]
        bin_edges, bin_centers, bin_hwidth = binner(x_data,bin_number)
        means,sems = data_histogrammer(x_data,y,bin_edges)
        ax = fig.add_subplot(inner[1+n // 2, n % 2])
        ax.errorbar(x=bin_centers, y=means, xerr=bin_hwidth, yerr=sems, linestyle='none', marker='^',ms=5,color="cornflowerblue",capsize=2)

        if comp_mode:
            comp_y = comp_y_data[:,n]
            means_comp,sems_comp = data_histogrammer(x_data,comp_y,bin_edges)
            ax.errorbar(x=bin_centers, y=means_comp, xerr=bin_hwidth, yerr=sems_comp, linestyle='none', marker='^',ms=5,color="orange",capsize=2)
            if n == 0:
                source_handles = [Line2D([0], [0], color='cornflowerblue', lw=2, label=f'{model_name} Output'),
                                Line2D([0], [0], color='orange', lw=2, label='Baseline Comparison \n Model'),]
                ax.legend(handles=source_handles,loc="center left",bbox_to_anchor=(1.02, 0.5),fontsize=8)

        if x_units == None:
            ax.set_xlabel(rf'{x_name}')
        else:
            ax.set_xlabel(rf'{x_name} ({x_units})')

        ax.set_ylabel(rf'{y_name}')
        ax.set_title(rf'{model_name} Predicted {flav_classes[n]}-jet Probabilities')
        ax.grid()

### Truth Sorted Profile Histogram

def profile_hist_plotter_truth(fig,
                               gs_cell,
                               x_data,
                               y_data,
                               truth_flavour,
                               bin_number,
                               x_name,
                               y_name,
                               model_name,
                               x_units,
                               comp_mode=False,
                               comp_x_data=None,
                               comp_y_data=None,
                               comp_truth_flavour=None):
    """
    creates an approximate profile histogram by plotting the mean and sem of the y data at the midpoint of the binned x data, sorted by truth value of jets
    Inputs
    - x_data: data that will be histogrammed along the x axis
    - y_data: data which will provide the mean and sem values for the y axis
    - truth_flavour: actual flavour each jet corresponds to
    - bin_number: number of bins to separate the data into
    - x_name: name of x data to be plotted
    - y_name: name of y data to be plotted
    - model_name: name of inference run model
    - x_units: units of x_data
    - comp_mode: determines whether there is a baseline model output to be plotted for comparison against the training model output
    - comp_x_data: comparison x data, of the same type as x_data
    - comp_y_data: comparison y data of the same type as y_data
    - comp_truth_flavour: comparison truth flavour of all jets in comp data
    Returns:
    Subplots organised by model predicted flavour (is predicted flavour of training model if in comp mode)
    Requires:
    matplotlib, matplotlib.lines import Line2D, numpy, scipy
    """
    inner = gs_cell.subgridspec(3,2,height_ratios=[0.15, 1, 1])
    marker_styles = ["o","^","s","x"]
    linestyles = ["solid","dotted","dashed","dashdot"]

    title_ax = fig.add_subplot(inner[0, :])
    title_ax.axis("off")
    title_ax.set_title(rf'{model_name} Jet Flavour Probability Grouped By Truth Flavour \n Against {x_name}')

    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    truth_flavour = np.asarray(truth_flavour)

    bin_edges, bin_centers, bin_hwidth = binner(x_data,bin_number)


    for n in range(4):
        ax = fig.add_subplot(inner[1+n // 2, n % 2])
        i = 0
        for flavour in np.unique(truth_flavour):
            truth_mask = (truth_flavour == flavour)
            x_j = x_data[truth_mask]
            y_j = y_data[truth_mask,n]

            means,sems = data_histogrammer(x_j,y_j,bin_edges)

            ax.errorbar(x=bin_centers, y=means,xerr=bin_hwidth, yerr=sems, linestyle=" ", marker=marker_styles[i],
                        ms=8,color="cornflowerblue",capsize=2,label=flavour)
            if comp_mode:
                comp_truth_mask = (comp_truth_flavour == flavour)
                comp_x_j = comp_x_data[comp_truth_mask]
                comp_y_j = comp_y_data[comp_truth_mask,n]

                comp_means,comp_sems = data_histogrammer(comp_x_j,comp_y_j,bin_edges)

                ax.errorbar(x=bin_centers, y=comp_means,xerr=bin_hwidth, yerr=comp_sems, linestyle=" ", marker=marker_styles[i]
                            ,ms=8,color="orange",capsize=2,alpha=0.8)
            if x_units == None:
                ax.set_xlabel(rf'{x_name}')
            else:
                ax.set_xlabel(rf'{x_name} ({x_units})')
            ax.set_ylabel(rf'{y_name}')
            ax.set_title(rf'Model Predicted {flav_classes[n]}-jet Probabilities')

            i +=1
        if n == 0:
            data_legend = ax.legend(title="Jet Truth Flavour",loc="center left",bbox_to_anchor=(1.02, 0.5),fontsize=8,)

            if comp_mode:
                source_handles = [Line2D([0], [0], color="cornflowerblue", marker="o",linestyle="None", label=f'{model_name} Output'),
                                Line2D([0], [0], color="orange", marker="o",linestyle="None", label="Baseline Comparison \n Model Output"),]

                source_legend = ax.legend(handles=source_handles,loc="center left",bbox_to_anchor=(1.02, 0),fontsize=8,)
                ax.add_artist(data_legend)
        ax.grid()

### plotting classed jets by truth flavour by confidence threshold

def predict_count_plotter(fig,gs_cell,counts_dict,confidences,model_name,comp_mode=False,comp_counts_dict=None):
    """
    for an input dictionary of flavour counts of various labels, plots a graph for each predicted flavour with a line for the counts at each confidence
    organised by truth label
    inputs:
    counts_dict: dictionary of counts - each row is counts for all four truth flavours, each key corresponds to a different predicted flavour
    confidences: list of confidence thresholds used to determine counts_dict
    model_name: name of inference run model 
    comp_mode: determines if comparing inference output to a baseline model
    comp_counts_dict: if comp_mode == True is the counts dict for the comparison data
    returns:
    plot with 4 subplots
    Requires:
    matplotlib, matplotlib.lines import Line2D, numpy, scipy
    """
    inner = gs_cell.subgridspec(3,2,height_ratios=[0.10, 1, 1])
    linestyles = ["solid","dotted","dashed","dashdot"]

    title_ax = fig.add_subplot(inner[0, :])
    title_ax.axis("off")
    title_ax.set_title(f'{model_name} Jet Flavour Classification Grouped By Jet Truth Flavour\nAgainst Classification Threshold')

    jet_flavours = ["b", "c", "light", "tau"]
    for n in range(4):
        ax = fig.add_subplot(inner[1+n// 2, n % 2])
        for i in range(len(jet_flavours)):
            flav_counts = [row[i] for row in counts_dict[jet_flavours[n]]]
            ax.plot(confidences,flav_counts,label=jet_flavours[i],color="cornflowerblue",linestyle=linestyles[i])
            if comp_mode:
                comp_flav_counts = [row[i] for row in comp_counts_dict[jet_flavours[n]]]
                ax.plot(confidences,comp_flav_counts,color="orange",linestyle=linestyles[i]) 
        ax.set_title(rf'Model Predicted {jet_flavours[n]}-jet')
        ax.set_xlabel("Classification Threshold")
        ax.set_ylabel("Fraction of Jets Classified")
        if n == 0:
            data_legend = ax.legend(title="Jet Truth Flavour",loc="center left",bbox_to_anchor=(1.02, 0.5),fontsize=8,)

            if comp_mode:
                source_handles = [Line2D([0], [0], color="cornflowerblue", marker="o",linestyle="None", label=f'{model_name} Output'),
                                Line2D([0], [0], color="orange", marker="o",linestyle="None", label="Baseline Comparison \n Model Output"),]

                source_legend = ax.legend(handles=source_handles,loc="center left",bbox_to_anchor=(1.02, 0),fontsize=8,)
                ax.add_artist(data_legend)

        ax.grid()

