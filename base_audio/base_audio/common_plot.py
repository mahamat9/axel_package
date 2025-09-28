# -*- coding: utf-8 -*-
"""
Created on July 2024 (from old code)

@author: a23marmo

Defining common plotting functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# %% Plotting utils
def plot_me_this_spectrogram(spec, title = "Spectrogram", x_axis = "x_axis", y_axis = "y_axis", invert_y_axis = True, cmap = cm.Greys, figsize = None, norm = None, vmin = None, vmax = None):
    """
    Plots a spectrogram in a colormesh.
    """
    if figsize != None:
        plt.figure(figsize=figsize)
    elif spec.shape[0] == spec.shape[1]:
        plt.figure(figsize=(7,7))
    padded_spec = spec #pad_factor(spec)
    plt.pcolormesh(np.arange(padded_spec.shape[1]), np.arange(padded_spec.shape[0]), padded_spec, cmap=cmap, norm = norm, vmin = vmin, vmax = vmax, shading='auto')
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    if invert_y_axis:
        plt.gca().invert_yaxis()
    plt.show()
    
def pad_factor(factor):
    """
    Pads the factor with zeroes on both dimension.
    This is made because colormesh plots values as intervals (post and intervals problem),
    and so discards the last value.
    """
    padded = np.zeros((factor.shape[0] + 1, factor.shape[1] + 1))
    for i in range(factor.shape[0]):
        for j in range(factor.shape[1]):
            padded[i,j] = factor[i,j]
    return np.array(padded)

def plot_spec_with_annotations(factor, annotations, color = "yellow", title = None):
    """
    Plots a spectrogram with the segmentation annotation.
    """
    if factor.shape[0] == factor.shape[1]:
        plt.figure(figsize=(7,7))
    plt.title(title)
    padded_fac = pad_factor(factor)
    plt.pcolormesh(np.arange(padded_fac.shape[1]), np.arange(padded_fac.shape[0]), padded_fac, cmap=cm.Greys)
    plt.gca().invert_yaxis()
    for x in annotations:
        plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = color)
    plt.show()
    
def plot_spec_with_annotations_abs_ord(factor, annotations, color = "green", title = None, cmap = cm.gray):
    """
    Plots a spectrogram with the segmentation annotation in both x and y axes.
    """
    if factor.shape[0] == factor.shape[1]:
        plt.figure(figsize=(7,7))
    plt.title(title)
    padded_fac = pad_factor(factor)
    plt.pcolormesh(np.arange(padded_fac.shape[1]), np.arange(padded_fac.shape[0]), padded_fac, cmap=cmap)
    plt.gca().invert_yaxis()
    for x in annotations:
        plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = color)
        plt.plot([0,len(factor)], [x,x], '-', linewidth=1, color = color)
    plt.show()

def plot_spec_with_annotations_and_prediction(factor, annotations, predicted_ends, title = "Title"):
    """
    Plots a spectrogram with the segmentation annotation and the estimated segmentation.
    """
    if factor.shape[0] == factor.shape[1]:
        plt.figure(figsize=(7,7))
    plt.title(title)
    padded_fac = pad_factor(factor)
    plt.pcolormesh(np.arange(padded_fac.shape[1]), np.arange(padded_fac.shape[0]), padded_fac, cmap=cm.Greys)
    plt.gca().invert_yaxis()
    for x in annotations:
        plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = "#8080FF")
    for x in predicted_ends:
        if x in annotations:
            plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = "green")#"#17becf")
        else:
            plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = "orange")
    plt.show()
    