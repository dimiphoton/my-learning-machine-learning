import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def plot_histograms(data):
    """
    Plots histograms for the given data.

    Parameters:
    data (pandas.DataFrame): The data to plot histograms for.

    Returns:
    None
    """
    data.hist(edgecolor='black', linewidth=1.2)
    plt.show()

def plot_correlations(data):
    """
    Plots a correlation heatmap for the given data.

    Parameters:
    data (DataFrame): The input data to calculate correlations and plot the heatmap.

    Returns:
    None
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.show()
