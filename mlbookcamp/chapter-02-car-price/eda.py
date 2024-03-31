import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms(data):
    data.hist(edgecolor='black', linewidth=1.2)
    plt.pyplot.show()

def plot_correlations(data):
    plt.pyplot.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.pyplot.show()