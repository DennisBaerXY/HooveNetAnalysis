import matplotlib.pyplot as plt
import os

from common.constants import PLOT_DIR


def plot_data(data, labels, title, xlabel, ylabel, file_name):
    """Helper function to plot and save data."""
    plt.figure(figsize=(12, 8))
    for datum, label in zip(data, labels):
        plt.plot(datum, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, file_name))
    plt.show()