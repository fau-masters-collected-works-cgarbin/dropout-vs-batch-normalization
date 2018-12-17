"""Plot accuracy and loss data generated by Keras models fit() function.

Data is read from a JSON file, crated with json.dump(history).
"""
import json
import os
import glob
import re
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


def parse_command_line():
    """Parse command line parameters and return them."""
    ap = ArgumentParser(description='Plot Keras history from a JSON file.')
    ap.add_argument('--directory', type=str)
    ap.add_argument('--pattern', type=str)
    args = ap.parse_args()

    return args.directory, args.pattern


def get_title(file_name):
    """Create a title by extracting pieces of the file name."""

    if 'mlp' in file_name:
        network = re.search(r'nw=(.*?)_', file_name).group(1)
        optimizer = re.search(r'opt=(.*?)_', file_name).group(1)
        hidden_layers = re.search(r'hl=(.*?)_', file_name).group(1)
        units_per_layer = re.search(r'uhl=(.*?)_', file_name).group(1)
        epochs = re.search(r'e=(.*?)_', file_name).group(1)
        learning_rate = re.search(r'lr=(.*?)_', file_name).group(1)
        weight_decay = re.search(r'_d=(.*?)_', file_name).group(1)

        # Nicer text for humans for title and optimizer
        pretty_network = {'standard': 'Standard',
                          'dropout': 'Dropout',
                          'dropout_no_adjustment': 'Droput w/o adjustment',
                          'batch': 'Batch normalization'}
        pretty_optimizer = {'sgd': 'SGD', 'rmsprop': 'RMSProp'}

        # Note: three lines is the most I was able to fit with the standard
        # matlibplot title formatting (there are solutions to fit more lines,
        # but none are simple - at least the ones I could find)
        title = ('{} network, {} optimizer, trained for {} epochs\n'
                 '{} hidden layers, {} units per layer \n'
                 'Learning rate = {}, weight decay = {}').format(
            pretty_network[network], pretty_optimizer[optimizer], int(epochs),
            int(hidden_layers), int(units_per_layer), float(learning_rate),
            float(weight_decay))

        return title

    if 'cnn' in file_name:
        network = re.search(r'cifar_10_cnn_(.*?)_lr', file_name).group(1)
        learning_rate = re.search(r'lr=(.*?)_', file_name).group(1)
        units_dense_layer = re.search(r'udl=(.*?)_', file_name).group(1)
        epochs = re.search(r'e=(.*?)_', file_name).group(1)

        # Nicer text for humans for title and optimizer
        pretty_network = {'plain': 'Plain',
                          'dropout': 'Dropout',
                          'batch_normalization': 'Batch Normalization',
                          'batchnorm_dropout': 'Droput + Batch normalization'}

        # Note: three lines is the most I was able to fit with the standard
        # matlibplot title formatting (there are solutions to fit more lines,
        # but none are simple - at least the ones I could find)
        title = ('{} network, trained for {} epochs\n'
                 '{} units in dense layer, learning rate = {}').format(
            pretty_network[network], int(epochs),
            int(units_dense_layer), float(learning_rate))

        return title

    assert False  # Can't parse this file name


def get_max_y(file_name):
    """Get max y value to allow comparison of results (same scale)."""

    # These values were determined by hand. We could be fancy and read
    # all data files first to determine max y, but too much work for low
    # return at this point.
    if 'mlp' in file_name:
        return 0.5
    else:
        # Assume it's the CNN test
        return 3.5


def plot_history(history, file_name, show):
    """Plot the loss history created from druing the execution of Keras fit().

    Arguments:
      history {[dataframe]} -- The history data from the call to fit()
      file {[string]} -- Name of the input file (the one with the history)
      show {[Boolean]} -- True to also show on screen, False to just save it
    """

    # Extract only what we need to plot
    history = history[['loss', 'val_loss']]

    # Rename columns to a more user-friendly text
    history = history.rename(columns={'loss': 'Training loss',
                                      'val_loss': 'Validation Loss'})

    # Style with default seaborn, then change background (easier to read)
    sns.set()
    sns.set_style('white')

    # Fix the y axis scale for all graphs so we can compare tests
    plt.ylim(0, get_max_y(file_name))

    # Plot, add title, labels, etc.
    sns.lineplot(data=history)
    plt.title(get_title(file_name))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(frameon=False)
    plt.grid(True, axis='x', linestyle='dotted')

    # Save to disk as a .png file
    png_file = file_name.replace('.json', '.png')
    plt.savefig(png_file)

    if show:
        plt.show()


def plot_all_files(directory, pattern, show):
    full_path = os.path.join(directory, '*' + pattern + '*.json')
    all_files = glob.glob(full_path)

    for file_name in all_files:
        with open(file_name) as f:
            print('plotting ' + f.name)  # show progress to the user
            history = json.load(f)
            plot_history(pd.DataFrame.from_dict(history), file_name, show)


p = parse_command_line()

if all(param is None for param in p):
    # No command line parameter provided - running from within IDE. Build the
    # test configuration, warn the user and run in verbose mode.
    print('\n\n  --- No command-line parameters - running with defaults\n\n')

    # # Standard network - top entry
    # # Get all history files from a directory...
    # directory = './test_results/mlp/standard/sgd'
    # # ...and a specific pattern to select files
    # pattern = 'dropout_mnist_mlp_standard_sgd_nw=standard_opt=sgd_hl=002_uhl=2048_e=50_bs=0128_dri=0.10_drh=0.50_lr=0.1000_d=0.0000_m=0.95_mn=none_history'  # noqa

    # # Droput network - top entry
    # # Get all history files from a directory...
    # directory = './test_results/mlp/dropout/sgd'
    # # ...and a specific pattern to select files
    # pattern = 'dropout_mnist_mlp_dropout_sgd_nw=dropout_opt=sgd_hl=002_uhl=2048_e=50_bs=0128_dri=0.10_drh=0.50_lr=0.0100_d=0.0010_m=0.99_mn=3_history'  # noqa

    # Batchnorm network - top entry
    # Get all history files from a directory...
    directory = './test_results/mlp/batch_normalization/sgd'
    # ...and a specific pattern to select files
    pattern = 'batchnorm_mnist_mlp_sgd_nw=batch_normalization_opt=sgd_hl=004_uhl=2048_e=50_bs=0128_lr=0.0100_d=0.0000_m=0.95_history'  # noqa

    plot_all_files(directory, pattern, show=True)
else:
    directory, pattern = parse_command_line()
    plot_all_files(directory, pattern, show=False)
