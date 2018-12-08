"""
Plot accuracy and loss data generated by Keras models fit() function.

Data is read from a JSON file, crated with json.dump(history)
"""
import json
import os
import glob
import re
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import seaborn as sns


def parse_command_line():
    """Parse command line parameters and return them."""
    ap = ArgumentParser(description='Plot Keras history from a JSON file.')
    ap.add_argument("--directory", type=str)
    ap.add_argument("--pattern", type=str)
    args = ap.parse_args()

    return args.directory, args.pattern


def get_title(file_name):
    """Create a title by extracting pieces of the file name."""

    if "mlp" in file_name:
        network = re.search(r"nw=(.*?)_", file_name).group(1)
        optimizer = re.search(r"opt=(.*?)_", file_name).group(1)
        hidden_layers = re.search(r"hl=(.*?)_", file_name).group(1)
        units_per_layer = re.search(r"uhl=(.*?)_", file_name).group(1)
        epochs = re.search(r"e=(.*?)_", file_name).group(1)
        learning_rate = re.search(r"lr=(.*?)_", file_name).group(1)
        weight_decay = re.search(r"_d=(.*?)_", file_name).group(1)

        # Nicer text for humans for title and optimizer
        pretty_network = {"standard": "Standard",
                          "dropout": "Dropout",
                          "dropout_no_adjustment": "Droput w/o adjustment",
                          "batch_normalization": "Batch normalization"}
        pretty_optimizer = {"sgd": "SGD", "rmsprop": "RMSProp"}

        # Note: three lines is the most I was able to fit with the standard
        # matlibplot title formatting (there are solutions to fit more lines,
        # but none are simple - at least the ones I could find)
        title = ("{} network, {} optimizer, trained for {} epochs\n"
                 "{} hidden layers, {} units per layer \n"
                 "Learning rate = {}, weight decay = {}").format(
            pretty_network[network], pretty_optimizer[optimizer], int(epochs),
            int(hidden_layers), int(units_per_layer), float(learning_rate),
            float(weight_decay))

        return title

    if "cnn" in file_name:
        network = re.search(r"cifar_10_cnn_(.*?)_lr", file_name).group(1)
        learning_rate = re.search(r"lr=(.*?)_", file_name).group(1)
        units_dense_layer = re.search(r"udl=(.*?)_", file_name).group(1)
        epochs = re.search(r"e=(.*?)_", file_name).group(1)

        # Nicer text for humans for title and optimizer
        pretty_network = {"plain": "Plain",
                          "dropout": "Dropout",
                          "batch_normalization": "Batch Normalization",
                          "batchnorm_dropout": "Droput + Batch normalization"}

        # Note: three lines is the most I was able to fit with the standard
        # matlibplot title formatting (there are solutions to fit more lines,
        # but none are simple - at least the ones I could find)
        title = ("{} network, trained for {} epochs\n"
                 "{} units in dense layer, learning rate = {}").format(
            pretty_network[network], int(epochs),
            int(units_dense_layer), float(learning_rate))

        return title

    assert False  # Can't parse this file name


def plot_history(history, file, show):
    """Plot the loss history created from druing the execution of Keras fit().

    Arguments:
      history {[dataframe]} -- The history data from the call to fit()
      file {[string]} -- Name of the input file (the one with the history)
      show {[Boolean]} -- True to also show on screen, False to just save it
    """

    # Style with default seaborn, then change background (easier to read)
    sns.set()
    sns.set_style('white')

    # Fix the y axis scale for all graphs so we can compare graphs
    # Commented out for now - not sure it will be needed. If needed, need to
    # adapt based on the test
    # plt.ylim(0, 0.8)

    # Create a data source for the epochs - need this for the x axis
    num_epochs = len(history['loss'])
    epochs = range(1, num_epochs+1)

    # Plot loss data
    ax = sns.lineplot(x=epochs, y=history['loss'], label='Training loss')
    sns.lineplot(x=epochs, y=history['val_loss'], label='Test loss')

    # Add axis labels
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # Change x-axis tick labels (epoch) from float to integers
    plt.xticks(epochs)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.set_xlim(0, num_epochs)
    plt.grid(True, axis="x", linestyle="dotted")

    plt.legend(frameon=False)
    plt.title(get_title(file))

    # Save to disk as a .png file
    png_file = file.replace(".json", ".png")
    plt.savefig(png_file)

    if show:
        plt.show()


def plot_all_files(directory, pattern, show):
    full_path = os.path.join(directory, "*" + pattern + "*.json")
    all_files = glob.glob(full_path)

    for file in all_files:
        with open(file) as f:
            print("plotting " + f.name)  # show progress to the user
            history = json.load(f)
            plot_history(history, file, show)


# Change this to "False" when testing from the command line. Leave set to True
# when launching from the IDE and change the parameters below (it's faster
# than dealing with launch.json).
ide_test = True
if ide_test:
    # Show a warning to let user now we are ignoring command line parameters
    print("\n\n  --- Running from IDE - ignoring command line\n\n")
    # Get all history files from a directory...
    directory = "./cifar-10/analysis/quick-test"
    # ...and a specific pattern to select files
    pattern = "batch_normalization"
    plot_all_files(directory, pattern, show=True)
else:
    directory, pattern = parse_command_line()
    plot_all_files(directory, pattern, show=False)
