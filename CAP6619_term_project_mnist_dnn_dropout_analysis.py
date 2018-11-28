"""
CAP-6619 Deep Learning Fall 2018 term project
MNIST with standard deep neural network and dropout

Read test result files and generate graphs for analysis.
"""
import pandas as pd
import glob

# Get all result files from current directory
all_files = glob.glob("MNIST_DNN_Dropout_*.txt")
# Create a generator to get data from one file at a time
file_generator = (pd.read_csv(f, delim_whitespace=True) for f in all_files)
# Read all files using the generator and concatenates them (ignore_index=True
# creates a unique index across all files)
results = pd.concat(file_generator, ignore_index=True)
# Uncomment to save combined results into one file
# with open("results.txt", "w") as f:
#     results.to_string(f)
