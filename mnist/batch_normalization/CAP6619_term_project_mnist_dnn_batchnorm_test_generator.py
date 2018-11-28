"""
CAP-6619 Deep Learning Fall 2018 term project
MNIST with standard deep neural network and batch normalization

Create a shell script with all tests we need to execute.
"""
import itertools
import os
import stat

# All combinations of values we need to try
# This is the complete list - uncomment for final tests
# hidden_layers = ["1", "2", "3", "4"]
# units_per_layer = ["512", "1024", "2048"]
# epochs = ["2", "5", "10"]
# # 60 is the batch size used in the paper ("with 60 examples per mini-batch")
# batch_size = ["60", "128", "256"]

# This is a simplified list
hidden_layers = ["1", "2"]
units_per_layer = ["512", "1024"]
epochs = ["2", "5"]
batch_size = ["128"]

all_tests = list(itertools.product(
    hidden_layers, units_per_layer, epochs, batch_size,))

args_template = ("--hidden_layers {} --units_per_layer {} --epochs {} "
                 "--batch_size {}")
script_file = "batchnorm_mnist_tests.sh"
with open(script_file, "w") as f:
    f.write("#!/bin/bash\n")
    f.write("# This file was automatically generated\n\n")
    for i, test in enumerate(all_tests, start=1):
        args = args_template.format(test[0], test[1], test[2], test[3])
        f.write('echo "Testing {} of {} - {}"\n'.format(i, len(all_tests),
                                                        test))
        f.write("python3 CAP6619_term_project_mnist_dnn_batchnorm.py \\\n")
        f.write("   " + args + "\n\n")

# Make it executable (for the current user)
os.chmod(script_file, os.stat(script_file).st_mode | stat.S_IEXEC)
