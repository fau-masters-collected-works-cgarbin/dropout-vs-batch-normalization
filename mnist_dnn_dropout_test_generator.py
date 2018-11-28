"""
CAP-6619 Deep Learning Fall 2018 term project
MNIST with standard deep neural network and dropout

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
# batch_size = ["128", "256"]
# dropout_rate_input_layer = ["0.1", "0.2"]
# dropout_rate_hidden_layer = ["0.5"]
# dropout_lr_multiplier = ["10", "100"]
# dropout_momentum = ["0.95", "0.99"]
# max_norm_max_value = ["2", "3", "4"]

# This is a simplified list
hidden_layers = ["1", "2"]
units_per_layer = ["512", "1024"]
epochs = ["2", "5"]
batch_size = ["128"]
dropout_rate_input_layer = ["0.1"]
dropout_rate_hidden_layer = ["0.5"]
dropout_lr_multiplier = ["10"]
dropout_momentum = ["0.95"]
max_norm_max_value = ["3"]


all_tests = list(itertools.product(
    hidden_layers, units_per_layer, epochs, batch_size,
    dropout_rate_input_layer, dropout_rate_hidden_layer, dropout_lr_multiplier,
    dropout_momentum, max_norm_max_value))

args_template = ("--hidden_layers {} --units_per_layer {} --epochs {} "
                 "--batch_size {} --dropout_rate_input_layer {} "
                 "--dropout_rate_hidden_layer {} --dropout_lr_multiplier {} "
                 "--dropout_momentum {} --max_norm_max_value {}")
script_file = "dropout_mnist.sh"
with open(script_file, "w") as f:
    f.write("#!/bin/bash\n")
    f.write("# This file was automatically generated\n\n")
    for i, test in enumerate(all_tests, start=1):
        args = args_template.format(test[0], test[1], test[2], test[3],
                                    test[4], test[5], test[6], test[7],
                                    test[8])
        f.write('echo "Testing {} of {} - {}"\n'.format(i, len(all_tests),
                                                        test))
        f.write("python3 CAP6619_term_project_mnist_dnn_dropout.py \\\n")
        f.write("   " + args + "\n\n")

# Make it executable (for the current user)
os.chmod(script_file, os.stat(script_file).st_mode | stat.S_IEXEC)
