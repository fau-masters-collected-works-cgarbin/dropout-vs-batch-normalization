"""
CAP-6619 Deep Learning Fall 2018 term project
MNIST with standard deep neural network and dropout

Create a shell script with all tests we need to execute.
"""
import itertools
import os
import stat

# Some default values from Keras to keep in mind:
#  * Learning rate: SGD=0.01, RMSprop=0.001
#  * Momenum: SGD=0.0 (RMSprop doesn't have momentum)
#  * Decay: 0.0 for SGD and RMSprop
#  * MaxNorm:  not used in either, must be explicitly added

# All combinations of values we need to try
# This is the complete list - uncomment for final tests
# network = ["standard", "dropout_no_adjustment", "dropout"]
# optimizer = ["sgd", "rmsprop"]
# hidden_layers = ["1", "2", "3", "4"]
# units_per_layer = ["512", "1024", "2048"]
# epochs = ["2", "5", "10"]
# batch_size = ["128", "256"]
# dropout_rate_input_layer = ["0.1", "0.2"]
# dropout_rate_hidden_layer = ["0.5"]
# learning_rate = ["0.001", "0.01", "0.1"]
# decay = ["0.001", "0.01"]
# sgd_momentum = ["0.95", "0.99"]
# max_norm_max_value = ["none", "2", "3", "4"]

# This is a simplified list
network = ["standard", "dropout_no_adjustment", "dropout"]
optimizer = ["sgd", "rmsprop"]
hidden_layers = ["1"]
units_per_layer = ["512"]
epochs = ["2"]
batch_size = ["128"]
dropout_rate_input_layer = ["0.1"]
dropout_rate_hidden_layer = ["0.5"]
learning_rate = ["0.01"]
decay = ["0.001"]
sgd_momentum = ["0.95"]
max_norm_max_value = ["2"]


all_tests = list(itertools.product(
    network, optimizer, hidden_layers, units_per_layer, epochs, batch_size,
    dropout_rate_input_layer, dropout_rate_hidden_layer, learning_rate,
    decay, sgd_momentum, max_norm_max_value))

args_template = (
    "--network {} --optimizer {} --hidden_layers {} --units_per_layer {} "
    "--epochs {} --batch_size {} --dropout_rate_input_layer {} "
    "--dropout_rate_hidden_layer {} --learning_rate {} --decay {}"
    " --sgd_momentum {} --max_norm_max_value {}")
script_file = "mnist_tests.sh"
with open(script_file, "w") as f:
    f.write("#!/bin/bash\n")
    f.write("# This file was automatically generated\n\n")
    for i, test in enumerate(all_tests, start=1):
        args = args_template.format(
            test[0], test[1], test[2], test[3], test[4], test[5], test[6],
            test[7], test[8], test[9], test[10], test[11])
        f.write('echo "Testing {} of {} - {}"\n'.format(i, len(all_tests),
                                                        test))
        f.write("python3 CAP6619_term_project_mnist_dnn_dropout.py \\\n")
        f.write("   " + args + "\n\n")

# Make it executable (for the current user)
os.chmod(script_file, os.stat(script_file).st_mode | stat.S_IEXEC)
