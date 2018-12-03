"""
CAP-6619 Deep Learning Fall 2018 term project
MNIST with standard deep neural network and dropout

Create shell scripts with all tests we need to execute.
"""
import itertools
import os
import stat
from CAP6619_term_project_mnist_dropout_parameters import Parameters

# Some default values from Keras to keep in mind:
#  * Learning rate: SGD=0.01, RMSprop=0.001
#  * Momentum: SGD=0.0 (RMSprop doesn't have momentum)
#  * Decay: 0.0 for SGD and RMSprop
#  * MaxNorm:  not used in either, must be explicitly added

# This is a quick set of tests to test the overall sanity of the code.
quick_test = Parameters(
    experiment_name="dropout_mnist_dnn_quick_test",
    network=["standard", "dropout_no_adjustment", "dropout"],
    optimizer=["sgd", "rmsprop"],
    hidden_layers=["1"],
    units_per_layer=["512"],
    epochs=["2"],
    batch_size=["128"],
    dropout_rate_input_layer=["0.1"],
    dropout_rate_hidden_layer=["0.5"],
    learning_rate=["0.01"],
    decay=["0.001"],
    sgd_momentum=["0.95"],
    max_norm_max_value=["2"],
)


def create_test_file(p):

    tests = list(itertools.product(
        p.network, p.optimizer, p.hidden_layers, p.units_per_layer, p.epochs,
        p.batch_size, p.dropout_rate_input_layer, p.dropout_rate_hidden_layer,
        p.learning_rate, p.decay, p.sgd_momentum, p.max_norm_max_value))

    args_template = (
        "--experiment_name {} --network {} --optimizer {} --hidden_layers {} "
        "--units_per_layer {} --epochs {} --batch_size {} "
        "--dropout_rate_input_layer {} --dropout_rate_hidden_layer {} "
        "--learning_rate {} --decay {} --sgd_momentum {} "
        "--max_norm_max_value {}")

    file_name = p.experiment_name + ".sh"

    with open(file_name, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# This file was automatically generated\n\n")
        for i, test in enumerate(tests, start=1):
            args = args_template.format(
                p.experiment_name,
                test[0], test[1], test[2], test[3], test[4], test[5], test[6],
                test[7], test[8], test[9], test[10], test[11])
            f.write('echo "{} - test {} of {} - {}"\n'.format(
                p.experiment_name, i, len(tests), test))
            f.write("python3 CAP6619_term_project_mnist_dnn_dropout.py \\\n")
            f.write("   " + args + "\n\n")

    # Make it executable (for the current user)
    os.chmod(file_name, os.stat(file_name).st_mode | stat.S_IEXEC)


create_test_file(quick_test)
