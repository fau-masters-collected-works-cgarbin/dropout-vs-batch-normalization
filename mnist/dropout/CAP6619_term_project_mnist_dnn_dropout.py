"""
CAP-6619 Deep Learning Fall 2018 term project
MNIST with standard deep neural network and dropout

Dropout paper: http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf
"""
import time
import pandas as pd
import collections
import json
import os
from keras import models
from keras import layers
from keras import optimizers
from keras import backend
from keras.utils import to_categorical
from keras.constraints import max_norm
from keras.datasets import mnist
from datetime import datetime
from io import StringIO
from argparse import ArgumentParser


def create_model(parameters):
    """Create a model described by the given parameters."""
    # To make lines shorter
    p = parameters

    model = models.Sequential()

    if p.network == "standard":
        model.add(layers.Dense(p.units_per_layer,
                               activation='relu',
                               input_shape=(pixels_per_image,)))
        for _ in range(p.hidden_layers - 1):
            model.add(layers.Dense(p.units_per_layer, activation='relu'))
    elif p.network in ("dropout_no_adjustment", "dropout"):
        units_hidden_layer = 0
        if p.network == "dropout":
            # Adjust number of units in each layer: "...if an n-sized layer is
            # optimal for a standard neural net on any given task, a good
            # dropout net should have at least n/p units." [Note that Keras
            # uses a "drop" rate, not "keep", hence the "1 -"].
            units_hidden_layer = int(
                p.units_per_layer / (1 - p.dropout_rate_hidden_layer))
        else:
            units_hidden_layer = p.units_per_layer

        model.add(layers.Dropout(p.dropout_rate_input_layer,
                                 input_shape=(pixels_per_image,)))
        for _ in range(p.hidden_layers):
            # Reason to use he_normal initializer: source code the paper points
            # to has "initialization: DENSE_GAUSSIAN_SQRT_FAN_IN" for weights.
            if p.max_norm_max_value == "none":
                model.add(layers.Dense(units_hidden_layer, activation='relu',
                                       kernel_initializer='he_normal'))
            else:
                model.add(layers.Dense(units_hidden_layer, activation='relu',
                                       kernel_initializer='he_normal',
                                       kernel_constraint=max_norm(
                                           int(p.max_norm_max_value))))
            model.add(layers.Dropout(rate=p.dropout_rate_hidden_layer))
    else:
        assert False  # Invalid network type

    # All networks end with the sofmax layer to identify the 0-9 digits
    model.add(layers.Dense(10, activation='softmax'))

    # Create the optimizer
    optimizer = None
    if p.optimizer == "sgd":
        optimizer = optimizers.SGD(
            p.learning_rate, momentum=float(p.sgd_momentum), decay=p.decay)
    elif p.optimizer == "rmsprop":
        optimizer = optimizers.RMSprop(p.learning_rate, decay=p.decay)
    else:
        assert False  # Invalid optimizer

    # Compile the network and test it
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def test_model(parameters, end_experiment_callback):
    """Test one model: create, train, evaluate with test data and save
    results."""
    # To make lines shorter
    p = parameters

    model = create_model(parameters)

    start = time.process_time()
    model.fit(train_images, train_labels, epochs=p.epochs,
              batch_size=p.batch_size)
    training_time = time.process_time() - start

    start = time.process_time()
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    test_time = time.process_time() - start

    end_experiment_callback(parameters, model, test_loss,
                            test_acc, training_time, test_time)


def save_experiment(parameters, model, test_loss, test_acc,
                    training_time, test_time):
    """Save results from one experiment."""
    # To save some typing
    p = parameters

    # Even though we have information about the optimizer in the parameters,
    # we read directly from the model as insurance against coding mistakes.
    optimizer = model.optimizer
    optimizer_name = type(optimizer).__name__

    experiments.loc[len(experiments)] = [
        p.experiment_name, datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        "MNIST", p.network, optimizer_name, test_loss, test_acc,
        p.hidden_layers, p.units_per_layer, p.epochs, p.batch_size,
        p.dropout_rate_input_layer, p.dropout_rate_hidden_layer,
        backend.eval(optimizer.lr), p.decay, p.sgd_momentum,
        p.max_norm_max_value, model.count_params(), training_time, test_time]

    # Show progress so far to the user
    print(experiments)

    # Save progress so far into the file used for this experiment
    results_file = p.experiment_name + "_results.txt"
    if os.path.isfile(results_file):
        # File already exists - append data without column names.
        # First, get a formatted string; if we use to_string(header=False) it
        # will use only one space between columnds, instead of formatting
        # considering the column name (the header).
        output = StringIO()
        experiments.to_string(output)
        with open(results_file, "a") as f:
            f.write(os.linesep)
            f.write(output.getvalue().splitlines()[1])
        output.close()
    else:
        # File doesn't exist yet - create and write column names + data
        with open(results_file, "w") as f:
            experiments.to_string(f)

    # Save training history and model for this specific experiment.
    # The model object must be a trained model, which means it has a `history`
    # object with the training results for each epoch.
    # We need to save the history separately because `model.save` won't save
    # it - it saves only the model data.

    # File where the training history and model will be saved. The name encodes
    # the test the parameters used in the epxeriment.
    base_name_template = ("{}_nw={}_opt={}_hl={:03d}_uhl={:04d}_e={:02d}"
                          "_bs={:04d}_dri={:0.2f}_drh={:0.2f}_lr={:03.1f}"
                          "d={:0.4f}_m={}_mn={}")
    base_name = base_name_template.format(
        p.experiment_name, p.network, p.optimizer, p.hidden_layers,
        p.units_per_layer, p.epochs, p.batch_size,
        p.dropout_rate_input_layer, p.dropout_rate_hidden_layer,
        p.learning_rate, p.decay, p.sgd_momentum, p.max_norm_max_value,
    )

    with open(base_name + "_history.json", 'w') as f:
        json.dump(model.history.history, f)
    # Uncomment to save the model - it may take quite a bit of disk space
    # model.save(base_name + "_model.h5")


def parse_command_line():
    """Parse command line parameters into a `Parameters` variable."""
    ap = ArgumentParser(description='Dropout with MNIST data set.')

    # Format: short parameter name, long name, default value (if not specified)
    ap.add_argument("--experiment_name", type=str)
    ap.add_argument("--network", type=str)
    ap.add_argument("--optimizer", type=str)
    ap.add_argument("--hidden_layers", type=int)
    ap.add_argument("--units_per_layer", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--dropout_rate_input_layer", type=float)
    ap.add_argument("--dropout_rate_hidden_layer", type=float)
    ap.add_argument("--learning_rate", type=float)
    ap.add_argument("--decay", type=float)
    ap.add_argument("--sgd_momentum", type=str)
    ap.add_argument("--max_norm_max_value", type=str)

    args = ap.parse_args()

    return Parameters(
        experiment_name=args.experiment_name,
        network=args.network,
        optimizer=args.optimizer,
        hidden_layers=args.hidden_layers,
        units_per_layer=args.units_per_layer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout_rate_input_layer=args.dropout_rate_input_layer,
        dropout_rate_hidden_layer=args.dropout_rate_hidden_layer,
        learning_rate=args.learning_rate,
        decay=args.decay,
        sgd_momentum=args.sgd_momentum,
        max_norm_max_value=args.max_norm_max_value,
    )


# Store data from the experiments
experiments = pd.DataFrame(columns=[
    "ExperimentName", "TestTime", "DataSetName", "Network", "Optimizer",
    "TestLoss", "TestAccuracy", "HiddenLayers", "UnitsPerLayer", "Epochs",
    "BatchSize", "DropoutRateInput", "DropoutRateHidden", "LearningRate",
    "Decay", "SgdMomentum", "MaxNorm", "ModelParamCount", "TrainingCpuTime",
    "TestCpuTime"])

# Parameters to control the experiments.
Parameters = collections.namedtuple("Parameters", [
    # A brief description of the experiment. Will be used as part of file names
    # to prevent collisions with other experiments. Cannot contain spaces to
    # work correctly as a command line parameter.
    "experiment_name",
    # Type of network to test: "standard": no dropout, "dropout_no_adjustment":
    # dropout without adjusting units in each layer, "dropout": dropout with
    # layers adjusted as recommended in the paper.
    "network",
    # Type of optimizer to use: "sgd" or "rmsprop". The paper doesn't specify,
    # but it can be inferred that it's an SGD with adjusted learning rate.
    # I also tried RMSProp here because it's a popular one nowadays and the one
    # used in the Deep Learning With Python book. It results in good accuracy
    # with the default learning rate, even before dropout is applied.
    "optimizer",
    # Number of hidden layers in the network. When a dropout network is used,
    # each hidden layer will be followed by a dropout layer.
    "hidden_layers",
    # Number of units in each layer (note that dropout layers are adjusted,
    # increasing the number of units used in the network).
    "units_per_layer",
    # Number of epochs to train.
    "epochs",
    # Number of samples in each batch.
    "batch_size",
    # Dropout rate for the input layer ("For input layers, the choice depends
    # on the kind of input. For real-valued inputs (image patches or speech
    # frames), a typical value is 0.8.)" [Note: keras uses "drop", not "keep"]
    "dropout_rate_input_layer",
    # Dropout rate for the input layer ("Typical values of p for hidden units
    # are in the range 0.5 to 0.8.)" [Note: keras uses "drop", not "keep" rate]
    "dropout_rate_hidden_layer",
    # Learning rate, to adjust as recommended in the dropout paper ("...
    # dropout net should typically use 10-100 times the learning rate that was
    # optimal for a standard neural net.")
    "learning_rate",
    # Weight decay (L2). The source code the paper points to has "l2_decay"
    # set to 0.001. The default in Keras for SGD and RMSProp is 0.0.
    "decay",
    # Momentum for the SGD optimizer (not used in RMSProp), to adjust as
    # recommended in the dropout paper ("While momentum values of 0.9 are
    # common for standard nets, with dropout we found that values around 0.95
    # to 0.99 work quite a lot better."). Set to 0.0 to not use momentum.
    "sgd_momentum",
    # Max norm max value, or "none" to skip it. The paper recommends its usage
    # ("Although dropout alone gives significant improvements, using dropout
    # along with max-norm... Typical values of c range from 3 to 4.")
    "max_norm_max_value",
])

# The input shape: pixels_per_image pixels images from MNIST data set
pixels_per_image = 28 * 28

# Load and prepare data.
# Note that they are global variables used in the functions above. A future
# improvement could be to add them to the parameters data structure.
start = time.process_time()
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("Timing: load and prepare data: {0:.5f}s".format(
    time.process_time() - start))

train_images = train_images.reshape((60000, pixels_per_image))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, pixels_per_image))
test_images = test_images.astype('float32') / 255

# Change this to "False" when testing from the command line. Leave set to True
# when launching from the IDE and change the parameters below (it's faster
# than dealing with launch.json).
ide_test = True
# Show a warning to let user now we are ignoring command line parameters
if ide_test:
    print("\n\n  --- Running from IDE - ignoring command line\n\n")

p = None
if ide_test:
    p = Parameters(
        experiment_name="dropout_mnist_dnn",
        network="dropout_no_adjustment",
        optimizer="sgd",
        hidden_layers=2,
        units_per_layer=512,
        epochs=2,
        batch_size=128,
        dropout_rate_input_layer=0.1,
        dropout_rate_hidden_layer=0.5,
        learning_rate=0.1,
        decay=0.001,
        sgd_momentum=0.95,
        max_norm_max_value=2,
    )
else:
    p = parse_command_line()

test_model(p, save_experiment)
