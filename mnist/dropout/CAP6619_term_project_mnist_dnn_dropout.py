"""
CAP-6619 Deep Learning Fall 2018 term project
MNIST with standard deep neural network and dropout

Dropout paper: http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf
"""
import time
import pandas as pd
import collections
from keras import models
from keras import layers
from keras import optimizers
from keras import backend
from keras.utils import to_categorical
from keras.constraints import max_norm
from keras.datasets import mnist


def test_model(description, model, parameters, end_experiment_callback):
    """Test one model: train it, evaluate with test data, save results."""
    # To make lines shorter
    p = parameters

    start = time.process_time()
    model.fit(train_images, train_labels, epochs=p.epochs,
              batch_size=p.batch_size)
    training_time = time.process_time() - start

    start = time.process_time()
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    test_time = time.process_time() - start

    end_experiment_callback(description, parameters, model, test_loss,
                            test_acc, training_time, test_time)


def test_network_configurations(parameters,
                                standard_optimizer, dropout_optimizer,
                                end_experiment_callback):
    """Test all network configurations with the given parameters."""
    # To make lines shorter
    p = parameters

    # Standard network (no dropout)
    model = models.Sequential()
    model.add(layers.Dense(p.units_per_layer,
                           activation='relu', input_shape=(pixels_per_image,)))
    for _ in range(p.hidden_layers - 1):
        model.add(layers.Dense(p.units_per_layer, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=standard_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    test_model("standard_network", model, p, end_experiment_callback)

    # Dropout without adjustment to number of units (for comparison - not in
    # the original paper).
    # Dropout is applied to all layers, as shown in figure 1.b in the paper.
    model = models.Sequential()
    model.add(layers.Dropout(p.dropout_rate_input_layer,
                             input_shape=(pixels_per_image,)))
    for _ in range(p.hidden_layers):
        model.add(layers.Dense(p.units_per_layer, activation='relu',
                               kernel_initializer='he_normal',
                               kernel_constraint=max_norm(int(p.max_norm_max_value))))
    model.add(layers.Dropout(rate=p.dropout_rate_hidden_layer))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=dropout_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    test_model("dropout_no_adjustment", model, p, end_experiment_callback)

    # Adjust number of units in each layer: "...if an n-sized layer is optimal
    # for a standard neural net on any given task, a good dropout net should
    # have at least n/p units." [Keras is "drop", not "keep", hence the "1 -"].
    adjusted_units_hidden = int(
        p.units_per_layer / (1 - p.dropout_rate_hidden_layer))

    # Dropout with adjustment to number of units.
    # Dropout is applied to all layers, as shown in figure 1.b in the paper.
    # See also http://www.cs.toronto.edu/~nitish/dropout/mnist.pbtxt for code
    # used in the paper for more details on some of the parameters.
    model = models.Sequential()
    model.add(layers.Dropout(p.dropout_rate_input_layer,
                             input_shape=(pixels_per_image,)))
    for _ in range(p.hidden_layers):
        model.add(layers.Dense(adjusted_units_hidden, activation='relu',
                               kernel_initializer='he_normal',
                               kernel_constraint=max_norm(int(p.max_norm_max_value))))
        model.add(layers.Dropout(rate=p.dropout_rate_hidden_layer))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=dropout_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    test_model("dropout_units_adjusted", model, p, end_experiment_callback)

    # A note on L2 regularization: although the paper says it was "found to be
    # useful for dropout neural networks as well", during the tests it didn't
    # make a major difference (better is some cases, worse in others). It was
    # added as shown below. Perahps I didn't do it correctly?
    # model.add(...kernel_regularizer=regularizers.l2(0.001),


def save_experiment(description, parameters, model, test_loss, test_acc,
                    training_time, test_time):
    """Save results from one experiment."""
    # To save some typing
    p = parameters

    # Even though we have information about the optimizer in the parameters,
    # we read directly from the model as insurance against coding mistakes.
    optimizer = model.optimizer
    optimizer_name = type(optimizer).__name__

    experiments.loc[len(experiments)] = ["MNIST", p.network,
                                         optimizer_name, test_loss,
                                         test_acc, p.hidden_layers,
                                         p.units_per_layer,
                                         p.epochs, p.batch_size,
                                         p.dropout_rate_input_layer,
                                         p.dropout_rate_hidden_layer,
                                         backend.eval(optimizer.lr),
                                         p.momentum,
                                         p.max_norm_max_value,
                                         model.count_params(),
                                         training_time, test_time]
    # Show progress so far
    print(experiments)

    # File where the results will be saved (the name encodes the parameters
    # used in the experiments)
    base_name_prefix = "MNIST_DNN_Dropout"
    base_name_template = ("{}nw={}_opt={}_hl={:03d}_uhl={:04d}_e={:02d}"
                          "_bs={:04d}_dri={:0.2f}_drh={:0.2f}_lr={:03.1f}"
                          "_m={}_mn={}")
    base_name = base_name_template.format(
        base_name_prefix, p.network, p.optimizer, p.hidden_layers, p.units_per_layer, p.epochs, p.batch_size,
        p.dropout_rate_input_layer, p.dropout_rate_hidden_layer,
        p.learning_rate, p.momentum, p.max_norm_max_value,
    )

    # Save progress so far into one file
    with open(base_name + ".txt", "w") as f:
        experiments.to_string(f)

    # Save training history and model for this specific experiment.
    # The model object must be a trained model, which means it has a `history`
    # object with the training results for each epoch.
    # We need to save the history separately because `model.save` won't save
    # it - it saves only the model data.
    results_file = base_name + "_" + description + "_" + optimizer_name + "_"
    import json
    with open(results_file + "history.json", 'w') as f:
        json.dump(model.history.history, f)
    # Uncomment to save the model - it may take quite a bit of disk space
    # model.save(results_file + "model.h5")


def parse_command_line():
    """Parse command line parameters into a `Parameters` variable."""
    from argparse import ArgumentParser
    ap = ArgumentParser(description='Dropout with MNIST data set.')

    # Format: short parameter name, long name, default value (if not specified)
    ap.add_argument("--network", type=str)
    ap.add_argument("--optimizer", type=str)
    ap.add_argument("--hidden_layers", type=int)
    ap.add_argument("--units_per_layer", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--dropout_rate_input_layer", type=float)
    ap.add_argument("--dropout_rate_hidden_layer", type=float)
    ap.add_argument("--learning_rate", type=float)
    ap.add_argument("--momentum", type=str)
    ap.add_argument("--max_norm_max_value", type=str)

    args = ap.parse_args()

    return Parameters(
        network=args.network,
        optimizer=args.optimizer,
        hidden_layers=args.hidden_layers,
        units_per_layer=args.units_per_layer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout_rate_input_layer=args.dropout_rate_input_layer,
        dropout_rate_hidden_layer=args.dropout_rate_hidden_layer,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        max_norm_max_value=args.max_norm_max_value,
    )


def run_all_experiments(parameters):
    """Run all experiments: test all network configurations, with different
    optimizers."""
    # The SGD optimizer to use in standard networks (no dropout).
    optimizer_sgd_standard = optimizers.SGD()
    # The SGD optimizer to use in dropout networks.
    optimizer_sgd_dropout = optimizers.SGD(
        lr=backend.eval(optimizer_sgd_standard.lr) * p.learning_rate,
        momentum=float(p.momentum))

    # The RMSProp optimizer to use in standard networks (no dropout).
    # The paper doesn't mention what optimizer was used in the tests. It looks
    # like those tests were done with SGD. I tried RMSProp here because it's a
    # popular one nowadays and the one used in the Deep Learning With Python
    # book. It results in good accuracy with the default learning rate, even
    # before dropout is applied.
    optimizer_rmsprop_standard = optimizers.RMSprop()
    # The RMSProp optimizer to use in dropout networks.
    # Increasing the learn rate for the RMSProp optimizer resulted in much
    # worse accuracy. To prevent that we use the default optimizer for dropout.
    optimizer_rmsprop_dropout = optimizer_rmsprop_standard

    # Run the experiments with the SGD optimzer
    test_network_configurations(parameters,
                                standard_optimizer=optimizer_sgd_standard,
                                dropout_optimizer=optimizer_sgd_dropout,
                                end_experiment_callback=save_experiment)

    # Run the experiments with the RMSProp optimizer
    test_network_configurations(parameters,
                                standard_optimizer=optimizer_rmsprop_standard,
                                dropout_optimizer=optimizer_rmsprop_dropout,
                                end_experiment_callback=save_experiment)


# Store data from the experiments
experiments = pd.DataFrame(columns=["DataSetName", "Network", "Optimizer",
                                    "TestLoss", "TestAccuracy",
                                    "HiddenLayers", "UnitsPerLayer", "Epochs",
                                    "BatchSize", "DropoutRateInput",
                                    "DropoutRateHidden", "LearningRate",
                                    "Momentum", "MaxNorm",
                                    "ModelParamCount", "TrainingCpuTime",
                                    "TestCpuTime"])

# Parameters to control the experiments.
Parameters = collections.namedtuple("Parameters", [
    # Type of network to test: "standard": no dropout, "dropout_no_adjustment":
    # dropout without adjusting units in each layer, "dropout": dropout with
    # layers adjusted as recommended in the paper.
    "network",
    # Type of optimizer to use: "sgd" or "rmsprop". The paper doesn't specify,
    # but it can be inferred that it's an SGD with adjusted learning rate.
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
    # Momentum, to adjust as recommended in the dropout paper ("While momentum
    # values of 0.9 are common for standard nets, with dropout we found that
    # values around 0.95 to 0.99 work quite a lot better.")
    "momentum",
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

p = parse_command_line()
run_all_experiments(p)
