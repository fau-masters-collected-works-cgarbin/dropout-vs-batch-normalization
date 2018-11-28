"""
CAP-6619 Deep Learning Fall 2018 term project
MNIST with standard deep neural network and batch normalization

Batch normalization paper: https://arxiv.org/pdf/1502.03167.pdf
"""
import time
import pandas as pd
import collections
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical
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
                                optimizer, end_experiment_callback):
    """Test all network configurations with the given parameters."""
    # To make lines shorter
    p = parameters

    # Standard network (no batch normalization)
    model = models.Sequential()
    model.add(layers.Dense(p.units_per_layer,
                           activation='relu', input_shape=(28 * 28,)))
    for _ in range(p.hidden_layers - 1):
        model.add(layers.Dense(p.units_per_layer, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    test_model("standard_network", model, p, end_experiment_callback)

    # Batch normalization
    # "We added Batch Normalization to each hidden layer of the network,..."
    # Note on the ativation function: the paper states "Each hidden layer...
    # with sigmoid nonlinearity...", but tests with ReLU resulted in
    # significantly better accuracy for SGD and slightly better for RMSProp,
    # so all tests will be executed with ReLU.
    model = models.Sequential()
    model.add(layers.Dense(p.units_per_layer,
                           kernel_initializer='he_normal',
                           activation='relu', input_shape=(28 * 28,)))
    # Note on scale, from Keras doc: "When the next layer is linear (also e.g.
    # nn.relu), this can be disabled since the scaling will be done by the next
    # layer.", i.e. scale=True only in the layer before the softmax layer.
    scale = p.hidden_layers == 1  # Scale only if using only one layer
    model.add(layers.BatchNormalization(scale=scale))
    for i in range(p.hidden_layers - 1):
        model.add(layers.Dense(p.units_per_layer,
                               kernel_initializer='he_normal',
                               activation='relu'))
        scale = i == p.hidden_layers - 2  # Scale only last layer
        model.add(layers.BatchNormalization(scale=scale))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    test_model("batch_normalization", model, p, end_experiment_callback)


def save_experiment(description, parameters, model, test_loss, test_acc,
                    training_time, test_time):
    """Save results from one experiment"""
    # To save some typing
    p = parameters

    optimizer = model.optimizer
    optimizer_name = type(optimizer).__name__

    experiments.loc[len(experiments)] = [description, "MNIST",
                                         optimizer_name, test_loss,
                                         test_acc, p.hidden_layers,
                                         p.units_per_layer,
                                         p.epochs, p.batch_size,
                                         model.count_params(),
                                         training_time, test_time]
    # Show progress so far
    print(experiments)

    # File where the results will be saved (the name encodes the parameters
    # used in the experiments)
    base_name_prefix = "MNIST_DNN_BatchNorm"
    base_name_template = "{}_hl={:03d}_uhl={:04d}_e={:02d}_bs={:04d}"
    base_name = base_name_template.format(
        base_name_prefix, p.hidden_layers,
        p.units_per_layer, p.epochs, p.batch_size)

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
    ap.add_argument("--hidden_layers", default=2, type=int)
    ap.add_argument("--units_per_layer", default=512, type=int)
    ap.add_argument("--epochs", default=5, type=int)
    ap.add_argument("--batch_size", default=128, type=int)

    args = ap.parse_args()

    return Parameters(
        hidden_layers=args.hidden_layers,
        units_per_layer=args.units_per_layer,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


def run_all_experiments(parameters):
    """Run all experiments: test all network configurations, with different
    optimizers."""
    optimizer_sgd_standard = optimizers.SGD()
    optimizer_rmsprop_standard = optimizers.RMSprop()

    test_network_configurations(parameters,
                                optimizer=optimizer_sgd_standard,
                                end_experiment_callback=save_experiment)

    test_network_configurations(parameters,
                                optimizer=optimizer_rmsprop_standard,
                                end_experiment_callback=save_experiment)


# Store data from the experiments
experiments = pd.DataFrame(columns=["Description", "DataSetName", "Optimizer",
                                    "TestLoss", "TestAccuracy",
                                    "HiddenLayers", "UnitsPerLayer", "Epochs",
                                    "BatchSize", "ModelParamCount",
                                    "TrainingCpuTime", "TestCpuTime"])

# Parameters to control the experiments.
Parameters = collections.namedtuple("Parameters", [
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
])

# Load and prepare data
# Note that they are global variables used in the functions above. A future
# improvement could be to add them to the parameters data structure.
start = time.process_time()
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("Timing: load and prepare data: {0:.5f}s".format(
    time.process_time() - start))

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

p = parse_command_line()
run_all_experiments(p)
