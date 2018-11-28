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
from keras import backend
from keras.utils import to_categorical
from keras.datasets import mnist

# Store data from the experiments
experiments = pd.DataFrame(columns=["Description", "DataSetName", "Optimizer",
                                    "TestLoss", "TestAccuracy",
                                    "HiddenLayers", "UnitsPerLayer", "Epochs",
                                    "BatchSize", "LearningRate",
                                    "ModelParamCount", "TrainingCpuTime",
                                    "TestCpuTime"])


def run_experiment(description, model, parameters, end_experiment_callback):
    """Run an experiment: train and test the network"""
    # To make lines shorter
    p = parameters

    start = time.process_time()
    model.fit(train_images, train_labels, epochs=p.epochs,
              batch_size=p.batch_size)
    training_time = time.process_time() - start

    start = time.process_time()
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    test_time = time.process_time() - start

    end_experiment_callback(description, model, test_loss, test_acc,
                            training_time, test_time)


def test_network_configurations(parameters,
                                standard_optimizer,
                                end_experiment_callback):
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
    model.compile(optimizer=standard_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    run_experiment("standard_network", model, p, end_experiment_callback)

    # Batch normalization
    # "We added Batch Normalization to each hidden layer of the network,..."
    # TODO: use sigmoid
    # "Each hidden layer computes y = g(Wu+b) with sigmoid nonlinearity..."
    model = models.Sequential()
    model.add(layers.Dense(p.units_per_layer,
                           activation='relu', input_shape=(28 * 28,)))
    for _ in range(p.hidden_layers - 1):
        model.add(layers.Dense(p.units_per_layer, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=standard_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    run_experiment("batch_normalization", model, p, end_experiment_callback)


# Load and prepare data
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

p = Parameters(
    hidden_layers=3,
    units_per_layer=100,
    epochs=5,
    batch_size=60,
)

optimizer_sgd_standard = optimizers.SGD()
optimizer_rmsprop_standard = optimizers.RMSprop()

# File where the results will be saved (the name encodes the parameters used
# in the experiments)
file_name_prefix = "MNIST_DNN_BatchNorm"
file_name_template = ("{}_hl={:03d}_uhl={:04d}_bs={:04d}_")
file_name = file_name_template.format(
    file_name_prefix, p.hidden_layers, p.units_per_layer, p.batch_size)


def save_experiment(description, model, test_loss, test_acc, training_time,
                    test_time):
    """Save results from one experiment"""
    optimizer = model.optimizer
    optimizer_name = type(optimizer).__name__

    experiments.loc[len(experiments)] = [description, "MNIST",
                                         optimizer_name, test_loss,
                                         test_acc, p.hidden_layers,
                                         p.units_per_layer,
                                         p.epochs, p.batch_size,
                                         backend.eval(optimizer.lr),
                                         model.count_params(),
                                         training_time, test_time]

    # Summary of experiments - all in one file
    print(experiments)
    with open(file_name + ".txt", "w") as f:
        experiments.to_string(f)

    # Save training history and model for this specific experiment
    # The model object must be a trained model, which means it has a `history`
    # object with the training results for each epoch
    # We need to save the history separately because `model.save` won't save
    # it - it saves only the model data
    experiment_file = file_name + description + "_" + optimizer_name + "_"
    import json
    with open(experiment_file + "history.json", 'w') as f:
        json.dump(model.history.history, f)
    model.save(experiment_file + "model.h5")


test_network_configurations(p, standard_optimizer=optimizer_sgd_standard,
                            end_experiment_callback=save_experiment)

test_network_configurations(p, standard_optimizer=optimizer_rmsprop_standard,
                            end_experiment_callback=save_experiment)
