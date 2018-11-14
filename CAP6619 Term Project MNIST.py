"""
CAP-6619 Deep Learning Fall 2018 term project
MNIST with standard model, dropout and batch normalization
"""

import time
import pandas as pd
from keras import models
from keras import layers
from keras import optimizers
from keras import backend
from keras.utils import to_categorical
from keras.constraints import max_norm
from keras.datasets import mnist

# Store data from the experiments
experiments = pd.DataFrame(columns=["Description", "DataSetName", "TestLoss",
                                    "TestAccuracy", "NumberOfNodes", "Epochs",
                                    "BatchSize", "Optimizer", "LearningRate",
                                    "ModelParamCount", "TrainingCpuTime",
                                    "TestCpuTime"])


def save_experiments_results(file_name, display):
    if display:
        print(experiments)
    with open(file_name, "w") as outfile:
        experiments.to_string(outfile)


def run_experiment(description, model, number_of_nodes, epochs):
    """Run an experiment: train and test the network, save results"""
    print(description)

    start = time.process_time()
    batch_size = 128
    model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)
    training_time = time.process_time() - start

    start = time.process_time()
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    test_time = time.process_time() - start

    optimizer = model.optimizer

    experiments.loc[len(experiments)] = [description, "MNIST", test_loss,
                                         test_acc, number_of_nodes, epochs,
                                         batch_size, type(optimizer).__name__,
                                         backend.eval(optimizer.lr),
                                         model.count_params(),
                                         training_time, test_time]


def test_network_configurations(number_of_nodes, epochs, standard_optimizer,
                                dropout_optimizer, end_experiment_callback):
    """Test all network configurations with the given parameters."""
    # Standard network
    model = models.Sequential()
    model.add(layers.Dense(number_of_nodes,
                           activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(number_of_nodes, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=standard_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    run_experiment("Standard network",
                   model, number_of_nodes, epochs)
    end_experiment_callback()

    # Dropout network, no adjustment
    dropout_rate = 0.5
    model = models.Sequential()
    model.add(layers.Dropout(0.2, input_shape=(28 * 28,)))
    model.add(layers.Dense(number_of_nodes, activation='relu',
                           kernel_constraint=max_norm(3)))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(number_of_nodes, activation='relu',
                           kernel_constraint=max_norm(3)))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=dropout_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    run_experiment("Dropout network not adjusted",
                   model, number_of_nodes, epochs)
    end_experiment_callback()

    # Dropout network adjusted before
    model = models.Sequential()
    model.add(layers.Dropout(0.2, input_shape=(28 * 28,)))
    model.add(layers.Dense(int(number_of_nodes / dropout_rate),
                           activation='relu', kernel_constraint=max_norm(3)))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(number_of_nodes, activation='relu',
                           kernel_constraint=max_norm(3)))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=dropout_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    run_experiment("Dropout network adjusted before",
                   model, number_of_nodes, epochs)
    end_experiment_callback()

    # Dropout network, adjusted after
    model = models.Sequential()
    model.add(layers.Dropout(0.2, input_shape=(28 * 28,)))
    model.add(layers.Dense(number_of_nodes, activation='relu',
                           kernel_constraint=max_norm(3)))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(int(number_of_nodes / dropout_rate),
                           activation='relu', kernel_constraint=max_norm(3)))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=dropout_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    run_experiment("Dropout network adjusted after",
                   model, number_of_nodes, epochs)
    end_experiment_callback()

    # Dropout network, adjusted all layers
    model = models.Sequential()
    model.add(layers.Dropout(0.2, input_shape=(28 * 28,)))
    model.add(layers.Dense(int(number_of_nodes / dropout_rate),
                           activation='relu', kernel_constraint=max_norm(3)))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(int(number_of_nodes / dropout_rate),
                           activation='relu', kernel_constraint=max_norm(3)))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=dropout_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    run_experiment("Dropout network adjusted all",
                   model, number_of_nodes, epochs)
    end_experiment_callback()

    # Dropout network, dropout before output layer
    model = models.Sequential()
    model.add(layers.Dropout(0.2, input_shape=(28 * 28,)))
    model.add(layers.Dense(int(number_of_nodes / dropout_rate),
                           activation='relu', kernel_constraint=max_norm(3)))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(int(number_of_nodes / dropout_rate),
                           activation='relu', kernel_constraint=max_norm(3)))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=dropout_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    run_experiment("Dropout network all layers",
                   model, number_of_nodes, epochs)
    end_experiment_callback()


# Load and prepare data
start = time.process_time()
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
end = time.process_time()
print("Timing: load and prepare data: {0:.5f}s".format(end - start))

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# Parameters to control the experiments.
# Number of nodes in each layer (note that dropout layers are adjusted,
# increasing the number of nodes).
number_of_nodes = 1024
# Number of epochs for the quick training pass - gives an idea of how the
# experiment is going before we commit more time to it.
epochs_low = 2
# Number of epochs for the high(er) quality training pass - the one likely
# to be used in real-life applications (more representative of the accuracy
# we would expect from the network in actual applications).
epochs_high = 5
# Dropout learning rate multiplier, as recommended in the dropout paper
# ("... dropout net should typically use 10-100 times the learning rate that
# was optimal for a standard neural net.")
dropout_lr_multiplier = 10
# Momentum, as recommended in the dropout paper ("While momentum values of 0.9
# are common for standard nets, with dropout we found that values around 0.95
# to 0.99 work quite a lot better.")
dropout_momentum = 0.95

# SGD optimizers
# The default one
optimizer_sgd_default = optimizers.SGD()
default_sgd_learning_rate = backend.eval(optimizer_sgd_default.lr)
# The one recommended in the paper
optimizer_sgd_dropout = optimizers.SGD(
    lr=default_sgd_learning_rate*dropout_lr_multiplier,
    momentum=dropout_momentum)

# RMSProp optimizers
# The default one
# The paper doesn't mention what optimizer was used in the tests. It looks like
# those tests were done with SGD. I tried RMSProp here because it's a popular
# one nowadays and the one used in the Deep Learning With Python book. It
# results in good accuracy with the default learning rate. If we apply the
# paper suggestion (multiply by 10 or 100), accuracy is much lower.
optimizer_rmsprop_default = optimizers.RMSprop()

# File where the results will be saved (the name encode the parameters used
# in the experiments)
file_name = "MNSIT DNN nodes={} el={} eh={} dlrm={} dm={:.2f}.txt".format(
    number_of_nodes, epochs_low, epochs_high, dropout_lr_multiplier,
    dropout_momentum)


def save_step():
    save_experiments_results(file_name=file_name, display=True)


test_network_configurations(number_of_nodes=number_of_nodes,
                            epochs=epochs_low,
                            standard_optimizer=optimizer_sgd_default,
                            dropout_optimizer=optimizer_sgd_dropout,
                            end_experiment_callback=save_step)

test_network_configurations(number_of_nodes=number_of_nodes,
                            epochs=epochs_low,
                            standard_optimizer=optimizer_rmsprop_default,
                            dropout_optimizer=optimizer_rmsprop_default,
                            end_experiment_callback=save_step)

test_network_configurations(number_of_nodes=number_of_nodes,
                            epochs=epochs_high,
                            standard_optimizer=optimizer_sgd_default,
                            dropout_optimizer=optimizer_sgd_dropout,
                            end_experiment_callback=save_step)

test_network_configurations(number_of_nodes=number_of_nodes,
                            epochs=epochs_high,
                            standard_optimizer=optimizer_rmsprop_default,
                            dropout_optimizer=optimizer_rmsprop_default,
                            end_experiment_callback=save_step)
