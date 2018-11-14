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
                                    "TestAccuracy", "NumberOfUnits",
                                    "DropoutRateInput", "DropoutRateHidden",
                                    "Epochs", "BatchSize", "Optimizer",
                                    "LearningRate", "ModelParamCount",
                                    "TrainingCpuTime", "TestCpuTime"])


def save_experiments_results(file_name, display):
    if display:
        print(experiments)
    with open(file_name, "w") as outfile:
        experiments.to_string(outfile)


def run_experiment(description, model, number_of_units,
                   dropout_rate_input_layer, dropout_rate_hidden_layer,
                   epochs):
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
                                         test_acc, number_of_units,
                                         dropout_rate_input_layer,
                                         dropout_rate_hidden_layer,
                                         epochs, batch_size,
                                         type(optimizer).__name__,
                                         backend.eval(optimizer.lr),
                                         model.count_params(),
                                         training_time, test_time]


def test_network_configurations(number_of_units, dropout_rate_input_layer,
                                dropout_rate_hidden_layer, epochs,
                                max_norm_max_value, standard_optimizer,
                                dropout_optimizer, end_experiment_callback):
    """Test all network configurations with the given parameters."""
    # Standard network (no dropout)
    model = models.Sequential()
    model.add(layers.Dense(number_of_units,
                           activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(number_of_units, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=standard_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    run_experiment("Standard network",
                   model, number_of_units, dropout_rate_input_layer,
                   dropout_rate_hidden_layer, epochs)
    end_experiment_callback()

    # Adjust number of units in each layer: "...if an n-sized layer is optimal
    # for a standard neural net on any given task, a good dropout net should
    # have at least n/p units."
    adjusted_units_hidden = int(
        number_of_units / (1 - dropout_rate_hidden_layer))

    # Dropout without adjustment to number of units (for comparison)
    # Dropout is applied to all layers, as shown in figure 1.b in the paper
    model = models.Sequential()
    model.add(layers.Dropout(dropout_rate_input_layer,
                             input_shape=(28 * 28,)))
    model.add(layers.Dense(number_of_units, activation='relu',
                           kernel_constraint=max_norm(max_norm_max_value)))
    model.add(layers.Dropout(rate=dropout_rate_hidden_layer))
    model.add(layers.Dense(number_of_units, activation='relu',
                           kernel_constraint=max_norm(max_norm_max_value)))
    model.add(layers.Dropout(rate=dropout_rate_hidden_layer))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=dropout_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    run_experiment("Dropout, no unit adjustment",
                   model, number_of_units, dropout_rate_input_layer,
                   dropout_rate_hidden_layer, epochs)
    end_experiment_callback()

    # Dropout with adjustment to number of units
    # Dropout is applied to all layers, as shown in figure 1.b in the paper
    model = models.Sequential()
    model.add(layers.Dropout(dropout_rate_input_layer,
                             input_shape=(28 * 28,)))
    model.add(layers.Dense(adjusted_units_hidden, activation='relu',
                           kernel_constraint=max_norm(max_norm_max_value)))
    model.add(layers.Dropout(rate=dropout_rate_hidden_layer))
    model.add(layers.Dense(number_of_units, activation='relu',
                           kernel_constraint=max_norm(max_norm_max_value)))
    model.add(layers.Dropout(rate=dropout_rate_hidden_layer))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=dropout_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    run_experiment("Dropout, units adjusted",
                   model, number_of_units, dropout_rate_input_layer,
                   dropout_rate_hidden_layer, epochs)
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
# Number of units in each layer (note that dropout layers are adjusted,
# increasing the number of units used in the network).
number_of_units = 1024
# Dropout rate for the input layer ("For input layers, the choice depends on
# the kind of input. For real-valued inputs (image patches or speech frames),
# a typical value is 0.8.)" [Note: keras uses "drop", not "keep" rate]
dropout_rate_input_layer = 0.2
# Dropout rate for the input layer ("Typical values of p for hidden units are
# in the range 0.5 to 0.8.)" [Note: keras uses "drop", not "keep" rate]
dropout_rate_hidden_layer = 0.5
# Number of epochs for the quick training pass - gives an idea of how the
# experiment is going before we commit more time to it.
epochs_low = 2
# Number of epochs for the high(er) quality training pass - the one likely
# to be used in real-life applications (more representative of the accuracy
# we would expect from the network in actual applications).
epochs_high = 10
# Dropout learning rate multiplier, as recommended in the dropout paper
# ("... dropout net should typically use 10-100 times the learning rate that
# was optimal for a standard neural net.")
dropout_lr_multiplier = 10.0
# Momentum, as recommended in the dropout paper ("While momentum values of 0.9
# are common for standard nets, with dropout we found that values around 0.95
# to 0.99 work quite a lot better.")
dropout_momentum = 0.95
# Max norm max value. The paper recommends its usage ("Although dropout alone
# gives significant improvements, using dropout along with max-norm...") with
# a range of 3 to 4 ("Typical values of c range from 3 to 4.")
max_norm_max_value = 3

# File where the results will be saved (the name encodes the parameters used
# in the experiments)
file_name = "MNIST DNN units={:04d} dri={:0.2f} drh={:0.2f} el={:02d} eh={:03d} dlrm={:03.1f} dm={:0.2f} mn={}.txt" \
    .format(number_of_units, dropout_rate_input_layer, dropout_rate_hidden_layer,
            epochs_low, epochs_high, dropout_lr_multiplier,
            dropout_momentum, max_norm_max_value)

# SGD optimizers
# The default one
optimizer_sgd_default = optimizers.SGD()
default_sgd_learning_rate = backend.eval(optimizer_sgd_default.lr)
# The one recommended in the paper
optimizer_sgd_dropout = optimizers.SGD(
    lr=default_sgd_learning_rate * dropout_lr_multiplier,
    momentum=dropout_momentum)

# RMSProp optimizers
# The default one
# The paper doesn't mention what optimizer was used in the tests. It looks like
# those tests were done with SGD. I tried RMSProp here because it's a popular
# one nowadays and the one used in the Deep Learning With Python book. It
# results in good accuracy with the default learning rate.
optimizer_rmsprop_default = optimizers.RMSprop()
# Increasing the learn rate for the RMSProp optimizer resulted in much worse
# accuracy. To prevent that we use the default optimizer for dropout.
optimizer_rmsprop_dropout = optimizer_rmsprop_default


def save_step():
    save_experiments_results(file_name=file_name, display=True)


test_network_configurations(number_of_units=number_of_units,
                            dropout_rate_input_layer=dropout_rate_input_layer,
                            dropout_rate_hidden_layer=dropout_rate_hidden_layer,
                            epochs=epochs_low,
                            standard_optimizer=optimizer_sgd_default,
                            dropout_optimizer=optimizer_sgd_dropout,
                            max_norm_max_value=max_norm_max_value,
                            end_experiment_callback=save_step)

test_network_configurations(number_of_units=number_of_units,
                            dropout_rate_input_layer=dropout_rate_input_layer,
                            dropout_rate_hidden_layer=dropout_rate_hidden_layer,
                            epochs=epochs_low,
                            standard_optimizer=optimizer_rmsprop_default,
                            dropout_optimizer=optimizer_rmsprop_dropout,
                            max_norm_max_value=max_norm_max_value,
                            end_experiment_callback=save_step)

test_network_configurations(number_of_units=number_of_units,
                            dropout_rate_input_layer=dropout_rate_input_layer,
                            dropout_rate_hidden_layer=dropout_rate_hidden_layer,
                            epochs=epochs_high,
                            standard_optimizer=optimizer_sgd_default,
                            dropout_optimizer=optimizer_sgd_dropout,
                            max_norm_max_value=max_norm_max_value,
                            end_experiment_callback=save_step)

test_network_configurations(number_of_units=number_of_units,
                            dropout_rate_input_layer=dropout_rate_input_layer,
                            dropout_rate_hidden_layer=dropout_rate_hidden_layer,
                            epochs=epochs_high,
                            standard_optimizer=optimizer_rmsprop_default,
                            dropout_optimizer=optimizer_rmsprop_dropout,
                            max_norm_max_value=max_norm_max_value,
                            end_experiment_callback=save_step)
