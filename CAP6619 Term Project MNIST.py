"""
CAP-6619 Deep Learning Fall 2018 term project
MNIST with regular model, dropout and batch normalization
"""

import time
import pandas as pd
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical
from keras.datasets import mnist

# Store data from the experiments
experiments = pd.DataFrame(columns=["Description", "DataSetName", "TestLoss",
                                    "TestAccuracy", "NumeberOfNodes", "Epochs",
                                    "BatchSize", "Optimizer",
                                    "ModelParamCount", "TrainingCpuTime",
                                    "TestCpuTime"])


def run_experiment(description, model, number_of_nodes, epochs, optimizer):
    """Run an experiment: train and test the network, save results"""
    print(description)

    start = time.process_time()
    batch_size = 128
    model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)
    training_time = time.process_time() - start

    start = time.process_time()
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    test_time = time.process_time() - start

    experiments.loc[len(experiments)] = [description, "MNIST", test_loss,
                                         test_acc, number_of_nodes, epochs,
                                         batch_size, optimizer,
                                         model.count_params(),
                                         training_time, test_time]


def test_network_configurations(number_of_nodes, epochs, optimizer, file_name):
    """Test all network configurations with the given parameters."""
    # Standard network
    model = models.Sequential()
    model.add(layers.Dense(number_of_nodes,
                           activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(number_of_nodes, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    run_experiment("Standard network, 1024 nodes",
                   model, number_of_nodes, epochs, optimizer)

    # Dropout network, no adjustment
    dropout_rate = 0.5
    model = models.Sequential()
    model.add(layers.Dense(number_of_nodes,
                           activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(number_of_nodes, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizers.SGD(lr=0.1, momentum=0.95),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    run_experiment("Dropout network not adjusted, 1024 nodes",
                   model, number_of_nodes, epochs, optimizer)

    # Dropout network adjusted before
    model = models.Sequential()
    model.add(layers.Dense(int(number_of_nodes / dropout_rate),
                           activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(number_of_nodes, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizers.SGD(lr=0.1, momentum=0.95),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    run_experiment("Dropout network adjusted before, 1024 nodes",
                   model, number_of_nodes, epochs, optimizer)

    # Dropout network, adjusted after
    model = models.Sequential()
    model.add(layers.Dense(number_of_nodes,
                           activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(int(number_of_nodes / dropout_rate), activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizers.SGD(lr=0.1, momentum=0.95),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    run_experiment("Dropout network adjusted after, 1024 nodes",
                   model, number_of_nodes, epochs, optimizer)

    # Dropout network, adjusted all layers
    model = models.Sequential()
    model.add(layers.Dense(int(number_of_nodes / dropout_rate),
                           activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(int(number_of_nodes / dropout_rate), activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizers.SGD(lr=0.1, momentum=0.95),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    run_experiment("Dropout network adjusted all, 1024 nodes",
                   model, number_of_nodes, epochs, optimizer)

    # Dropout network, dropout before output layer
    model = models.Sequential()
    model.add(layers.Dense(int(number_of_nodes / dropout_rate),
                           activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(int(number_of_nodes / dropout_rate), activation='relu'))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizers.SGD(lr=0.1, momentum=0.95),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    run_experiment("Dropout network all layers, 1024 nodes",
                   model, number_of_nodes, epochs, optimizer)

    print(experiments)

    with open(file_name, "w") as outfile:
        experiments.to_string(outfile)


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

test_network_configurations(number_of_nodes=1024, epochs=2,
                            optimizer="sgd",
                            file_name="MNIST SGD 2 epochs.txt")

# test_network_configurations(number_of_nodes=1024, epochs=2,
#                             optimizer="rmsprop",
#                             file_name="MNIST RMSProp 2 epochs.txt")

test_network_configurations(number_of_nodes=1024, epochs=5,
                            optimizer="sgd",
                            file_name="MNIST SGD 5 epochs.txt")

# test_network_configurations(number_of_nodes=1024, epochs=5,
#                             optimizer="rmsprop",
#                             file_name="MNIST RMSProp 5 epochs.txt")
