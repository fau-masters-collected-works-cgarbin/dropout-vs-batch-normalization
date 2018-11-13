"""
CAP-6619 Deep Learning Fall 2018 term project
MNIST with regular model, dropout and batch normalization
"""

import time
import pandas as pd
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.datasets import mnist

# Store data from the experiments
experiments = pd.DataFrame(columns=["Description", "DataSetName", "Loss",
                                    "Accuracy", "Epochs", "BatchSize",
                                    "ModelParamCount", "TrainingCpuTime",
                                    "TestCpuTime"])


def add_experiment(description, data_set_name, loss, accuracy, epochs,
                   batch_size, model_param_count, training_time, test_time):
    """Add an entry to the dataframe that keeps track of experiments."""
    new_row = [description, data_set_name, loss, accuracy, epochs,
               batch_size, model_param_count, training_time, test_time]
    experiments.loc[len(experiments)] = new_row


def run_experiment(description, model):
    """Run an experiment: train and test the network, save results"""
    print(description)

    start = time.process_time()
    epochs = 2
    batch_size = 128
    model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)
    training_time = time.process_time() - start

    start = time.process_time()
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    test_time = time.process_time() - start

    add_experiment(description, "MNIST", test_loss, test_acc,
                   epochs, batch_size, model.count_params(),
                   training_time, test_time)


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

# Standard network -----------------------------------------------------------
model = models.Sequential()
model.add(layers.Dense(1024, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

run_experiment("Standard network, 1024 nodes", model)

# Dropout network, no adjustemt ----------------------------------------------
dropout_rate = 0.5
model = models.Sequential()
model.add(layers.Dense(1024, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dropout(rate=dropout_rate))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

run_experiment("Dropout network not adjusted, 1024 nodes", model)

# Dropout network adjusted before --------------------------------------------
model = models.Sequential()
model.add(layers.Dense(int(1024 / dropout_rate),
                       activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dropout(rate=dropout_rate))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

run_experiment("Dropout network adjusted before, 1024 nodes", model)

# Dropout network, adjusted after --------------------------------------------
model = models.Sequential()
model.add(layers.Dense(1024, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dropout(rate=dropout_rate))
model.add(layers.Dense(int(1024 / dropout_rate), activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

run_experiment("Dropout network adjusted after, 1024 nodes", model)

# Dropout network, adjusted all layers ---------------------------------------
model = models.Sequential()
model.add(layers.Dense(int(1024 / dropout_rate),
                       activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dropout(rate=dropout_rate))
model.add(layers.Dense(int(1024 / dropout_rate), activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

run_experiment("Dropout network adjusted all, 1024 nodes", model)

# Dropout network, dropout before output layer -------------------------------
model = models.Sequential()
model.add(layers.Dense(int(1024 / dropout_rate),
                       activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dropout(rate=dropout_rate))
model.add(layers.Dense(int(1024 / dropout_rate), activation='relu'))
model.add(layers.Dropout(rate=dropout_rate))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

run_experiment("Dropout network all layers, 1024 nodes", model)

print(experiments)
