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
from keras import backend
from keras.datasets import mnist
from datetime import datetime


def create_model(parameters):
    """Create a model described by the given parameters."""
    # To make lines shorter
    p = parameters

    # The only network type currently supported.
    # Use the dropout code to test standard networks (no dropout, not batch
    # normalization).
    assert p.network == "batch_normalization"

    # "We added Batch Normalization to each hidden layer of the network,..."
    # Note on the ativation function: the paper states "Each hidden layer...
    # with sigmoid nonlinearity...", but tests with ReLU resulted in
    # significantly better accuracy for SGD and slightly better for RMSprop,
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

    # Create the optimizer
    optimizer = None
    if p.optimizer == "sgd":
        optimizer = optimizers.SGD(p.learning_rate, decay=p.decay)
    elif p.optimizer == "rmsprop":
        optimizer = optimizers.RMSprop(p.learning_rate, decay=p.decay)
    else:
        assert False  # Invalid optimizer

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
    # From the paper: "Shuffle training examples more thoroughly." shuffle=True
    # is the default in model.fit already, so no need to explicitly add it.
    model.fit(train_images, train_labels, epochs=p.epochs,
              batch_size=p.batch_size)
    training_time = time.process_time() - start

    start = time.process_time()
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    test_time = time.process_time() - start

    end_experiment_callback(parameters, model, test_loss, test_acc,
                            training_time, test_time)


def save_experiment(parameters, model, test_loss, test_acc,
                    training_time, test_time):
    """Save results from one experiment"""
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
        backend.eval(optimizer.lr), p.decay, model.count_params(),
        training_time, test_time]
    # Show progress so far
    print(experiments)

    # File where the results will be saved (the name encodes the parameters
    # used in the experiments)
    base_name_prefix = "MNIST_DNN_BatchNorm"
    base_name_template = ("{}_hl={:03d}_uhl={:04d}_e={:02d}_bs={:04d}_o={}"
                          "_lr={:0.4f}_d={:0.4f}")
    base_name = base_name_template.format(
        base_name_prefix, p.hidden_layers, p.units_per_layer, p.epochs,
        p.batch_size, p.optimizer, p.learning_rate, p.decay)

    # Save progress so far into one file
    with open(base_name + ".txt", "w") as f:
        experiments.to_string(f)

    # Save training history and model for this specific experiment.
    # The model object must be a trained model, which means it has a `history`
    # object with the training results for each epoch.
    # We need to save the history separately because `model.save` won't save
    # it - it saves only the model data.
    results_file = base_name + "_" + p.experiment_name + "_" + optimizer_name + "_"
    import json
    with open(results_file + "history.json", 'w') as f:
        json.dump(model.history.history, f)
    # Uncomment to save the model - it may take quite a bit of disk space
    # model.save(results_file + "model.h5")


def parse_command_line():
    """Parse command line parameters into a `Parameters` variable."""
    from argparse import ArgumentParser
    ap = ArgumentParser(description='Batch normalization with MNIST data set.')

    # Format: short parameter name, long name, default value (if not specified)
    ap.add_argument("--experiment_name", type=str)
    ap.add_argument("--network", type=str)
    ap.add_argument("--optimizer", type=str)
    ap.add_argument("--hidden_layers", default=2, type=int)
    ap.add_argument("--units_per_layer", default=512, type=int)
    ap.add_argument("--epochs", default=5, type=int)
    ap.add_argument("--batch_size", default=128, type=int)
    ap.add_argument("--learning_rate", default=0.01, type=float)
    ap.add_argument("--decay", type=float)

    args = ap.parse_args()

    return Parameters(
        experiment_name=args.experiment_name,
        network=args.network,
        optimizer=args.optimizer,
        hidden_layers=args.hidden_layers,
        units_per_layer=args.units_per_layer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        decay=args.decay,
    )


# Store data from the experiments
experiments = pd.DataFrame(columns=[
    "ExperimentName", "TestTime", "DataSetName", "Network", "Optimizer",
    "TestLoss", "TestAccuracy", "HiddenLayers", "UnitsPerLayer", "Epochs",
    "BatchSize", "LearningRate", "Decay", "ModelParamCount", "TrainingCpuTime",
    "TestCpuTime"])

# Parameters to control the experiments.
Parameters = collections.namedtuple("Parameters", [
    # A brief description of the experiment. Will be used as part of file names
    # to prevent collisions with other experiments. Cannot contain spaces to
    # work correctly as a command line parameter.
    "experiment_name",
    # Type of network to test: only valid choice is "batch_normalization". The
    # standard network type (no batch normalization) can be tested with the
    # dropout code, to avoid duplication.
    "network",
    # Type of optimizer to use: "sgd" or "rmsprop".
    "optimizer",
    # Number of hidden layers in the network. When a batch normalization
    # network is used, each hidden layer will be followed by a batch
    # normalization layer.
    "hidden_layers",
    # Number of units in each layer.
    "units_per_layer",
    # Number of epochs to train.
    "epochs",
    # Number of samples in each batch.
    "batch_size",
    # Learning rate - can be increased for batch normalization ("In a batch-
    # normalized model, we have been able to achieve a training speedup from
    # higher learning rates, with no ill side effects").
    # The default Keras values are SGD: 0.01, RMSProp: 0.001 (see
    # https://keras.io/optimizers/).
    "learning_rate",
    # Weight decay (L2). The source code the paper points to has "l2_decay"
    # set to 0.001. The default in Keras for SGD and RMSProp is 0.0.
    "decay",
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
        experiment_name="batchnorm_mnist_dnn",
        network="batch_normalization",
        optimizer="sgd",
        hidden_layers=1,
        units_per_layer=512,
        epochs=2,
        batch_size=128,
        learning_rate=0.1,
        decay=0.0,
    )
else:
    p = parse_command_line()

test_model(p, save_experiment)
