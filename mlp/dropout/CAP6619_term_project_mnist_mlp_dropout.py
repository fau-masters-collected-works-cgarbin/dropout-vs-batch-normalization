"""
CAP-6619 Deep Learning Fall 2018 term project
MNIST with standard deep neural network and dropout

Dropout paper: http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf
"""
import time
import pandas as pd
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
from CAP6619_term_project_mnist_mlp_dropout_parameters import Parameters


def create_model(parameters):
    """Create a model described by the given parameters."""
    # To make lines shorter
    p = parameters

    model = models.Sequential()

    if p.network == 'standard':
        model.add(layers.Dense(p.units_per_layer,
                               activation='relu',
                               input_shape=(pixels_per_image,)))
        for _ in range(p.hidden_layers - 1):
            model.add(layers.Dense(p.units_per_layer, activation='relu'))
    elif p.network in ('dropout_no_adjustment', 'dropout'):
        units_hidden_layer = 0
        if p.network == 'dropout':
            # Adjust number of units in each layer: '...if an n-sized layer is
            # optimal for a standard neural net on any given task, a good
            # dropout net should have at least n/p units.' [Note that Keras
            # uses a 'drop' rate, not 'keep', hence the '1 -'].
            units_hidden_layer = int(
                p.units_per_layer / (1 - p.dropout_rate_hidden_layer))
        else:
            units_hidden_layer = p.units_per_layer

        model.add(layers.Dropout(p.dropout_rate_input_layer,
                                 input_shape=(pixels_per_image,)))
        for _ in range(p.hidden_layers):
            # Reason to use he_normal initializer: source code the paper points
            # to has 'initialization: DENSE_GAUSSIAN_SQRT_FAN_IN' for weights.
            if p.max_norm_max_value == 'none':
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
    if p.optimizer == 'sgd':
        optimizer = optimizers.SGD(
            p.learning_rate, momentum=float(p.sgd_momentum), decay=p.decay)
    elif p.optimizer == 'rmsprop':
        optimizer = optimizers.RMSprop(p.learning_rate, decay=p.decay)
    else:
        assert False  # Invalid optimizer

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def test_model(parameters, end_experiment_callback, verbose):
    """Test one model: create, train, evaluate with test data and save
    results."""
    # To make lines shorter
    p = parameters

    model = create_model(parameters)

    start = time.process_time()
    model.fit(train_images, train_labels, epochs=p.epochs,
              batch_size=p.batch_size,
              validation_data=(test_images, test_labels),
              verbose=verbose)
    training_time = time.process_time() - start

    start = time.process_time()
    test_loss, test_acc = model.evaluate(
        test_images, test_labels, verbose=verbose)

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
        p.experiment_name, datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),
        'MNIST', p.network, optimizer_name, test_loss, test_acc,
        p.hidden_layers, p.units_per_layer, p.epochs, p.batch_size,
        p.dropout_rate_input_layer, p.dropout_rate_hidden_layer,
        backend.eval(optimizer.lr), p.decay, p.sgd_momentum,
        p.max_norm_max_value, model.count_params(), training_time, test_time]

    # Show progress so far to the user
    print(experiments)

    # Save progress so far into the file used for this experiment
    results_file = p.experiment_name + '_results.txt'
    # First, get a formatted string; if we use to_string(header=False) it
    # will use only one space between columns, instead of formatting
    # considering the column name (the header).
    # Also ensure that we use a fixed-length size for the network name to
    # keep the columns aligned.
    output = StringIO()
    experiments.to_string(output, formatters={
                          'Network': '{:>25}'.format}, header=True)
    if os.path.isfile(results_file):
        # File already exists - append data without column names.
        with open(results_file, 'a') as f:
            f.write(os.linesep)
            f.write(output.getvalue().splitlines()[1])
        output.close()
    else:
        # File doesn't exist yet - create and write column names + data
        with open(results_file, 'w') as f:
            f.write(output.getvalue())

    # Save training history and model for this specific experiment.
    # The model object must be a trained model, which means it has a `history`
    # object with the training results for each epoch.
    # We need to save the history separately because `model.save` won't save
    # it - it saves only the model data.

    # File where the training history and model will be saved. The name encodes
    # the test the parameters used in the epxeriment.
    base_name_template = ('{}_nw={}_opt={}_hl={:03d}_uhl={:04d}_e={:02d}'
                          '_bs={:04d}_dri={:0.2f}_drh={:0.2f}_lr={:0.4f}'
                          '_d={:0.4f}_m={}_mn={}')
    base_name = base_name_template.format(
        p.experiment_name, p.network, p.optimizer, p.hidden_layers,
        p.units_per_layer, p.epochs, p.batch_size,
        p.dropout_rate_input_layer, p.dropout_rate_hidden_layer,
        p.learning_rate, p.decay, p.sgd_momentum, p.max_norm_max_value,
    )

    with open(base_name + '_history.json', 'w') as f:
        json.dump(model.history.history, f)
    # Uncomment to save the model - it may take quite a bit of disk space
    # model.save(base_name + '_model.h5')


def parse_command_line():
    """Parse command line parameters into a `Parameters` variable."""
    ap = ArgumentParser(description='Dropout with MNIST data set.')

    ap.add_argument('--experiment_name', type=str)
    ap.add_argument('--network', type=str)
    ap.add_argument('--optimizer', type=str)
    ap.add_argument('--hidden_layers', type=int)
    ap.add_argument('--units_per_layer', type=int)
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--batch_size', type=int)
    ap.add_argument('--dropout_rate_input_layer', type=float)
    ap.add_argument('--dropout_rate_hidden_layer', type=float)
    ap.add_argument('--learning_rate', type=float)
    ap.add_argument('--decay', type=float)
    ap.add_argument('--sgd_momentum', type=str)
    ap.add_argument('--max_norm_max_value', type=str)

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
    'ExperimentName', 'TestTime', 'DataSetName', 'Network', 'Optimizer',
    'TestLoss', 'TestAccuracy', 'HiddenLayers', 'UnitsPerLayer', 'Epochs',
    'BatchSize', 'DropoutRateInput', 'DropoutRateHidden', 'LearningRate',
    'Decay', 'SgdMomentum', 'MaxNorm', 'ModelParamCount', 'TrainingCpuTime',
    'TestCpuTime'])


# The input shape: pixels_per_image pixels images from MNIST data set
pixels_per_image = 28 * 28

# Load and prepare data.
# Note that they are global variables used in the functions above. A future
# improvement could be to add them to the parameters data structure.
start = time.process_time()
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print('Timing: load and prepare data: {0:.5f}s'.format(
    time.process_time() - start))

train_images = train_images.reshape((60000, pixels_per_image))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, pixels_per_image))
test_images = test_images.astype('float32') / 255

# Change this to 'False' when testing from the command line. Leave set to True
# when launching from the IDE and change the parameters below (it's faster
# than dealing with launch.json).
ide_test = True
# Show a warning to let user now we are ignoring command line parameters
if ide_test:
    print('\n\n  --- Running from IDE - ignoring command line\n\n')

p = None
if ide_test:
    p = Parameters(
        experiment_name='dropout_mnist_mlp',
        network='dropout',
        optimizer='rmsprop',
        hidden_layers=4,
        units_per_layer=512,
        epochs=5,
        batch_size=128,
        dropout_rate_input_layer=0.1,
        dropout_rate_hidden_layer=0.5,
        learning_rate=0.001,
        decay=0.0,
        sgd_momentum='none',
        max_norm_max_value='none',
    )
else:
    p = parse_command_line()

test_model(p, save_experiment, verbose=1 if ide_test else 2)
