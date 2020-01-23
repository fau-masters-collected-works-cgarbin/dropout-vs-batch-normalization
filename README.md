# Deep Learning Term Project

This repository has the code used for Florida Atlantic University's CAP-6619 Deep Learning class
term project, Fall 2018.

The objective of the term project was to compare [Dropout](https://arxiv.org/abs/1207.0580) and
[Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf):

- Effect on training time
- Effect on test (inference) time
- Effect on accuracy
- Effect on memory usage

Python and Keras were used to test several network configurations. See more details below.

The report is available [here](./report/CAP6619_term_project_cgarbin.pdf). Note that this is not
the report I delivered for the class. Every so often I correct a mistake in it or improve an
item. To see the original report, go to the first version in the file history.

# What the report covers

The report compares the performance and accuracy of [Dropout](https://arxiv.org/abs/1207.0580) and
[Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf):

- How long it takes to train a network (to run a specific number of epochs, to be more precise).
- How long it takes to make a prediction with a trained network.
- How much memory the network uses, measured indirectly with Keras' `param_count()`.

To gather those numbers the code runs a series of tests with different network configurations
and different hyperparameters.

The network configurations tested:

- MLP - multilayer perceptron network (only densely connected layers) was tested with [MNIST](http://yann.lecun.com/exdb/mnist/).
- CNN - convolutional neural network (convolutional and max-pooling layers) was tested with
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

Several hyperparameters were tested: number of layers, number of units in each layer,
learning rate, weight decay, dropout rates for input layer and hidden layer. The MLP network
was also tested with a non-adaptive optimizer (SGD) and an adaptive optimizer (RMSProp). See
the [report](./report/CAP6619_term_project_cgarbin.pdf) for more details.

The raw results are available [in this folder](./test_results). The 
[report](./report/CAP6619_term_project_cgarbin.pdf) has some analysis of those results.
More analysis could be done on those results.

## Publication in Springer's Multimedia Tools and Applications

This report was later published in [Springer's Multimedia Tools and Applications](https://link.springer.com/article/10.1007/s11042-019-08453-9).

# Results from experiments

Raw data generated from the experiments executed for the report are available
[in this directory](./test_results).

Analysis of the results is available [in the report](./report/CAP6619_term_project_cgarbin.pdf).

More analysis could be done with the data collected. That's what I could do
based on the time I had and my limited knowledge at that point.

# How to install the environment and run the experiments

## How to install the environment and dependencies

#### Install Python 3

The project uses Python 3. 

Verify that you have Python 3.x installed: `python --version` should print `Python 3.x.y`. If
it prints `Python 2.x.y`, try `python3 --version`. If that still doesn't work, please install
Python 3.x before proceeding. The official Python download site is
[here](https://www.python.org/downloads/).

From this point on, the instructions assume that **Python 3 is installed as `python3`**. 

#### Clone the repository

```bash
git clone https://github.com/fau-masters-collected-works-cgarbin/cap6619-deep-learning-term-project.git
```

The repository is now in the directory `cap6619-deep-learning-term-project`.

#### Create a Python virtual environment

**IMPORTANT**: The project is configured for TensorFlow without GPU. If you are using it on a GPU-enabled system, open `requirements.txt` and follow th instructions there to use the GPU version of TensorFlow.

The project depends on specific versions of Keras and TensorFlow. The safest way to install the
correct versions, without affecting other projects you have on your computer, is to create a Python
virtual environment specifically for this project.

The official guide to Python virtual environment is [here](https://docs.python.org/3/tutorial/venv.html).

Execute these commands to create and activate a virtual environment for the project:

1. `cd cap6619-deep-learning-term-project` (if you are not yet in the project directory)
1. `python3 -m venv env`
1. `source env/bin/activate` (or in Windows: `env\Scripts\activate.bat`)

#### Install the dependencies

The project dependencies are listed in the `requirements.txt` file. To install them,
execute this command:

`pip install -r requirements.txt`

This may take several minutes to complete. Once it is done, you are ready to run the
experiments.

## How to run the experiments

The code is split into these directories:

- `mlp`: the MLP tests
- `cnn`: the CNN tests

Within each directory, the files are named with the network configuration they test.

### MLP experiments

The experiments are driven by the combination of parameters defined in the test generator
file. The parameters are specified in named tuples. This is the one used to generate MLP
Batch Normalization tests with the SGD optimizer:

```python
    batchnorm_sgd = Parameters(
        experiment_name='batchnorm_mnist_mlp_sgd',
        network=['batch_normalization'],
        optimizer=['sgd'],
        hidden_layers=['2', '3', '4'],
        units_per_layer=['1024', '2048'],
        epochs=['5', '20', '50'],
        batch_size=['128'],
        # Test with the Keras default 0.01 and a higer rate because the paper
        # recommends 'Increase learning rate.'
        learning_rate=['0.01', '0.1'],
        # Test with Keras default 0.0 (no decay) and a small decay
        decay=['0.0', '0.0001'],
        # Test with Keras default (no momentum) and some momentum
        sgd_momentum=['0.0', '0.95'],
    )
```

#### MLP with batch normalization

```bash
cd mlp/batch_normalization
python3 CAP6619_term_project_mnist_mlp_batchnorm_test_generator.py

# For a quick check of the environment
./batchnorm_mnist_mlp_quick_test.sh

# Using SGD
./batchnorm_mnist_mlp_sgd.sh
# Using RMSProp
./batchnorm_mnist_mlp_rmsprop.sh

# Merges all tests into one file and top 10 files
python3 CAP6619_term_project_mnist_mlp_batchnorm_analysis.py
```

#### MLP with dropout

```bash
cd mlp/dropout
python3 CAP6619_term_project_mnist_mlp_dropout_test_generator.py

# For a quick check of the environment
./dropout_mnist_mlp_quick_test.sh
# Regular MLP network (no dropout) with SGD to use as baseline
./dropout_mnist_mlp_standard_sgd.sh
# Regular MLP network (no dropout) with RMSprop to use as baseline
./dropout_mnist_mlp_standard_rmsprop.sh
# Dropout MLP network without adjustment and with SGD
./dropout_mnist_mlp_dropout_no_adjustment_sgd.sh		
# Dropout MLP network with adjustment and with SGD
./dropout_mnist_mlp_dropout_sgd.sh
# Regular MLP network without adjustment with RMSprop
./dropout_mnist_mlp_dropout_no_adjustment_rmsprop.sh	
# Regular MLP network without adjustment with RMSprop
./dropout_mnist_mlp_dropout_rmsprop.sh			

# Merges all tests into one file and top 10 files
./CAP6619_term_project_mnist_mlp_dropout_analysis.py
```

### CNN experiments

The experiments are driven by command line parameters. Shell scripts encapsulate the
experiements.

```bash
cd cnn

# For a quick check of the environment
./cnn_test_quick.sh
# All CNN experiments
./cnn_test_all.sh 
```

### Where results are saved

Results are saved in these files:

- A `..._results.txt` file collects the data for each test, e.g. training time, 
  model parameter count, etc. There is one line for each test. See an example
  [here](./test_results/mlp/batch_normalization/sgd/batchnorm_mnist_mlp_sgd_results.txt).
- Several `..._history.json` files, one for each test. It contains the training
  and validation loss/accuracy. It's a JSON file with the contents of the `History`
  object created by Keras during training. The name of the file encodes the values
  of the hyperparameters used for that text.
  See several examples [in this directory](./test_results/mlp/batch_normalization/sgd).  

# What needs to be improved

Gathering the data was a great learning experience. Knowing what I know now, I'd
have done a few things differently:

- Force overfit: Dropout and Batch Normalization fight overfitting. Therefore,
  more interesting data would have been produced if first I had made sure the
  network was overfitting. This could have been done by reducing the number of
  samples used to train the network.
- Extract more data from the results: the results collected quite a bit of data
  for each combination of parameters. Only some basic analysis was done to write
  the report. More analysis, e.g. what is the effect of learning rate changes, of
  momentum changes, etc., could be done. 
- Extract repeated code: there is a fair bit of copy and paste in the code,
  especially in the CNN tests. It should be refactored and removed.
- Split the standard MLP tests from the Dropout MLP tests: they are embedded in
  one file. It would be easier to manage them if they were in separate files,
  like it was done for the CNN code.
- See more "TODO" in the code: there are few "TODO" in the code, pointing to 
  more specific improvements that could be done.

# License

Licensed as [MIT](./LICENSE). If you use parts of this project, I'd linking back to
this repository and a citation of the
[Springer Multimedia Tools and Applications paper](https://link.springer.com/article/10.1007/s11042-019-08453-9#citeas).

And please let me know you are referring to it - personal curiosity.

# Miscellanea

This was my first real-file (of sorts) experience with machine learning and the first time
I wrote a significant amount of Python and Keras code. It's not a polished result by any means.

Suggestions for improvements are always appreciated.
