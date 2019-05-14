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

The report compares the peformance and accuracy of [Dropout](https://arxiv.org/abs/1207.0580) and
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

A number of hyperparameters were tested: number of layers, number of units in each layer,
learning rate, weigth decay, dropout rates for input layer and hidden layer. The MLP network
was also tested with a non-adaptive optimizer (SGD) and an adaptive optimizer (RMSProp). See
the [report](./report/CAP6619_term_project_cgarbin.pdf) for more details.

The raw results are available [in this folder](./test_results). The 
[report](./report/CAP6619_term_project_cgarbin.pdf) has some analysis of those results.
More analysis could be done on those results.

# How the code is structured

The code is split into these directories:

- `mlp`: the MLP tests
- `cnn`: the CNN tests
- `reference_implementations`: sample code from other sources, used as initial tests and
   to get more familiar with Keras.

Within each directory the files are named with the network configuration they test.

# How to run the experiments

The code is written with the purpose of running combinations of hyperparameters. To 
facilitate that, a Python file generates bash scripts that can be used to exercise that
combination of parameters.

For example, to test all combinations of MLP Batch Normalization:

1. Run `./mlp/batch_normalization/CAP6619_term_project_mnist_mlp_batchnorm_test_generator.py`
   to create the shell scripts with the tests to cover the combination of hyperparameters.
1. Run the shell scripts.

### What drives the combination of hyperparameters to test

The tests are driven by the combination of parameters defined in the test generator file.
The parameters are specified in named tuples. This is the one used to generate MLP Batch
Normalization tests with the SGD optimizer:

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

Change the named tuple to run different tests.

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

# Results from experiments

Raw data generated from the experiments executed for the report are available
[in this directory](./test_results).

Analysis of the results is available [in the report](./report/CAP6619_term_project_cgarbin.pdf).

More analysis could be done with the data collected. That's what I could do
based on the time I had and my limited knowledge at that point.

# What needs to be improved

Gathering the data was a great learning experience. Knowing what I know now, I'd
have done a few things differently:

- Force overfit: Dropout and Batch Normalization fight overfitting. Therefore,
  more interesting data would have been produced if first I had made sure the
  network was overfitting. This could have been done by reducing the number of
  samples used to train the newtwork.
- Extract more data from the results: the results collected quite a bit of data
  for each combination of parameters. Only some basic analysis was done to write
  the report. More analysis, e.g. what is the effect of learning rate changes, of
  momentum changes, etc., could be done. 
- Extract repeated code: there is a fair bit of copy and paste in the code,
  especially in the CNN tests. It should be refactored and removed.
- Split the standard MLP tests from the Dropout MLP tests: they are embedded in
  one file. It would be easier to manage them if they were in separate files,
  like it was it was done for the CNN code.
- See more "TODO" in the code: there are few "TODO" in the code, pointing to 
  more specific improvements that could be done.

# License

Licensed as [MIT](./LICENSE). If you use parts of this project, I'd appreciate a
reference to this work.

And please let me know you are referring to it - personal curiosity.

# Miscellanea

This was my first real-file (of sorts) experience with machine learning and the first time
I wrote a significant amount of Python and Keras code. It's not a polished result by any means.
Suggestions for improvements are always appreciated.


