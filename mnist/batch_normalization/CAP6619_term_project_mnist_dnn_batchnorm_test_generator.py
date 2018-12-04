"""
CAP-6619 Deep Learning Fall 2018 term project
MNIST with standard deep neural network and batch normalization

Create shell scripts with all tests we need to execute.

Batch normalization paper: https://arxiv.org/pdf/1502.03167.pdf

Some default values from Keras to keep in mind:

* Learning rate: SGD=0.01, RMSprop=0.001
* Momentum: SGD=0.0 (RMSprop doesn't have momentum)
* Decay: 0.0 for SGD and RMSprop
* MaxNorm:  not used in either, must be explicitly added

Some notes from the paper (verbatim) about hyperparameters adjustments:

{quote}
Simply adding Batch Normalization to a network does not take full advantage of
our method. To do so, we further changed the network and its training
parameters, as follows:

* Increase learning rate. In a batch-normalized model, we have been able to
  achieve a training speedup from higher learning rates, with no ill side
  effects (Sec. 3.3).
* Remove Dropout. As described in Sec. 3.4, Batch Normalization fulfills some
  of the same goals as Dropout. Removing Dropout from Modified BN-Inception
  speeds up training, without increasing overfitting.
* Reduce the L2 weight regularization. While in Inception an L2 loss on the
  model parameters controls overfitting, in Modified BN-Inception the weight of
  this loss is reduced by a factor of 5. We find that this improves the
  accuracy on the held-out validation data.
* Accelerate the learning rate decay. In training Inception, learning rate was
  decayed exponentially. Because our network trains faster than Inception, we
  lower the learning rate 6 times faster.
* Remove Local Response Normalization While Inception and other networks
  (Srivastava et al., 2014) benefit from it, we found that with Batch
  Normalization it is not necessary.
* Shuffle training examples more thoroughly. We enabled within-shard shuffling
  of the training data, which prevents the same examples from always appearing
  in a mini-batch together. This led to about 1% improvements in the validation
  accuracy, which is consistent with the view of of Batch Normalization as a
  regularizer (Sec. 3.4): the randomization inherent in our method should be
  most beneficial when it affects an example differently each time it is seen.
* Reduce the photometric distortions. Because batchnormalized networks train
  faster and observe each training example fewer times, we let the trainer
  focus on more “real” images by distorting them less.
{quote}
"""
import itertools
import os
import stat

# All combinations of values we need to try
# This is the complete list - uncomment for final tests
# hidden_layers = ["1", "2", "3", "4"]
# units_per_layer = ["512", "1024", "2048"]
# epochs = ["2", "5", "10"]
# # 60 is the batch size used in the paper ("with 60 examples per mini-batch")
# batch_size = ["60", "128", "256"]
# optimizer = ["sgd", "rmsprop"]
# learning_rate = ["0.1", "0.01", "0.001"]

# This is a simplified list
hidden_layers = ["1", "2"]
units_per_layer = ["512"]
epochs = ["2"]
batch_size = ["128"]
optimizer = ["sgd", "rmsprop"]
learning_rate = ["0.1", "0.01"]

all_tests = list(itertools.product(
    hidden_layers, units_per_layer, epochs, batch_size, optimizer,
    learning_rate))

args_template = ("--hidden_layers {} --units_per_layer {} --epochs {} "
                 "--batch_size {} --optimizer {} --learning_rate {}")
script_file = "batchnorm_mnist_tests.sh"
with open(script_file, "w") as f:
    f.write("#!/bin/bash\n")
    f.write("# This file was automatically generated\n\n")
    for i, test in enumerate(all_tests, start=1):
        args = args_template.format(test[0], test[1], test[2], test[3],
                                    test[4], test[5])
        f.write('echo "Testing {} of {} - {}"\n'.format(i, len(all_tests),
                                                        test))
        f.write("python3 CAP6619_term_project_mnist_dnn_batchnorm.py \\\n")
        f.write("   " + args + "\n\n")

# Make it executable (for the current user)
os.chmod(script_file, os.stat(script_file).st_mode | stat.S_IEXEC)
