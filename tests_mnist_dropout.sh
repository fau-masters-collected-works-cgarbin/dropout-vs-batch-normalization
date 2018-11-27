#!/bin/bash
# Dropout paper is at http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf

# Configuration used in Deep Learning with Python, chapter 2
# The code in the book achieves ~98% test accuracy with this configuration
# (without using dropout), with an RMSProp optimizer (this is important)
# This is a baseline test, to check that we can achieve the same accuracy
# with this test code - if not, we may have a problem in the code
# Results: also ~98% accuracy, so the code is behaving as expected
# python3 CAP6619_term_project_mnist_dnn_dropout.py --hidden_layers 1 \
#   --units_per_layer 512 --epochs 5 --batch_size 128 \
#   --dropout_rate_input_layer 0.1 --dropout_rate_hidden_layer 0.5 \
#   --dropout_lr_multiplier 10 --dropout_momentum 0.95 --max_norm_max_value 3

# Test the configuration documented in appendix B.1 of the dropout paper
# Source code used in the paper is at http://www.cs.toronto.edu/~nitish/dropout/
# For the MNIST experiment: http://www.cs.toronto.edu/~nitish/dropout/mnist.pbtxt
# "The architectures shown in Figure 4 include all combinations of 2, 3, and 4
# layer networks with 1024 and 2048 units in each layer. Thus, there are six
# architectures in all. For all the architectures (including the ones reported
# in Table 2), we used p = 0.5 in all hidden layers and p = 0.8 in the input
# layer. A final momentum of 0.95 and weight constraints with c = 2 was used
# in all the layers."
# NOTE: low number of epochs - see below for higher numbers
# Results: SGD test accuracy: ~25%, RMSProp: 95%
# python3 CAP6619_term_project_mnist_dnn_dropout.py --hidden_layers 2 \
#   --units_per_layer 1024 --epochs 2 --batch_size 60 \
#   --dropout_rate_input_layer 0.2 --dropout_rate_hidden_layer 0.5 \
#   --dropout_lr_multiplier 10 --dropout_momentum 0.95 --max_norm_max_value 2

# # Adding more epochs
# # Results: about the same as low number of epochs
# python3 CAP6619_term_project_mnist_dnn_dropout.py --hidden_layers 2 \
#   --units_per_layer 1024 --epochs 5 --batch_size 60 \
#   --dropout_rate_input_layer 0.2 --dropout_rate_hidden_layer 0.5 \
#   --dropout_lr_multiplier 10 --dropout_momentum 0.95 --max_norm_max_value 2

# Reduce dropout in the input layer
# Results: SGD test accuracy: ~10%, RMSProp: ~96%
# python3 CAP6619_term_project_mnist_dnn_dropout.py --hidden_layers 2 \
#   --units_per_layer 1024 --epochs 2 --batch_size 60 \
#   --dropout_rate_input_layer 0.1 --dropout_rate_hidden_layer 0.5 \
#   --dropout_lr_multiplier 10 --dropout_momentum 0.95 --max_norm_max_value 2

# Increase dropout in input layer again, increase batch size
# Results: SGD test accuracy: ~93%, RMSProp: ~97%
# Conclusion: batch size makes a big difference
# python3 CAP6619_term_project_mnist_dnn_dropout.py --hidden_layers 2 \
#   --units_per_layer 1024 --epochs 2 --batch_size 128 \
#   --dropout_rate_input_layer 0.2 --dropout_rate_hidden_layer 0.5 \
#   --dropout_lr_multiplier 10 --dropout_momentum 0.95 --max_norm_max_value 2

# Increase max norm max value
# Results: SGD test accuracy: ~92%, RMSProp: ~97%
# Conclusion: not much of a difference, but need to train with more epochs
# python3 CAP6619_term_project_mnist_dnn_dropout.py --hidden_layers 2 \
#   --units_per_layer 1024 --epochs 2 --batch_size 128 \
#   --dropout_rate_input_layer 0.2 --dropout_rate_hidden_layer 0.5 \
#   --dropout_lr_multiplier 10 --dropout_momentum 0.95 --max_norm_max_value 3

# Increase max norm max value, reduce input layer dropout rate
# Results: SGD test accuracy: ~93%, RMSProp: ~97%
# Conclusion: not much of a difference, but need to train with more epochs
# python3 CAP6619_term_project_mnist_dnn_dropout.py --hidden_layers 2 \
#   --units_per_layer 1024 --epochs 2 --batch_size 128 \
#   --dropout_rate_input_layer 0.1 --dropout_rate_hidden_layer 0.5 \
#   --dropout_lr_multiplier 10 --dropout_momentum 0.95 --max_norm_max_value 3
