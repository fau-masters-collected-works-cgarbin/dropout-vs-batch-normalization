#!/bin/bash
# Dropout paper is at http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf

# Test the configuration documented in appendix B.1 of the dropout paper
# Source code used in the paper is at http://www.cs.toronto.edu/~nitish/dropout/
# For the MNIST experiment: http://www.cs.toronto.edu/~nitish/dropout/mnist.pbtxt
# "The architectures shown in Figure 4 include all combinations of 2, 3, and 4
# layer networks with 1024 and 2048 units in each layer. Thus, there are six
# architectures in all. For all the architectures (including the ones reported
# in Table 2), we used p = 0.5 in all hidden layers and p = 0.8 in the input
# layer. A final momentum of 0.95 and weight constraints with c = 2 was used
# in all the layers."
python3 CAP6619_term_project_mnist_dnn_dropout.py --hidden_layers 2 \
  --units_per_layer 1024 --epochs 2 --batch_size 60 \
  --dropout_rate_input_layer 0.2 --dropout_rate_hidden_layer 0.5 \
  --dropout_lr_multiplier 10 --dropout_momentum 0.95 --max_norm_max_value 2
