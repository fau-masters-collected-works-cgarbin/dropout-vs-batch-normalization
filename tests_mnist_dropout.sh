 #!/bin/bash
 # Dropout paper is at http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf

 # Test the configuration documented in appendix B.1 of the dropout paper
 # Source code used in the paper is at http://www.cs.toronto.edu/~nitish/dropout/
 # For the MNIST experiment: http://www.cs.toronto.edu/~nitish/dropout/mnist.pbtxt
 python3 CAP6619_term_project_mnist_dnn_dropout.py --hidden_layers=2 --units_per_layer=800 \
    --epochs=5, --batch_size=60, --dropout_rate_input_layer=0.2, --dropout_rate_hidden_layer=0.5 \
    --dropout_lr_multiplier=10 --dropout_momentum=0.95 --max_norm_max_value=5
