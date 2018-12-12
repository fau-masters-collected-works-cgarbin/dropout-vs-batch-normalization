"""Parameters to control the experiments.

These parameters drive the code. Modify them to test different configurations.
"""
import collections

Parameters = collections.namedtuple('Parameters', [
    # A brief description of the experiment. Will be used as part of file names
    # to prevent collisions with other experiments. Cannot contain spaces to
    # work correctly as a command line parameter.
    'experiment_name',
    # Type of network to test: 'standard': no dropout, 'dropout_no_adjustment':
    # dropout without adjusting units in each layer, 'dropout': dropout with
    # layers adjusted as recommended in the paper.
    'network',
    # Type of optimizer to use: 'sgd' or 'rmsprop'. The paper doesn't specify,
    # but it can be inferred that it's an SGD with adjusted learning rate.
    # I also tried RMSProp here because it's a popular one nowadays and the one
    # used in the Deep Learning With Python book. It results in good accuracy
    # with the default learning rate, even before dropout is applied.
    'optimizer',
    # Number of hidden layers in the network. When a dropout network is used,
    # each hidden layer will be followed by a dropout layer.
    'hidden_layers',
    # Number of units in each layer (note that dropout layers are adjusted,
    # increasing the number of units used in the network).
    'units_per_layer',
    # Number of epochs to train.
    'epochs',
    # Number of samples in each batch.
    'batch_size',
    # Dropout rate for the input layer ('For input layers, the choice depends
    # on the kind of input. For real-valued inputs (image patches or speech
    # frames), a typical value is 0.8.)' [Note: keras uses 'drop', not 'keep']
    'dropout_rate_input_layer',
    # Dropout rate for the input layer ('Typical values of p for hidden units
    # are in the range 0.5 to 0.8.)' [Note: keras uses 'drop', not 'keep' rate]
    'dropout_rate_hidden_layer',
    # Learning rate, to adjust as recommended in the dropout paper ('...
    # dropout net should typically use 10-100 times the learning rate that was
    # optimal for a standard neural net.')
    'learning_rate',
    # Weight decay (L2). The source code the paper points to has 'l2_decay'
    # set to 0.001. The default in Keras for SGD and RMSProp is 0.0.
    'decay',
    # Momentum for the SGD optimizer (not used in RMSProp), to adjust as
    # recommended in the dropout paper ('While momentum values of 0.9 are
    # common for standard nets, with dropout we found that values around 0.95
    # to 0.99 work quite a lot better.'). Set to 0.0 to not use momentum.
    'sgd_momentum',
    # Max norm max value, or 'none' to skip it. The paper recommends its usage
    # ('Although dropout alone gives significant improvements, using dropout
    # along with max-norm... Typical values of c range from 3 to 4.')
    'max_norm_max_value',
])
