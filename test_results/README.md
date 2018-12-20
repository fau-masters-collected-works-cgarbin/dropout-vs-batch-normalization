This folder contains the results from the experiments.

There are two types of files:

- Summary files (.txt): one line for each experiment, with the hyperparameters used for the experiment, the accuracy,
  training time and test time. The name of the file encodes information about the test executed.
  This is an [example](test_results/mlp/standard/rmsprop/dropout_mnist_mlp_standard_rmsprop_results.txt)
  of a summary file.
- Training/validation loss and accuracy: (.json): training and validation loss and accuracy for each epoch of the training
  process. It's the data returned by Keras' `model.fit(...)` function. The name of the file encodes the values of the
  hyparameters used for the test.
  This is an [example](test_results/mlp/standard/rmsprop/dropout_mnist_mlp_standard_rmsprop_nw=standard_opt=rmsprop_hl=002_uhl=1024_e=05_bs=0128_dri=0.10_drh=0.50_lr=0.0010_d=0.0000_m=0.0_mn=2_history.json)
  of such a file.
  
  
