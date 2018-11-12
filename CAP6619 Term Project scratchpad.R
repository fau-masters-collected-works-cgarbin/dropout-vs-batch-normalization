rm(list = ls())

setwd("~/fau/cap6619/termproject")

library(keras)
library(tictoc)

# Saves the network configuration used in experiments and the results
experiments <-
  data.frame(
    Description = character(), # Description of the experiment (network type, layers, etc.)
    DataSetName = character(), # Data set use for the experiment
    Loss = numeric(),
    Accuracy = numeric(),
    Epochs = numeric(), # Number of epochs used in training
    BatchSize = numeric(), # Batch size used in training
    ModelJson = character(), # Model description in JSON format (from model_to_json)
    stringsAsFactors = FALSE
  )

# 0 for no logging to stdout
# 1 for progress bar logging
# 2 for one log line per epoch <-- this may work better for the produced .html
keras_verbose <- 2

tic("Load data test and training data")
mnist <- dataset_mnist()
toc()

# Encode the labels in categories
train_labels <- to_categorical(mnist$train$y)
test_labels <- to_categorical(mnist$test$y)

# Reshape and scale the data in the format that matches the netowrk we will use
# Before: integer array of shape (60000, 28, 28) and interval [0, 255]
# After: double array of shape (60000, 28 * 28) and interval [0, 1]
tic("Baseline: reshape data")
train_images <- array_reshape(mnist$train$x, c(60000, 28 * 28))
train_images <- train_images / 255
test_images <- array_reshape(mnist$test$x, c(10000, 28 * 28))
test_images <- test_images / 255
toc()

#' Run the MNIST experiment on the given model: train and evaluate th the model.
#'
#' @param model The model to use to run the experiment
#'
#' @return A list with parameters to train the model and the evaluation results.
run_experiment_mnist <- function(model) {
  model %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )

  # To get repeatable results with random numbers
  set.seed(123)

  tic("Training")
  epochs <- 20
  batch_size <- 64
  model %>% fit( train_images, train_labels, epochs = epochs, batch_size = batch_size,
    verbose = keras_verbose
  )
  toc()

  tic("Evaluation")
  evaluation <- model %>% evaluate(test_images, test_labels)
  toc()

  list(
    epochs = epochs,
    batch_size = batch_size,
    evaluation = evaluation
  )
}


# Standard network ----------------------------------------------------------------------------------

tic("Standard network, 2 layers, 1024 neurons")
model <- keras_model_sequential() %>%
  layer_dense(units = 1024, activation = "relu", input_shape = c(28 * 28)) %>%
  layer_dense(units = 1024, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

results <- run_experiment_mnist(model)
experiments[nrow(experiments) + 1, ] <- list(
  "Standard network, 2 layers, 1024 nodes", "MNIST",
  results$evaluation$loss, results$evaluation$acc,
  results$epochs, results$batch_size, model_to_json(model))
toc()

# Dropout network ----------------------------------------------------------------------------------

# TODO: add dropout as first layer
# See https://keras.rstudio.com/reference/layer_dropout.html

tic("Dropout network, 2 layers, 1024 neurons")
dropout_rate = 0.5
model <- keras_model_sequential() %>%
  layer_dense(units = 1024, activation = "relu", input_shape = c(28 * 28)) %>%
  layer_dropout(dropout_rate) %>%
  layer_dense(units = 1024, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

results <- run_experiment_mnist(model)
experiments[nrow(experiments) + 1, ] <- list(
  "Dropout network, 2 layers, 1024 nodes", "MNIST",
  results$evaluation$loss, results$evaluation$acc,
  results$epochs, results$batch_size, model_to_json(model))
toc()

tic("Dropout network with adjusted input units, 2 layers, 1024 neurons")
dropout_rate = 0.5
model <- keras_model_sequential() %>%
  layer_dense( units = 1024 / dropout_rate, activation = "relu", input_shape = c(28 * 28)) %>%
  layer_dropout(dropout_rate) %>%
  layer_dense(units = 1024, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

results <- run_experiment_mnist(model)
experiments[nrow(experiments) + 1, ] <- list(
  "Dropout network with adjusted input units, 2 layers, 1024 nodes", "MNIST",
  results$evaluation$loss, results$evaluation$acc,
  results$epochs, results$batch_size, model_to_json(model) )
toc()

