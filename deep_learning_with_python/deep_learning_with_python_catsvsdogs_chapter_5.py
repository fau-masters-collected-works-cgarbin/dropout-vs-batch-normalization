"""
Cat vs. Dogs based on chapter 5 of Deep Learning with Python
"""

import os
import collections


def prepare_image_dirs(source_dir, dest_base_dir):
    """Copy cats/dogs images from the Kaggle file into the directory structure
    we will use for the data generators.

    Kaggle's data set has 25, 000 images. After downloading it we will create a
    smaller data set with 1, 000 of each class (2, 000 total), a validation set
    with 500 samples of each class and a test set with 500 samples of each
    class.

    * Download Kaggle's data set from www.kaggle.com/c/dogs-vs-cats/data
    * Uncompress it
    * Uncompress the train.zip file created above - this is the source dir

    Arguments:
        source_dir {string} -- Directory where the Kaggle train data set is.
        dest_base_dir {string} -- Base destination directory.

    Returns:
        tuple -- Full path to the train, test and validation directories.
    """

    import shutil

    # Create directory structure for train, validation and test images
    os.makedirs(dest_base_dir, exist_ok=True)

    def create_dir(base_dir, new_dir):
        d = os.path.join(base_dir, new_dir)
        os.makedirs(d, exist_ok=True)
        return d
    train_dir = create_dir(dest_base_dir, "train")
    train_cats_dir = create_dir(train_dir, "cats")
    train_dogs_dir = create_dir(train_dir, "dogs")
    validation_dir = create_dir(dest_base_dir, "validation")
    validation_cats_dir = create_dir(validation_dir, "cats")
    validation_dogs_dir = create_dir(validation_dir, "dogs")
    test_dir = create_dir(dest_base_dir, "test")
    test_cats_dir = create_dir(test_dir, "cats")
    test_dogs_dir = create_dir(test_dir, "dogs")

    # Copy a subset of the data to the directories
    def copy_images(image, r, dest_dir):
        fnames = ["{}.{}.jpg".format(image, i) for i in r]
        for fname in fnames:
            src = os.path.join(source_dir, fname)
            dst = os.path.join(dest_dir, fname)
            # Check first, to avoid copying MBs of data again
            if not os.path.isfile(dst):
                shutil.copyfile(src, dst)
    copy_images("cat", range(1000), train_cats_dir)
    copy_images("cat", range(1000, 1500), validation_cats_dir)
    copy_images("cat", range(1500, 2000), test_cats_dir)
    copy_images("dog", range(1000), train_dogs_dir)
    copy_images("dog", range(1000, 1500), validation_dogs_dir)
    copy_images("dog", range(1500, 2000), test_dogs_dir)

    # Check if we got what we need
    assert len(os.listdir(train_cats_dir)) == 1000
    assert len(os.listdir(train_dogs_dir)) == 1000
    assert len(os.listdir(validation_cats_dir)) == 500
    assert len(os.listdir(validation_dogs_dir)) == 500
    assert len(os.listdir(test_cats_dir)) == 500
    assert len(os.listdir(test_dogs_dir)) == 500

    return train_dir, test_dir, validation_dir


def create_image_generators(train_dir, validation_dir, parameters):
    from keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=parameters.batch_size,
        class_mode="binary")

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=parameters.batch_size,
        class_mode="binary")

    return train_generator, validation_generator


def run_experiment(train_generator, validation_generator, parameters):
    from keras import layers
    from keras import models
    from keras import optimizers

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu",
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy",
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=["acc"])

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=parameters.steps_per_epoch,
        epochs=parameters.epochs,
        validation_data=validation_generator,
        validation_steps=p.validation_steps)

    return model, history


def plot_accuracy_loss(history):
    """Plots the accuracy and loss of a model's training and validation epochs.

    Arguments:
        history {History} -- A Keras History object returned from model.fit
    """

    import matplotlib.pyplot as plt

    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, "bo", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()

    plt.show()


# Parameters to control the experiments.
Parameters = collections.namedtuple("Parameters", [
    # Number of epochs to train.
    "epochs",
    # Number of validations steps.
    "validation_steps",
    # Number of samples in each batch.
    "batch_size",
    # Number of steps (samples) to use in each epoch (we need this because we
    # are using a data generator, which is an infinite supplier of samples, so
    # we need to limit it with this parameter).
    "steps_per_epoch",
])

p = Parameters(
    epochs=5,
    validation_steps=5,
    batch_size=20,
    steps_per_epoch=100,
)

train_dir, test_dir, validation_dir = prepare_image_dirs(
    source_dir="./deep_learning_with_python/kaggledataset/train",
    dest_base_dir="./deep_learning_with_python/catsvsdogs")

train_generator, validation_generator = create_image_generators(
    train_dir, validation_dir, p)

model, history = run_experiment(train_generator, validation_generator, p)
model.save("cats_and_dogs_small_1.h5")
plot_accuracy_loss(history)
