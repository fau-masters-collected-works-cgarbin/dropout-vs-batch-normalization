# Notes on upgrading to TensorFlow 2.0

Resources:

- [TensorFlow migration guide](https://www.tensorflow.org/guide/migrate)
- [TensorFlow 2.0 official CNN image classification tutorial](https://www.tensorflow.org/tutorials/images/cnn), showing how to
  use the APIs.
- [Another image classification tutorial](https://lambdalabs.com/blog/tensorflow-2-0-tutorial-01-image-classification-basics)
- [A good explanation of what changed for Keras in TensorFlow 2.0](https://www.pyimagesearch.com/2019/10/21/keras-vs-tf-keras-whats-the-difference-in-tensorflow-2-0/)

## Upgrading a simple case

To understand what is needed to upgrade to TF 2.0, I first tried to upgrade the Keras CNN reference
code in ./reference_implementations/keras_sample_code/keras_cifar_10.py. This is self-contained
piece of code, a good starting point.

### First attempt - migration script

First I tried the [TensorFlow upgrade tool](https://www.tensorflow.org/guide/upgrade).

```bash
cd reference_implementations
cd keras_sample_code
tf_upgrade_v2 --infile keras_cifar_10.py --inplace
```

The result was this:

```text
TensorFlow 2.0 Upgrade Script
-----------------------------
Converted 1 files
Detected 1 issues that require attention
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
File: keras_cifar_10.py
--------------------------------------------------------------------------------
keras_cifar_10.py:139:0: WARNING: *.save requires manual check. (This warning is only applicable
if the code saves a tf.Keras model) Keras model.save now saves to the Tensorflow SavedModel format
by default, instead of HDF5. To continue saving to HDF5, add the argument save_format='h5' to the
save() function.
```

But it didn't change any of the code. Specifying an output file with `tf_upgrade_v2 --infile keras_cifar_10.py --outfile keras_cifar_10_tf2.py`
had the same result (same warning as above, no code changes).

I was expecting that code like this:

```python
import keras
...
from keras.models import Sequential
...
model = Sequential()
```

Would change to something like this:

```python
import tensorflow as tf
...
from tf.keras.models import Sequential
...
model = Sequential()
```

Perhaps the migration script deals with the low-level TensorFlow 1.x APIs, not with code that
already uses Keras?

### Second attempt - change code by hand

After the failed upgrade attempt (above), I decided to change the code by hand.

The major TensorFlow 2.0 change that affects this project is the fact that the TensorFlow Keras
(`tf.keras`) now in sync with the latest standalone Keras. We no longer need to import Keras as a
separate package. [This post](https://www.pyimagesearch.com/2019/10/21/keras-vs-tf-keras-whats-the-difference-in-tensorflow-2-0/)
has a good explanation of what has changed and what it means for the standalone Keras (summary: stop
using it).

Code that looked like this:

```python
import keras
...
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
...
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=...
```

Now, according to the [image CNN example](https://www.tensorflow.org/tutorials/images/cnn), should
look like this:

```python
import tensorflow as tf

from tensorflow.keras import layers, models  # <--- Keras is now in Tensorflow
...
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=...
```

Some of the differences between these two pieces of code are cosmetic, e.g. importing all layers,
instead of importing the actual layer by name.

```python
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPooling2D
...
# No change needed from this point on
```

That almost worked. A few things also changed within Keras. For example, some optimizer name change:
`keras.optimizers.rmsprop` is now `tf.keras.optimizers.RMSprop`.

This last change was enough to make the code run.

Running the code resulted in two warnings:

```bash
/Users/cgarbin/fau/cap6619-deep-learning-term-project/env/lib/python3.7/site-packages/tensorflow/python/keras/engine/training.py:1844: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.
  warnings.warn('`Model.fit_generator` is deprecated and '
2021-01-03 17:08:33.361043: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
```

The first warning is fixed by changing `model.fit_generator(...)` to `model.fit(...)`. The second,
about [MILR](https://blog.tensorflow.org/2019/04/mlir-new-intermediate-representation.html), is
benign according to this [Stack Overflow exchange](https://stackoverflow.com/q/63886762/336802).
It doesn't cite a source, but, as a comment points out, the `I` indicates it an informational
message, so I'll assume it is not important for now.

After fixing the warning, the model was trained for ten epochs as quick test. These are the results:

```text
Epoch 1/10
1563/1563 [==============================] - 130s 83ms/step - loss: 2.0301 - accuracy: 0.2472 - val_loss: 1.6036 - val_accuracy: 0.4226
Epoch 2/10
1563/1563 [==============================] - 132s 84ms/step - loss: 1.6377 - accuracy: 0.4046 - val_loss: 1.4037 - val_accuracy: 0.4954
Epoch 3/10
1563/1563 [==============================] - 125s 80ms/step - loss: 1.5093 - accuracy: 0.4554 - val_loss: 1.3394 - val_accuracy: 0.5108
Epoch 4/10
1563/1563 [==============================] - 122s 78ms/step - loss: 1.4310 - accuracy: 0.4831 - val_loss: 1.2952 - val_accuracy: 0.5282
Epoch 5/10
1563/1563 [==============================] - 121s 77ms/step - loss: 1.3618 - accuracy: 0.5130 - val_loss: 1.2847 - val_accuracy: 0.5510
Epoch 6/10
1563/1563 [==============================] - 116s 74ms/step - loss: 1.3000 - accuracy: 0.5363 - val_loss: 1.1381 - val_accuracy: 0.5964
Epoch 7/10
1563/1563 [==============================] - 122s 78ms/step - loss: 1.2365 - accuracy: 0.5632 - val_loss: 1.0442 - val_accuracy: 0.6304
Epoch 8/10
1563/1563 [==============================] - 122s 78ms/step - loss: 1.1876 - accuracy: 0.5779 - val_loss: 1.0072 - val_accuracy: 0.6470
Epoch 9/10
1563/1563 [==============================] - 124s 79ms/step - loss: 1.1532 - accuracy: 0.5927 - val_loss: 0.9692 - val_accuracy: 0.6606
Epoch 10/10
1563/1563 [==============================] - 122s 78ms/step - loss: 1.1083 - accuracy: 0.6087 - val_loss: 0.9298 - val_accuracy: 0.6728
313/313 [==============================] - 4s 13ms/step - loss: 0.9298 - accuracy: 0.6728
Test loss: 0.9298253059387207
Test accuracy: 0.6728000044822693
```

For comparison, these are the results using TensorFlow 1.0 (the code before these modifications):

```text
Epoch 1/10
1563/1563 [==============================] - 120s 77ms/step - loss: 1.8583 - acc: 0.3171 - val_loss: 1.5361 - val_acc: 0.4552
Epoch 2/10
1563/1563 [==============================] - 120s 77ms/step - loss: 1.5630 - acc: 0.4288 - val_loss: 1.3759 - val_acc: 0.5114
Epoch 3/10
1563/1563 [==============================] - 115s 74ms/step - loss: 1.4551 - acc: 0.4738 - val_loss: 1.2691 - val_acc: 0.5380
Epoch 4/10
1563/1563 [==============================] - 133s 85ms/step - loss: 1.3732 - acc: 0.5082 - val_loss: 1.2035 - val_acc: 0.5677
Epoch 5/10
1563/1563 [==============================] - 145s 93ms/step - loss: 1.3077 - acc: 0.5343 - val_loss: 1.2077 - val_acc: 0.5671
Epoch 6/10
1563/1563 [==============================] - 152s 97ms/step - loss: 1.2445 - acc: 0.5584 - val_loss: 1.1648 - val_acc: 0.5823
Epoch 7/10
1563/1563 [==============================] - 139s 89ms/step - loss: 1.1985 - acc: 0.5755 - val_loss: 1.0348 - val_acc: 0.6357
Epoch 8/10
1563/1563 [==============================] - 134s 86ms/step - loss: 1.1601 - acc: 0.5904 - val_loss: 1.0441 - val_acc: 0.6323
Epoch 9/10
1563/1563 [==============================] - 138s 88ms/step - loss: 1.1268 - acc: 0.6025 - val_loss: 1.0514 - val_acc: 0.6312
Epoch 10/10
1563/1563 [==============================] - 128s 82ms/step - loss: 1.1023 - acc: 0.6085 - val_loss: 0.9951 - val_acc: 0.6577
10000/10000 [==============================] - 4s 426us/step
Test loss: 0.9951238998413086
Test accuracy: 0.6577
```

The execution time and accuracy are comparable. With that, we can confirm that the changes are
correct.

## Upgrading all files in the project

Beyond the changes described above, the following changes needed to be done to other project files:

- Model history accuracy key name changed from `val_acc` to `val_accuracy`. Example where it is
  used: `model.history.history['val_accuracy']`

## Conclusions

For code that is already using Keras, migration requires changing from `from keras import ...` to
`from tensorflow.keras import ...`, followed by some minor tweaks (for example, new names for
optimizers, mentioned above).

For code that uses the TensorFlow APIs, the
[TensorFlow upgrade tool](https://www.tensorflow.org/guide/upgrade) may be useful. It was not
useful in this case because the project was mostly written using the Keras APIs already.
