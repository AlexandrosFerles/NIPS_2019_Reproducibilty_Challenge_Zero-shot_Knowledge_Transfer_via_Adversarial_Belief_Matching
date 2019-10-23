import collections
import json
import numpy as np
from tensorflow import keras
import tensorflow as tf


# Borrowed from https://github.com/ozan-oktay/Attention-Gated-Networks


def json_file_to_pyobj(filename):
    def _json_object_hook(d): return collections.namedtuple('X', d.keys())(*d.values())

    def json2obj(data): return json.loads(data, object_hook=_json_object_hook)

    return json2obj(open(filename).read())


def pad(image):
    return np.pad(image, pad_width=((4, 4), (4, 4), (0, 0)), mode='reflect')


def crop(image, size):
    return tf.image.random_crop(image, size)


def flip_left_right(image, seed=42):
    return tf.image.random_flip_left_right(image, seed)


def flip_upside_down(image, seed=42):
    return tf.image.random_flip_up_down(image, seed)


def apply_transformations(x_train, x_test, seed=42):
    # train transformation
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - 0.4377) / 0.1980
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - 0.4438) / 0.2010
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - 0.4728) / 0.1970

    # test transformation
    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - 0.4377) / 0.1980
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - 0.4438) / 0.2010
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - 0.4728) / 0.1970

    for index, image in enumerate(x_train):
        im = pad(image)
        im = crop(im, (32, 32, 3))
        im = flip_upside_down(im, seed)
        # im = flip_left_right(image, seed)

        x_train[index, :, :, :] = im

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    return x_train, x_test


def cifar10loaders(train_batch_size=128, test_batch_size=10, seed=42):
    # Tuple of Numpy arrays
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train, x_test = apply_transformations(x_train, x_test, seed)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).batch(train_batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(test_batch_size)

    return train_ds, test_ds


def adjust_learning_rate_scratch(optimizer, epoch, epoch_thresholds=[60, 120, 160]):
    if epoch == epoch_thresholds[0]:
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] / 5
    elif epoch == epoch_thresholds[1]:
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] / 5
    elif epoch == epoch_thresholds[2]:
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] / 5

    return optimizer
