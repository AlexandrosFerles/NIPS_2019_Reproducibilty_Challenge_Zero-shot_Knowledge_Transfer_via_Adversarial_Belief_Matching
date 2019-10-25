import collections
import json
import numpy as np
from tensorflow import keras
import tensorflow as tf
# Borrowed from https://github.com/ozan-oktay/Attention-Gated-Networks
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import LearningRateSchedule


def json_file_to_pyobj(filename):
    def _json_object_hook(d): return collections.namedtuple('X', d.keys())(*d.values())

    def json2obj(data): return json.loads(data, object_hook=_json_object_hook)

    return json2obj(open(filename).read())


def pad(x: tf.Tensor) -> tf.Tensor:
    return tf.pad(x, ((4, 4), (4, 4), (0, 0)), mode='REFLECT')


def crop(x: tf.Tensor) -> tf.Tensor:
    return tf.image.random_crop(x, (32, 32, 3))


def flip_left_right(x: tf.Tensor) -> tf.Tensor:
    return tf.image.random_flip_left_right(x)


def flip_upside_down(x: tf.Tensor) -> tf.Tensor:
    return tf.image.random_flip_up_down(x)


def pad1(image):
    return np.pad(image, pad_width=((4, 4), (4, 4), (0, 0)), mode='reflect')


def crop1(image, size):
    return tf.image.random_crop(image, size)


def flip_left_right1(image, seed=42):
    return tf.image.random_flip_left_right(image, seed)


def flip_upside_down1(image, seed=42):
    return tf.image.random_flip_up_down(image, seed)


def apply_transformations(x_train, x_test, seed=42):
    # train transformation

    for index, image in enumerate(x_train):

        if index % 5000 == 0:
            print('preprocessed ', index)

        im = pad1(image)
        im = crop1(im, (32, 32, 3))
        im = flip_upside_down1(im, seed)
        # im = flip_left_right(image, seed)

        x_train[index, :, :, :] = im

    x_train = x_train / 255
    x_test = x_test / 255

    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - 0.4377) / 0.1980
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - 0.4438) / 0.2010
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - 0.4728) / 0.1970

    # test transformation
    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - 0.4377) / 0.1980
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - 0.4438) / 0.2010
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - 0.4728) / 0.1970

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    return x_train, x_test


def cifar10loaders(train_batch_size=128, test_batch_size=10, seed=42):
    # Tuple of Numpy arrays
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train, x_test = apply_transformations(x_train, x_test, seed)

    # x_train = x_train[0:100,:,:,:]
    # y_train = y_train[0:100,:]
    #
    # x_test = x_test[0:100, :, :, :]
    # y_test = y_test[0:100, :]

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(train_batch_size)
    # train_ds = train_ds.map(lambda x, y: (tf.div(tf.cast(x, tf.float32), 255.0), y))
    # train_ds = train_ds.map(lambda x, y: (pad(x), y))
    # train_ds = train_ds.map(lambda x, y: (crop(x,), y))
    # train_ds = train_ds.map(lambda x, y: (flip_upside_down(x), y))
    # clip values inside [0,1] if outside
    # train_ds = train_ds.map(lambda x, y: (tf.clip_by_value(x, 0, 1), y))
    #     train_ds = train_ds.batch(train_batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(test_batch_size)
    # test_ds = test_ds.map(lambda x, y: (tf.div(tf.cast(x, tf.float32), 255.0), y))

    return train_ds, test_ds


class MultiStepLearningRateScheduler(LearningRateSchedule):
    def __init__(self, initial_learning_rate):
        super(MultiStepLearningRateScheduler, self).__init__()
        self.learning_rate = tf.cast(initial_learning_rate, tf.float32)

    @tf.function
    def __call__(self, step):

        if tf.math.mod(step, tf.cast(60, tf.float32)) == 0 or \
                tf.math.mod(step, tf.cast(120, tf.float32)) == 0 or \
                tf.math.mod(step, tf.cast(160, tf.float32)) == 0:

            self.learning_rate = tf.math.divide(self.learning_rate, tf.cast(5, tf.float32))
        print('learning rate ', self.learning_rate)
        return self.learning_rate

    def get_config(self):
        return self.learning_rate
