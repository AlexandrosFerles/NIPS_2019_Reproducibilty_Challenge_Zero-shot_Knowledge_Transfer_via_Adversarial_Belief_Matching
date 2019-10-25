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

    # for index, image in enumerate(x_train):
    #
    #     if index % 5000 == 0:
    #         print('preprocessed ', index)
    #
    #     #im = pad(image)
    #     #im = crop(im, (32, 32, 3))
    #     #im = flip_upside_down(im, seed)
    #     image = flip_upside_down(image, seed)
    #     # im = flip_left_right(image, seed)
    #
    #     x_train[index, :, :, :] = image

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

    x_train = x_train / 255
    x_test = x_test / 255

    x_train, x_test = apply_transformations(x_train, x_test, seed)

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
    train_ds = train_ds.batch(train_batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(test_batch_size)
    # test_ds = test_ds.map(lambda x, y: (tf.div(tf.cast(x, tf.float32), 255.0), y))

    return train_ds, test_ds


class CustomLearningRateScheduler(keras.callbacks.Callback):
    """
    Keras Callback for the learning rate schedule.
    The decay can be either at every update (every mini-batch)
    or at every epoch.
    It follows the half-cosine decay with restart
    in https://arxiv.org/abs/1608.03983 and add decay to
    the maximum learning rate in each cycle.
    """

    def __init__(self, initial_lr):
        """
        Instantiates a Keras Callback for learning rate scheduling.

        Arguments:
            initial_lr: maximum learning rate of the first cycle.
            T_cycle: period of the cycles.
            min_lr: minimum learning rate in all the cycles.
            n_batches: number of batches in one epoch (rounded up if float).
            n_epochs: number of epochs of the training.
            update_type: either 'batch' or 'epoch'.
                Determines if the learning rate changes at every batch or at every epoch.

        Returns:
            A Keras Callback that changes the learning
            rate according to the schedule.
        """
        super(CustomLearningRateScheduler, self).__init__()
        self.all_lr = []
        self.epoch = None
        self.curr_lr = initial_lr

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if self.update_type == 'epoch':
            scheduled_lr = self.lr_schedule(t=epoch)

            tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
            # print('\nEpoch %05d: Learning rate is %6.6f.' % (epoch, scheduled_lr))
            self.all_lr.append(scheduled_lr)

    def lr_schedule(self, t):
        """
        Computes the value of the learning rate.

        Arguments:
            t: the counter of either the global batch or epoch.
            lr: lr of current epoch
        Returns:
            The learning rate for this batch or epoch.

        """

        if t == 60 or t == 120 or t == 160:
            return self.curr_lr / 5
        else:
            return self.curr_lr
