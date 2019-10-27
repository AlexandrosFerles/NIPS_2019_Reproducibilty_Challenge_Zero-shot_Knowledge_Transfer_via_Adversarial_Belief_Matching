import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
import random
import numpy as np
import math


WEIGHT_DECAY = 5e-4
KERNEL_INIT = 'he_normal'

# Clear notebook
tf.keras.backend.clear_session()


# Check that GPU is used
tf.config.experimental.list_physical_devices('GPU')

class WideResBlock1(layers.Layer):

    def __init__(self, input_features, output_features, stride, subsample_input, increase_filters):
        super(WideResBlock1, self).__init__()
        self.activation = layers.ReLU()

        self.batch_norm1 = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, beta_regularizer=l2(WEIGHT_DECAY),
                                                     gamma_regularizer=l2(WEIGHT_DECAY))
        self.batch_norm2 = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, beta_regularizer=l2(WEIGHT_DECAY),
                                                     gamma_regularizer=l2(WEIGHT_DECAY))

        self.conv1 = layers.Conv2D(output_features, kernel_size=3, strides=stride, padding='same', use_bias=False,
                                   kernel_initializer=KERNEL_INIT, kernel_regularizer=l2(WEIGHT_DECAY))

        self.conv2 = layers.Conv2D(output_features, kernel_size=3, strides=1, padding='same', use_bias=False,
                                   kernel_initializer=KERNEL_INIT, kernel_regularizer=l2(WEIGHT_DECAY))

        self.subsample_input = subsample_input
        self.increase_filters = increase_filters
        if subsample_input:
            self.conv_inp = layers.Conv2D(output_features, kernel_size=1, strides=2, padding='valid', use_bias=False,
                                          kernel_initializer=KERNEL_INIT, kernel_regularizer=l2(WEIGHT_DECAY))
        elif increase_filters:
            self.conv_inp = layers.Conv2D(output_features, kernel_size=1, strides=1, padding='valid', use_bias=False,
                                          kernel_initializer=KERNEL_INIT, kernel_regularizer=l2(WEIGHT_DECAY))

    @tf.function
    def call(self, x, training=True):
        if self.subsample_input or self.increase_filters:
            x = self.batch_norm1(x, training=training)
            x = self.activation(x)
            x1 = self.conv1(x)
        else:
            x1 = self.batch_norm1(x, training=training)
            x1 = self.activation(x1)
            x1 = self.conv1(x1)
        x1 = self.batch_norm2(x1, training=training)
        x1 = self.activation(x1)
        x1 = self.conv2(x1)

        if self.subsample_input or self.increase_filters:
            return self.conv_inp(x) + x1
        else:
            return x + x1


class WideResBlock(layers.Layer):

    def __init__(self, input_features, output_features, stride):
        super(WideResBlock, self).__init__()
        self.activation = layers.ReLU()

        self.batch_norm1 = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, beta_regularizer=l2(WEIGHT_DECAY),
                                                     gamma_regularizer=l2(WEIGHT_DECAY))
        self.batch_norm2 = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, beta_regularizer=l2(WEIGHT_DECAY),
                                                     gamma_regularizer=l2(WEIGHT_DECAY))

        self.conv1 = layers.Conv2D(output_features, kernel_size=3, strides=stride, padding='same', use_bias=False,
                                   kernel_initializer=KERNEL_INIT, kernel_regularizer=l2(WEIGHT_DECAY))
        self.conv2 = layers.Conv2D(output_features, kernel_size=3, strides=stride, padding='same', use_bias=False,
                                   kernel_initializer=KERNEL_INIT, kernel_regularizer=l2(WEIGHT_DECAY))

    @tf.function
    def call(self, x, training=True):
        x1 = self.batch_norm1(x, training=training)
        x1 = self.activation(x1)
        x1 = self.conv1(x1)
        x1 = self.batch_norm2(x1, training=training)
        x1 = self.activation(x1)
        x1 = self.conv2(x1)

        return x1 + x


class Nblocks(tf.keras.Model):

    def __init__(self, N, input_features, output_features, stride, subsample_input, increase_filters):
        super(Nblocks, self).__init__()

        # self.NblockLayers = tf.keras.Sequential()
        layers = []
        # self.NblockLayers.add(WideResBlock1(input_features, output_features, stride, subsample_input, increase_filters))
        layers.append(WideResBlock1(input_features, output_features, stride, subsample_input, increase_filters))
        for i in range(1, N):
            # self.NblockLayers.add(WideResBlock(output_features, output_features, stride=1))
            layers.append(WideResBlock(output_features, output_features, stride=1))
        self.NblockLayers = tf.keras.Sequential(layers)

    @tf.function
    def call(self, x, training=True):
        return self.NblockLayers(x, training=training)


class WideResNet(tf.keras.Model):

    def __init__(self, d, k, n_classes, output_features, strides):
        super(WideResNet, self).__init__()
        self.conv1 = layers.Conv2D(filters=output_features, kernel_size=3, strides=strides[0], padding='same',
                                   use_bias=False, kernel_initializer=KERNEL_INIT, kernel_regularizer=l2(WEIGHT_DECAY))

        filters = [16 * k, 32 * k, 64 * k]
        self.out_filters = filters[-1]
        N = (d - 4) // 6
        increase_filters = k > 1

        self.block1 = Nblocks(N, input_features=output_features, output_features=filters[0], stride=strides[1],
                              subsample_input=False, increase_filters=increase_filters)
        self.block2 = Nblocks(N, input_features=filters[0], output_features=filters[1], stride=strides[2],
                              subsample_input=True, increase_filters=True)
        self.block3 = Nblocks(N, input_features=filters[1], output_features=filters[2], stride=strides[3],
                              subsample_input=True, increase_filters=True)

        self.batch_norm = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, beta_regularizer=l2(WEIGHT_DECAY),
                                                     gamma_regularizer=l2(WEIGHT_DECAY))
        self.activation = layers.ReLU()
        self.avg_pool = layers.AveragePooling2D(pool_size=8)
        self.fc = layers.Dense(n_classes, bias_initializer='zeros', kernel_regularizer=l2(WEIGHT_DECAY), activation='softmax')

    @tf.function
    def call(self, x, training=True):
        x = self.conv1(x)
        attention1 = self.block1(x, training=training)
        attention2 = self.block2(attention1, training=training)
        attention3 = self.block3(attention2, training=training)
        out = self.batch_norm(attention3, training=training)
        out = self.activation(out)
        out = self.avg_pool(out)
        out = tf.reshape(out, (-1, self.out_filters))

        return self.fc(out), attention1, attention2, attention3


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    # For debugging
    # tf.config.experimental_run_functions_eagerly(True)

    # change d and k if you want to check a model other than WRN-40-2
    d = 40
    k = 2
    strides = [1, 1, 2, 2]
    net = WideResNet(d=d, k=k, n_classes=10, output_features=16, strides=strides)

    # verify that an output is produced
    sample_input = tf.ones([1, 32, 32, 3])
    out = net(sample_input)[0]
    print(out)

    # Summarize model
    net.summary()
