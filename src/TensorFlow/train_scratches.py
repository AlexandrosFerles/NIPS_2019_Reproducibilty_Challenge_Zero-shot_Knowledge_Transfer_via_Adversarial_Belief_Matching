import numpy as np
import random

from keras_preprocessing.image import ImageDataGenerator

from tf_utils import CustomLearningRateScheduler
from tf_utils import json_file_to_pyobj
from WideResNetTF import WideResNet
from WideResNetTF import WideResNet
import tensorflow as tf


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_step(images, labels, model, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images)[0]
        loss = tf.reduce_mean(loss_object(labels, predictions))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # loss_fn = lambda: loss_object(labels, model(images)[0])
    # var_list_fn = lambda: model.trainable_weights
    # optimizer.minimize(loss_fn, var_list_fn)

    train_loss(loss)
    print('train loss ', train_loss.result())
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels, model, test_loss, test_accuracy):
    predictions = model(images, training=False)[0]
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


def loss_object(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=False)


def learning_rate_schedule(current_lr, epoch):
    if epoch == 60 or epoch == 120 or epoch == 160:
        return current_lr / 5
    else:
        return current_lr


def _train_seed(model, loaders, log=False, checkpoint=False, logfile='', checkpointFile=''):
    train_ds, test_ds = loaders
    epochs = 200

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    best_test_set_accuracy = 0

    # checkpoint every 1 iteration
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, checkpointFile, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    for epoch in range(epochs):

        # update learning rate with scheduler
        new_lr = learning_rate_schedule(current_lr=optimizer.get_config()['learning_rate'], epoch=epoch)
        optimizer.learning_rate = new_lr

        print('lr for epoch ', epoch, ' is : ', optimizer.learning_rate)
        for images, labels in train_ds:
            train_step(images, labels, model, optimizer, train_loss, train_accuracy)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels, model, test_loss, test_accuracy)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))

        ckpt.step.assign_add(1)
        epoch_accuracy = test_accuracy.result() * 100
        if log:
            with open(logfile, 'a') as temp:
                temp.write('Accuracy at epoch {} is {}%\n'.format(epoch + 1, epoch_accuracy))

        if epoch_accuracy > best_test_set_accuracy:
            best_test_set_accuracy = epoch_accuracy
            # if checkpoint:
            #     save_path = manager.save()
            #     print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

    return best_test_set_accuracy


def _train_seed_amateur(model, loaders, log=False, checkpoint=False, logfile='', checkpointFile=''):
    (x_train, y_train), (x_test, y_test) = loaders

    lr_decay = CustomLearningRateScheduler(initial_lr=0.1)

    callbacks = [lr_decay]
    # callbacks = []

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)

    model.build(input_shape=(None,) + x_train[0].shape)
    model.summary()

    model.compile(optimizer=optimizer, loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[tf.keras.metrics.sparse_categorical_accuracy])

    image_gen = ImageDataGenerator(featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   data_format='channels_last')
    image_gen.fit(x_train)
    x_train = (x_train - image_gen.mean) / image_gen.std
    x_test = (x_test - image_gen.mean) / image_gen.std

    history = model.fit(x_train, y_train, epochs=200, batch_size=128, validation_split=0.1,
                        callbacks=callbacks, verbose=1)

    loss, acc = model.evaluate(x_test, y_test)

    results = history.history

    filename = 'history_epochs{4}_{0}_batchsize{1}_eta{2}_{3}'.format(args.dataset,
                                                                      args.batch_size,
                                                                      str(args.learning_rate).replace('.', '_'),
                                                                      model.get_filename(),
                                                                      args.epochs)

    results['test_loss'] = loss
    results['test_acc'] = acc

    print('test loss is {} and acc is {}'.format(loss, acc))
    return acc


def train(args):
    json_options = json_file_to_pyobj(args.config)
    training_configurations = json_options.training

    wrn_depth = training_configurations.wrn_depth
    wrn_width = training_configurations.wrn_width
    dataset = training_configurations.dataset.lower()
    seeds = [int(seed) for seed in training_configurations.seeds]
    log = bool(training_configurations.checkpoint)

    if log:
        logfile = training_configurations.logfile
        with open(logfile, 'w') as temp:
            temp.write('WideResNet-{}-{} scratch in {}\n'.format(wrn_depth, wrn_width, training_configurations.dataset))
    else:
        logfile = ''

    checkpoint = bool(training_configurations.checkpoint)

    test_set_accuracies = []

    for seed in seeds:

        if dataset == 'cifar10':

            from tf_utils import cifar10loaders
            loaders = cifar10loaders(seed=seed)

        elif dataset == 'svhn':

            from utils import svhnLoaders
            loaders = svhnLoaders()
        else:
            ValueError('Datasets to choose from: CIFAR10 and SVHN')

        set_seed(seed)

        if log:
            with open(logfile, 'a') as temp:
                temp.write('------------------- SEED {} -------------------\n'.format(seed))

        strides = [1, 1, 2, 2]
        model = WideResNet(d=wrn_depth, k=wrn_width, n_classes=10, output_features=16, strides=strides)

        checkpointFile = '_wrn-{}-{}-seed-{}-{}-dict.pth'.format(wrn_depth, wrn_width, dataset,
                                                                 seed) if checkpoint else ''
        best_test_set_accuracy = _train_seed(model, loaders, log, checkpoint, logfile, checkpointFile)
        # best_test_set_accuracy = _train_seed_amateur(model, loaders, log, checkpoint, logfile, checkpointFile)

        if log:
            with open(logfile, 'a') as temp:
                temp.write('Best test set accuracy of seed {} is {}\n'.format(seed, best_test_set_accuracy))

        test_set_accuracies.append(best_test_set_accuracy)

        if log:
            with open(logfile, 'a') as temp:
                temp.write('Best test set accuracy of seed {} is {}\n'.format(seed, best_test_set_accuracy))

    mean_test_set_accuracy, std_test_set_accuracy = np.mean(test_set_accuracies), np.std(test_set_accuracies)

    if log:
        with open(logfile, 'a') as temp:
            temp.write(
                'Mean test set accuracy is {} with standard deviation equal to {}\n'.format(mean_test_set_accuracy,
                                                                                            std_test_set_accuracy))


if __name__ == '__main__':
    import argparse

    #tf.config.experimental_run_functions_eagerly(True)

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    parser = argparse.ArgumentParser(description='WideResNet Scratches')

    parser.add_argument('-config', '--config', help='Training Configurations', required=True)

    args = parser.parse_args()

    train(args)
