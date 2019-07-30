# Copyright 2017 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#!/usr/bin/env python

import os
import errno
import time

import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np

# from tensorflow import keras
from tiny_imagenet import load_tiny_imagenet_data

flags = tf.app.flags

flags.DEFINE_integer("num_step", "1000",
                    "number of global steps(i.e. batches) to perform")

flags.DEFINE_integer("batch_size", "32",
                    "training batch size")
flags.DEFINE_string("data_dir", "/opt/ai/input/data",
                    "data dir")
FLAGS = flags.FLAGS

layers = tf.layers

tf.logging.set_verbosity(tf.logging.INFO)


def conv_model(feature, target, mode):
    """2-layer convolution model."""
    # Convert the target to a one-hot tensor of shape (batch_size, 10) and
    # with a on-value of 1 for each one-hot vector of length 10.
    target = tf.one_hot(tf.cast(target, tf.int32), 200, 1, 0)

    # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
    # image width and height final dimension being the number of color channels.
    pixel_num = 64  # shape of input images should be pixel_num x pixel num
    feature = tf.reshape(feature, [-1, pixel_num, pixel_num, 3])

    # shape of input tensor is 64x64x3
    # First conv layer will compute 64 features for each 3x3 patch TWICE -> equivalent to use 5x5 kernel
    with tf.variable_scope('conv_layer1'):
        h_conv1_1 = layers.conv2d(feature, 64, kernel_size=[3, 3],
                                  activation=tf.nn.relu, padding="SAME")

        h_conv1_2 = layers.conv2d(h_conv1_1, 64, kernel_size=[3, 3],
                                  activation=tf.nn.relu, padding="SAME")

        h_pool1 = tf.nn.max_pool(
            h_conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # shape of input tensor is 32x32x64
    # Second conv layer will compute 128 features for each 3x3 patch TWICE -> equivalent to use 5x5 kernel
    with tf.variable_scope('conv_layer2'):
        h_conv2_1 = layers.conv2d(h_pool1, 128, kernel_size=[3, 3],
                                  activation=tf.nn.relu, padding="SAME")

        h_conv2_2 = layers.conv2d(h_conv2_1, 128, kernel_size=[3, 3],
                                  activation=tf.nn.relu, padding="SAME")

        h_pool2 = tf.nn.max_pool(
            h_conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # shape of input tensor if 16x16x128
    # Third conv layer will compute 256 features for each 3x3 patch THRICE -> equivalent to use 7x7 kernel
    with tf.variable_scope('conv_layer3'):
        h_conv3_1 = layers.conv2d(h_pool2, 256, kernel_size=[3,3],
                                  activation=tf.nn.relu, padding="SAME")

        h_conv3_2 = layers.conv2d(h_conv3_1, 256, kernel_size=[3, 3],
                                  activation=tf.nn.relu, padding="SAME")

        h_conv3_3 = layers.conv2d(h_conv3_2, 256, kernel_size=[3, 3],
                                  activation=tf.nn.relu, padding="SAME")

        h_pool3 = tf.nn.max_pool(
            h_conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # shape of input tensor if 8x8x256
    # Third conv layer will compute 512 features for each 3x3 patch THRICE -> equivalent to use 7x7 kernel
    with tf.variable_scope('conv_layer4'):
        h_conv4_1 = layers.conv2d(h_pool3, 512, kernel_size=[3, 3],
                                  activation=tf.nn.relu, padding="SAME")

        h_conv4_2 = layers.conv2d(h_conv4_1, 512, kernel_size=[3, 3],
                                  activation=tf.nn.relu, padding="SAME")

        h_conv4_3 = layers.conv2d(h_conv4_2, 512, kernel_size=[3, 3],
                                  activation=tf.nn.relu, padding="SAME")

        h_pool4 = tf.nn.max_pool(
            h_conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # shape of input tensor if 4x4x512
    # Third conv layer will compute 512 features for each 3x3 patch THRICE -> equivalent to use 7x7 kernel
    with tf.variable_scope('conv_layer5'):
        h_conv5_1 = layers.conv2d(h_pool4, 512, kernel_size=[3, 3],
                                  activation=tf.nn.relu, padding="SAME")

        h_conv5_2 = layers.conv2d(h_conv5_1, 512, kernel_size=[3, 3],
                                  activation=tf.nn.relu, padding="SAME")

        h_conv5_3 = layers.conv2d(h_conv5_2, 512, kernel_size=[3, 3],
                                  activation=tf.nn.relu, padding="SAME")

        h_pool5 = tf.nn.max_pool(
            h_conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # now tensor 'h_pool_5' has shape of 2x2x512
        # reshape tensor into a batch of vectors
        h_pool5_flat = tf.reshape(h_pool5, [-1, int((pixel_num/32) * (pixel_num/32) * 512)])  # pooling 5 times, 2^5 = 32

    # Densely connected layer with 4096 neurons.
    h_fc1 = layers.dropout(
        layers.dense(h_pool5_flat, 4096, activation=tf.nn.relu),
        rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Densely connected layer with 4096 neurons.
    h_fc2 = layers.dropout(
        layers.dense(h_fc1, 4096, activation=tf.nn.relu),
        rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Compute logits (1 per class) and compute loss.
    logits = layers.dense(h_fc2, 200, activation=None)
    loss = tf.losses.softmax_cross_entropy(target, logits)

    return tf.argmax(logits, 1), loss


def main(_):
    # Horovod: initialize Horovod.
    hvd.init()

    # Load TinyImageNet dataset
    path = FLAGS.data_dir
    # filenames = ['train.tfrecord', 'test.tfrecord', 'test.tfrecord']
    # filename = tf.placeholder(dtype=tf.string)
    filename = 'train.tfrecord'

    dataset = load_tiny_imagenet_data(path, filename)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.repeat(FLAGS.num_step)
    iterator = dataset.make_one_shot_iterator()

    image, label = iterator.get_next()


    predict, loss = conv_model(image, label, tf.estimator.ModeKeys.TRAIN)

    # Horovod: adjust learning rate based on number of GPUs.
    opt = tf.train.RMSPropOptimizer(0.001 * hvd.size())

    # Horovod: add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(opt)

    global_step = tf.train.get_or_create_global_step()
    train_op = opt.minimize(loss, global_step=global_step)
    
    hooks = [
        # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
        # from rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights
        # or restored from a checkpoint.
        hvd.BroadcastGlobalVariablesHook(0),

        # Horovod: adjust number of steps based on number of GPUs.
        tf.train.StopAtStepHook(last_step=FLAGS.num_step // hvd.size()),

    
        tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': loss},
                                   every_n_iter=10),
    ]

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting them.
    # checkpoint_dir = './checkpoints' if hvd.rank() == 0 else None
    checkpoint_dir = None

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    time_begin = time.time()
    print('training begins @ %f' % time_begin)
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           hooks=hooks,
                                           config=config) as mon_sess:
        while not mon_sess.should_stop():
            # Run a training step synchronously.
            # image_, label_ = next(training_batch_generator)
            mon_sess.run(train_op)
    time_end = time.time()
    print('training ends @ %f' % time_end)
    time_elapse = time_end - time_begin
    print('training elapsed time : %f s' % time_elapse)

if __name__ == "__main__":
    tf.app.run()
