from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def model_fn(features, labels, mode, params=None):
    """in this example we use MNIST Dataset to train this model, the image shape is [28, 28]"""
    feature_columns = tf.feature_column.numeric_column(key='images', shape=[28, 28])
    net = tf.feature_column.input_layer(features, feature_columns)

    # two Dense layers with dropout
    for j in range(2):
        net = tf.layers.dropout(
            tf.layers.dense(net, 128, activation=tf.nn.relu),
            rate=0.2, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # compute logits
    logits = tf.layers.dense(net, 10, activation=None)

    # compute predictions
    predicted_classes = tf.argmax(logits, 1)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        # assemble predictions
        predictions = {
            'class_id': predicted_classes,
            'probabilities': tf.nn.softmax(logits),
            'logits': logits
        }

        # define the export_outputs for serving
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                tf.estimator.export.PredictOutput(predictions),
            'predict_images':
                tf.estimator.export.PredictOutput({'scores': tf.nn.softmax(logits)})
        }
        
        return tf.estimator.EstimatorSpec(mode, 
                                          predictions=predictions,
                                          export_outputs=export_outputs,
                                          )

    # compute loss
    onehot_labels=tf.one_hot(labels, 10, 1, 0)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # compute evaluation metrics: accuracy
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, 
                                          loss=loss, 
                                          eval_metric_ops=metrics
                                         )

    # create training op
    assert mode == tf.estimator.ModeKeys.TRAIN
    
    global_step = tf.train.get_global_step()
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=global_step)

    # define a logging tensor hook

    hooks = [
        tf.train.LoggingTensorHook(tensors={'step': global_step, 'accuracy': accuracy[0]},
                                            every_n_iter=100)
    ]

    return tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        train_op=train_op,
        training_chief_hooks=hooks,
    )


def train_input_fn(path, params=None):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = tf.cast((x_train / 255.0), tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices(({'images': x_train}, y_train))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(64)
    dataset = dataset.repeat()
    return dataset.prefetch(buffer_size=None)


def eval_input_fn(path, params=None):
    mnist = tf.keras.datasets.mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = tf.cast((x_test / 255.0), tf.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices(({'images': x_test}, y_test))
    dataset = dataset.batch(1)

    return dataset.prefetch(buffer_size=None)


def serving_input_receiver_fn():
    features = tf.placeholder(tf.float32, shape=[1, 28, 28], name='images')
    receiver_features = {'images': features}
    return tf.estimator.export.build_raw_serving_input_receiver_fn(receiver_features)

