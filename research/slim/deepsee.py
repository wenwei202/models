# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import re

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_float(
    'map_density', 1.0,
    'The fraction of map to draw.')

tf.app.flags.DEFINE_string(
    'layer_name', None, 'The name of layer to analyze seemap.')

FLAGS = tf.app.flags.FLAGS

def _add_mean(images):
  return tf.add(images, tf.constant([123.68, 116.78, 103.94]))

def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])
    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    ####################
    # Define the model #
    ####################
    logits, end_points = network_fn(images)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    # get representation with max contribution
    if FLAGS.layer_name:
      target_features = end_points[FLAGS.layer_name]
    else:
      target_features = logits
    target_shape = target_features.get_shape().as_list()
    axis2reduce = range(1, len(target_shape))

    # get map
    max_features = tf.reduce_max(target_features, axis=axis2reduce)
    themaps = tf.gradients(max_features, images)
    themaps = themaps[0]
    with tf.control_dependencies([tf.Print(themaps,[themaps],first_n=3)]):
      #themaps = tf.multiply(tf.sign(images), themaps)
      #themaps = tf.multiply(images, themaps) # Count the contribution of each pixel to classification
      themaps = tf.abs(themaps)

    # negative and positive maps contributed to classification
    where_cond = tf.less(themaps, 0.0001)
    #neg_maps = tf.where(where_cond,
    #                    tf.abs(themaps),
    #                    tf.zeros_like(themaps))
    pos_maps = tf.where(where_cond,
                       tf.zeros_like(themaps),
                       themaps)
    # scale to (0, 1)
    max_see = tf.reduce_max(pos_maps, -1, keep_dims=True)
    max_see = tf.reduce_max(max_see, -2, keep_dims=True)
    max_see = tf.reduce_max(max_see, -3, keep_dims=True)
    pos_maps = tf.divide(pos_maps, max_see)

    # rgb to gray
    channel_num = pos_maps.get_shape().as_list()[-1]
    if channel_num==3:
      pos_maps = tf.image.rgb_to_grayscale(pos_maps)

    # get threshold
    pos_frac = FLAGS.map_density # how much fraction of pos_maps used for visualization
    nonzero_frac = 1.0 - tf.nn.zero_fraction(pos_maps)
    thre = tf.contrib.distributions.percentile(pos_maps, 100 - 100*pos_frac*nonzero_frac)

    # get shown images
    #thre = 0.2
    overlay_maps = tf.where(tf.less(pos_maps, thre),
                        tf.zeros_like(pos_maps),
                        tf.ones_like(pos_maps))
    # brightest color -- Fluorescent Yellow-Green rgb(153, 255, 0)
    overlay_maps = tf.concat([tf.zeros_like(overlay_maps), overlay_maps*255, tf.zeros_like(overlay_maps)], 3)
    if FLAGS.model_name == 'lenet':
      shown_images = images*128 + 128
      shown_images = tf.image.grayscale_to_rgb(shown_images)
    elif re.match('^(inception).*', FLAGS.model_name):
      shown_images = (images + 1)*128
    else:
      shown_images = _add_mean(images)
    #shown_images = tf.where(tf.greater(tf.tile(pos_maps, [1,1,1,3]), thre),
    #                        overlay_maps,
    #                        shown_images)
    infuse_coef = tf.tile(pos_maps, [1,1,1,3])
    shown_images = (1-infuse_coef)*(shown_images*0.4)+infuse_coef*overlay_maps

    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)

    #for cnt in range(0, max_outputs):
    #  cur_predict = tf.reshape(tf.gather(predictions, [cnt]), [])
    #  cur_label = tf.reshape(tf.gather(labels, [cnt]),[])
    #  cur_max_logit = tf.reshape(tf.gather(max_logits, [cnt]),[])
    #  cur_map = tf.gather(themaps, [cnt])
    #  tf.summary.scalar('prediction/%d'%(cnt),cur_predict)
    #  tf.summary.scalar('label/%d' % (cnt), cur_label)
    #  tf.summary.scalar('correct/%d' % (cnt), tf.cast(tf.equal(cur_predict, cur_label), tf.int32))
    #  tf.summary.scalar('max_logit/%d' % (cnt), cur_max_logit )
    #  tf.summary.scalar('bias_rate/%d' % (cnt), (cur_max_logit-tf.reduce_sum(cur_map))/cur_max_logit)
    with tf.control_dependencies([tf.Print(themaps, [predictions], first_n=3, summarize=FLAGS.batch_size)]):
      with tf.control_dependencies([tf.Print(themaps, [labels], first_n=3, summarize=FLAGS.batch_size)]):
        themaps = tf.identity(themaps)

    # add to summary
    max_outputs = FLAGS.batch_size
    tf.summary.image('images', shown_images, max_outputs=max_outputs)
    tf.summary.image('themaps', themaps, max_outputs=max_outputs)
    tf.summary.image('pos_maps', pos_maps, max_outputs=max_outputs)
    # tf.summary.image('neg_maps', neg_maps, max_outputs=max_outputs)

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'Recall_5': slim.metrics.streaming_recall_at_k(
            logits, labels, 5),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.items():
      summary_name = 'eval/%s' % name
      op = tf.summary.scalar(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    slim.evaluation.evaluate_once(
        master=FLAGS.master,
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=list(names_to_updates.values()),
        variables_to_restore=variables_to_restore)


if __name__ == '__main__':
  tf.app.run()
