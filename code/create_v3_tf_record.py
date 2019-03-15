# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""
Convert the Tsinghua-Tencent 100K (TT100K) dataset to TFRecord for traffic sign detection.

Author: Niranjan Thakurdesai

Example usage:
    python create_tt100k_tf_record.py --output_dir=<path-to-tfrecord-output-directory> --label_map_path=<path-to-label-map-pbtxt>
    --train_file=<path-to-training-file> --validation_file=<path-to-validation-file>
"""

import json
import os
import logging
import hashlib
import io
import pandas as pd

import PIL.Image
import tensorflow as tf
from object_detection.utils import label_map_util, dataset_util


# Command-line arguments
flags = tf.app.flags
flags.DEFINE_string('output_dir', '', 'Path to directory where output TFRecords should be saved')
flags.DEFINE_string('label_map_path', '', 'Path to label map proto')
flags.DEFINE_string('train_file', '', 'Path to training data file')
flags.DEFINE_string('validation_file', '', 'Path to validation data file')
FLAGS = flags.FLAGS


def df_to_tf_example(data, image_subdirectory):
    """
    Convert Python dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
    data: dict holding annotations and bounding boxes of objects in a single image
    image_subdirectory: String specifying subdirectory within the TT100K dataset directory holding the actual image data

    Returns:
    example: The converted tf.Example.

    Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """

    img_path = os.path.join(image_subdirectory, data['filename'])
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width, height = image.size

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    for obj in data['object']:
        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)
        class_name = obj['class_name']
        classes_text.append(class_name.encode('utf8'))
        classes.append(obj['class'])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return example


def create_tf_record(filename, label_map_dict, output_filename):
    """Creates a TFRecord file from examples.
    Args:
      filename: Text file with image filenames and object annotations
      label_map_dict: The label map dictionary (.pbtxt file)
      output_filename: Path to where output file is saved
    """
    # Read the input txt file
    df = pd.read_csv(filename, sep=",", names = ["image_filename", "xmin", "ymin", "xmax", "ymax", "class"])
    
    writer = tf.python_io.TFRecordWriter(output_filename)

    # Iterate through examples
    count = 0
    for row in df.itertuples(index=True, name='Pandas'):
        count += 1
        if count % 500 == 0:
            logging.info('On image %d of %d', count, len(df.index))

        with tf.gfile.GFile(row.image_filename, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        if image.format != 'JPEG':
            raise ValueError('Image format not JPEG')
        key = hashlib.sha256(encoded_jpg).hexdigest()

        width, height = image.size

        xmin = [float(row.xmin) / width]
        ymin = [float(row.ymin) / height]
        xmax = [float(row.xmax) / width]
        ymax = [float(row.ymax) / height]
        classes = [1]
        class_name = 'red'
        classes_text = [class_name.encode('utf8')]

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(
                (df.image_filename).encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(
                (df.image_filename).encode('utf8')),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))

        writer.write(tf_example.SerializeToString())

    writer.close()


def read_image_filenames(image_dir, filenames):
    """
    Stores filenames of jpg images from given directory in a list
    :param image_dir: path to directory containing images
    :param filenames: list storing filenames
    """
    for file in os.listdir(image_dir):
        if file.endswith(".jpg"):
            filenames.append(file)


def main(_):
    logging.basicConfig(level=logging.INFO)  # print logs

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    train_output_path = os.path.join(FLAGS.output_dir, 'v4_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'v4_val.record')

    # Create TF record files for training and test data separately
    logging.info('Reading training images')
    create_tf_record(FLAGS.train_file, label_map_dict, train_output_path)

    logging.info('Reading test images')
    create_tf_record(FLAGS.validation_file, label_map_dict, val_output_path)


if __name__ == '__main__':
    tf.app.run()
