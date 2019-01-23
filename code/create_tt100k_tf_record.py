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
    python create_tt100k_tf_record.py --data_dir=<path-to-TT100K-dataset-root-directory> --output_dir=<path-to-tfrecord-
    output-directory> --label_map_path=<path-to-label-map-pbtxt>
"""

import json
import os
import logging
import hashlib
import io

import PIL.Image
import tensorflow as tf
from object_detection.utils import label_map_util, dataset_util


# Command-line arguments
flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to the dataset')
flags.DEFINE_string('output_dir', '', 'Path to directory where output TFRecords should be saved')
flags.DEFINE_string('label_map_path', '', 'Path to label map proto')
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


def create_tf_record(output_filename, label_map_dict, annos, image_dir, examples):
    """Creates a TFRecord file from examples.

    Args:
      output_filename: Path to where output file is saved.
      label_map_dict: The label map dictionary (.pbtxt file).
      annos: Annotations dictionary.
      image_dir: Directory where image files are stored.
      examples: Examples to parse and save to tf record.
    """
    writer = tf.python_io.TFRecordWriter(output_filename)

    # Iterate through examples
    for idx, example in enumerate(examples):
        if idx % 500 == 0:
            logging.info('On image %d of %d', idx, len(examples))

        id = os.path.splitext(example)[0]   # dict id is the image filename without extension
        objects = annos['imgs'][id]['objects']

        data = {
            'filename': example,
            'object': []
        }

        # List of red sign classes. For each example, we save a category only if it is a red sign. (dict taken from
        # Nicolas)
        red_sign_classes = ['p1', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p2', 'p20',
                            'p21',
                            'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8',
                            'p9', 'pa10', 'pa12', 'pa13', 'pa14', 'pa8', 'pax', 'pb', 'pc', 'ph1.5', 'ph2', 'ph2.1',
                            'ph2.2', 'ph2.4', 'ph2.5', 'ph2.6', 'ph2.8', 'ph2.9', 'ph3', 'ph3.2', 'ph3.3', 'ph3.5',
                            'ph3.8',
                            'ph4', 'ph4.2', 'ph4.3', 'ph4.4', 'ph4.5', 'ph4.8', 'ph5', 'ph5.3', 'ph5.5', 'phx', 'pl0',
                            'pl10', 'pl100', 'pl110', 'pl120', 'pl15', 'pl20', 'pl25', 'pl3', 'pl30', 'pl35', 'pl4',
                            'pl40',
                            'pl5', 'pl50', 'pl60', 'pl65', 'pl70', 'pl80', 'pl90', 'plx', 'pm1.5', 'pm10', 'pm13',
                            'pm15',
                            'pm2', 'pm2.5', 'pm20', 'pm25', 'pm30', 'pm35', 'pm40', 'pm46', 'pm5', 'pm50', 'pm55',
                            'pm8',
                            'pn40', 'pr10', 'pr100', 'pr20', 'pr30', 'pr40', 'pr45', 'pr50', 'pr60', 'pr70', 'pr80',
                            'prx',
                            'pw2', 'pw2.5', 'pw3', 'pw3.2', 'pw3.5', 'pw4', 'pw4.2', 'pw4.5', 'pwx', 'po']

        # Iterate through all objects (i.e. signs)
        for obj in objects:
            class_name = obj['category']
            if class_name in red_sign_classes:
                class_id = label_map_dict[class_name]
                bbox = obj['bbox']
                data['object'].append({
                    'bndbox': {
                        'xmin': bbox['xmin'],
                        'ymin': bbox['ymin'],
                        'xmax': bbox['xmax'],
                        'ymax': bbox['ymax']
                    },
                    'class': class_id,
                    'class_name': class_name
                })

        # Convert dict to TF proto
        tf_example = df_to_tf_example(data, image_dir)

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

    data_dir = FLAGS.data_dir
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    logging.info('Reading from TT100K dataset.')
    image_dir = data_dir
    image_dir_train = os.path.join(image_dir, 'train')
    image_dir_test = os.path.join(image_dir, 'test')

    train_examples = []
    test_examples = []

    # Read filenames of TT100K training images
    read_image_filenames(image_dir_train, train_examples)

    # Read filenames of TT100K test images
    read_image_filenames(image_dir_test, test_examples)

    # open json file containing annotations
    annos_filename = os.path.join(data_dir, 'annotations.json')
    annos = json.loads(open(annos_filename).read())

    train_output_path = os.path.join(FLAGS.output_dir, 'tt100k_red_signs_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'tt100k_red_signs_val.record')

    # Create TF record files for training and test data separately
    logging.info('Reading training images')
    create_tf_record(train_output_path, label_map_dict, annos, image_dir_train, train_examples)

    logging.info('Reading test images')
    create_tf_record(val_output_path, label_map_dict, annos, image_dir_test, test_examples)


if __name__ == '__main__':
    tf.app.run()
