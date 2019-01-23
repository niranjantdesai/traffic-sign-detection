"""
Generate .pbtxt label map of the Tsinghua-Tencent 100K (TT100K) dataset for use with the TensorFlow object detection
API. Adds all signs in the dataset.

Author: Niranjan Thakurdesai
"""

import argparse
import json
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", help="Path to TT100K dataset root directory")
    parser.add_argument("-o", "--output_file", help="Path to output .pbtxt file")
    args = parser.parse_args()

    # open annotations json file
    annos_filename = os.path.join(args.data_dir, 'annotations.json')
    annos = json.loads(open(annos_filename).read())

    # Read labels from annotations json file and write them to .pbtxt
    label_map_path = args.output_file
    label_map_file = open(label_map_path, 'a')
    for idx, label in enumerate(annos['types']):
        out = 'item {\n'
        out += '  id: ' + str(idx + 1) + '\n'
        out += '  name: \'' + label + '\'\n'
        out += '}\n\n'

        label_map_file.write(out)

    # Close the file
    label_map_file.close()
