"""
Generate .pbtxt label map of the Tsinghua-Tencent 100K (TT100K) dataset for use with the TensorFlow object detection
API. Adds only red circle signs in the dataset.

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

    # List of red sign classes. For each example, we save a category only if it is a red sign. (dict taken from Nicolas)
    red_sign_classes = ['p1', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p2', 'p20', 'p21',
                        'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8',
                        'p9', 'pa10', 'pa12', 'pa13', 'pa14', 'pa8', 'pax', 'pb', 'pc', 'ph1.5', 'ph2', 'ph2.1',
                        'ph2.2', 'ph2.4', 'ph2.5', 'ph2.6', 'ph2.8', 'ph2.9', 'ph3', 'ph3.2', 'ph3.3', 'ph3.5', 'ph3.8',
                        'ph4', 'ph4.2', 'ph4.3', 'ph4.4', 'ph4.5', 'ph4.8', 'ph5', 'ph5.3', 'ph5.5', 'phx', 'pl0',
                        'pl10', 'pl100', 'pl110', 'pl120', 'pl15', 'pl20', 'pl25', 'pl3', 'pl30', 'pl35', 'pl4', 'pl40',
                        'pl5', 'pl50', 'pl60', 'pl65', 'pl70', 'pl80', 'pl90', 'plx', 'pm1.5', 'pm10', 'pm13', 'pm15',
                        'pm2', 'pm2.5', 'pm20', 'pm25', 'pm30', 'pm35', 'pm40', 'pm46', 'pm5', 'pm50', 'pm55', 'pm8',
                        'pn40', 'pr10', 'pr100', 'pr20', 'pr30', 'pr40', 'pr45', 'pr50', 'pr60', 'pr70', 'pr80', 'prx',
                        'pw2', 'pw2.5', 'pw3', 'pw3.2', 'pw3.5', 'pw4', 'pw4.2', 'pw4.5', 'pwx', 'po']

    # open annotations json file
    annos_filename = os.path.join(args.data_dir, 'annotations.json')
    annos = json.loads(open(annos_filename).read())

    # Read labels from annotations json file and write them to .pbtxt
    label_map_path = args.output_file
    label_map_file = open(label_map_path, 'a')
    idx = 0
    for label in annos['types']:
        if label in red_sign_classes:
            out = 'item {\n'
            out += '  id: ' + str(idx + 1) + '\n'
            out += '  name: \'' + label + '\'\n'
            out += '}\n\n'

            label_map_file.write(out)
            idx += 1

    # Close the file
    label_map_file.close()
