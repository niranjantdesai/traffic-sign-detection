# Traffic Sign Detection with TensorFlow Object Detection API

Instructions for training and evaluating networks from [TensorFlow object detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for sign detection using the TT100K dataset.

## AWS setup
If you are using AWS, follow the instructions [here](https://github.gatech.edu/schou33/aws_instructions#create-deep-learning-ami-for-gpu-supported-instance) to set up an EC2 instance. Copy the TT100K dataset from S3 using 
```buildoutcfg
aws s3 sync s3://<path-to-tt100k>
```
If you're using a local GPU, skip this step.

Activate the in-built Conda environment for TensorFlow using
```buildoutcfg
source activate <environment-name>
```
Refer the logs printed after you connect to the instance for the appropriate environment name.

## Install TensorFlow object detection API
Note that AWS Deep Learning AMIs come with TensorFlow but not the object detection API preinstalled.

1. Clone the TensorFlow models repo from [here](https://github.com/tensorflow/models).
    ```buildoutcfg
    mkdir tensorflow
    git clone <url>
    ```
2. Follow the instructions [here] to complete the installation. You can use [this](scripts/tf_object_detection_install.sh) script to install most of the dependencies. Make sure that you test the installation by running
    ```buildoutcfg
    python object_detection/builders/model_builder_test.py
    ```
    from ```tensorflow/models/research```. If you are getting errors, try reactivating the Conda environment.
    
## Generate label map ProtoBuf
Generate the label map .pbtxt file by running
```buildoutcfg
python code/generate_label_map.py -d <path-to-TT100K-dataset-root-directory> -o <path-to-output-pbtxt-file>
```

## Create TF record files
Create training and test .record files by running
```buildoutcfg
python create_tt100k_tf_record.py --data_dir=<path-to-TT100K-dataset-root-directory> --output_dir=<path-to-tfrecord-output-directory> --label_map_path=<path-to-label-map-pbtxt>
```
Here, ```output_dir``` is the path to the directory where you want to store the .record files which the Python script will generate.