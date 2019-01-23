#!/bin/sh
sudo apt-get update  # To get the latest package lists
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
pip install --user Cython
pip install --user contextlib2
pip install --user jupyter
pip install --user matplotlib
pip install --user absl-py
