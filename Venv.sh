#!/bin/bash

mkdir PyVenv 
cd PyVenv


python3 -m venv tf
source tf/bin/activate
pip install --upgrade pip
pip install tensorflow
pip install matplotlib
pip install tensorflow-hub
pip install tensorflow-datasets
pip install seaborn
pip install git+https://github.com/tensorflow/docs
pip install pyyaml h5py
pip install -U keras-tuner
pip install IPython



