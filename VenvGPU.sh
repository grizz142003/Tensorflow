#!/bin/bash

mkdir PyVenv 
cd PyVenv


python3 -m venv tf
source tf/bin/activate
pip install --upgrade pip

#if [[$(lshw -C display | grep vendor) =~ Nvidia ]]; then
#    echo 'Your computer has a GPU installed please use this webpage to install #tensorflow to work with your GPU'
#    xdg-open https://www.tensorflow.org/install/pip
#else
#    pip install tensorflow
#fi
pip install tensorflow[and-cuda]
pushd $(dirname $(python -c 'print(__import__("tensorflow").__file__)'))
ln -svf ../nvidia/*/lib/*.so* .
popd
ln -sf $(find $(dirname $(dirname $(python -c "import nvidia.cuda_nvcc;         
print(nvidia.cuda_nvcc.__file__)"))/*/bin/) -name ptxas -print -quit) $VIRTUAL_ENV/bin/ptxas

pip install matplotlib
pip install tensorflow-hub
pip install tensorflow-datasets
pip install seaborn
pip install git+https://github.com/tensorflow/docs
pip install pyyaml h5py
pip install -U keras-tuner
pip install IPython



