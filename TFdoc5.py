import matplotlib.pyplot as  plt
import numpy as np
import pandas as pd
import seaborn as sns

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement','horsepower', 'Weight', 'Acceleratation', 'Model Year', 'Origin']

rawDS = pd.read_csv(url, names=column_names,
                     na_values='?', comment='\t',
                       sep=' ', skipinitialspace=True)

dataset = rawDS.copy()
print(dataset.tail())
print(dataset.isna().sum())

dataset = dataset.dropna()

dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
print(dataset.tail())

trainDS = dataset.sample(frac=0.8, random_state=0)
testDS = dataset.drop(trainDS.index)

sns.pairplot(trainDS[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')

trainDS.describe().transpose()

trainFeat = trainDS.copy()
testFeat = testDS.copy()

trainLBL = trainFeat.pop('MPG')
testLBL = testFeat.pop('MPG')

trainDS.describe().transpose()[['mean','std']]

normalizer = tf.keras.layers.Normalization(axis=1)

normalizer.adapt(np.array(trainFeat))

print(normalizer.mean.numpy())

first = np.array(trainFeat[:1])

with np.printoptions(precision=2, suppress=True):
    print('First example:', first)
    print()
    print('Normalized:',normalizer(first.numpy()))



