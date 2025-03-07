import matplotlib.pyplot as  plt
import numpy as np
import pandas as pd
import seaborn as sns

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement','Horsepower', 'Weight', 'Acceleratation', 'Model Year', 'Origin']

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

normalizer = tf.keras.layers.Normalization(axis=-1)

normalizer.adapt(np.array(trainFeat))

print(normalizer.mean.numpy())

first = np.array(trainFeat[:1])
first = first.astype(np.float32)

with np.printoptions(precision=2, suppress=True):
    print('First example:', first)
    print()
    print('Normalized:', normalizer(first).numpy())

HP = np.array(trainFeat['Horsepower'])
HPnormalizer = layers.Normalization(input_shape=[1,], axis=None)
HPnormalizer.adapt(HP)

HPmodel = tf.keras.Sequential([
    HPnormalizer,
    layers.Dense(units=1)
])

HPmodel.summary()
HPmodel.predict(HP[:10])

HPmodel.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error'
)

history = HPmodel.fit(
    trainFeat['Horsepower'],
    trainLBL,

    #epochs HERERERER
    epochs=40,
    validation_split = 0.2
)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plotLoss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0,10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()

plotLoss(history)

testRes = {}

testRes['HPmodel'] = HPmodel.evaluate(
    testFeat['Horsepower'],
    testLBL, verbose=0
)

x = tf.linspace(0.0, 250, 251)
y = HPmodel.predict(x)

def plot_horsepower(x,y):
    plt.scatter(trainFeat['Horsepower'], trainLBL, label='Data')
    plt.plot(x,y, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()

plot_horsepower(x,y)

linearModel = tf.keras.Sequential([
    normalizer, 
    layers.Dense(units=1)
])

linearModel.predict(trainFeat[:10])
linearModel.layers[1].kernel

linearModel.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error'
)

history = linearModel.fit(
    trainFeat,
    trainLBL,
    verbose=0,
    epochs=40,
    validation_split = 0.2
)

plotLoss(history)

testRes['linearModel'] = linearModel.evaluate(
    testFeat, testLBL, verbose=0
)

def buildModel(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model

dnnHPmodel= buildModel(HPnormalizer)
dnnHPmodel.summary()

history = dnnHPmodel.fit(
    trainFeat['Horsepower'],
    trainLBL,
    validation_split=0.2,
    epochs=40
)
plotLoss(history)

x = tf.linspace(0.0, 250, 251)
y = dnnHPmodel.predict(x)

plot_horsepower(x,y)
          
testRes['dnnHPmodel'] = dnnHPmodel.evaluate(
    testFeat['Horsepower'], testLBL,
    verbose=0)
dnn_model = buildModel(normalizer)
dnn_model.summary()

history = dnn_model.fit(
    trainFeat,
    trainLBL,
    validation_split=0.2,
    epochs=40
)

plotLoss(history)

testRes['dnn_model'] = dnn_model.evaluate(testFeat, testLBL, verbose=0)

pd.DataFrame(testRes, index=['Mean absolute error [MPG]']).T


test_predictions = dnn_model.predict(testFeat).flatten()

a = plt.axes(aspect='equal')
plt.scatter(testLBL, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

dnn_model.save('dnn_model.keras')

reloaded = tf.keras.models.load_model('dnn_model.keras')

testRes['reloaded'] = reloaded.evaluate(
    testFeat, testLBL, verbose=0
)

pd.DataFrame(testRes, index=['Mean absolute error [MPG]']).T
