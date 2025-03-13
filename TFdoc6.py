import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling 
import tensorflow_docs.plots

from IPython import display
from matplotlib import pyplot as plt

import numpy as np

import pathlib
import shutil
import tempfile

#funtions which are normally scattered around the code moved to top for Viewing purposes
def pack_row(*row):
    label = row[0]
    features = tf.stack(row[1:],1)
    return features, label

def get_optimizer():
    return tf.keras.optimizers.Adam(LearnRate)

def  get_callbacks(name):
    return [
        tfdocs.modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200,mode='min'),
        tf.keras.callbacks.TensorBoard(logdir/name),
    ]
def compile_fit(model, name, optimizer=None, max_epochs=10000):
    if optimizer is None:
        optimizer = get_optimizer()
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[
                      tf.keras.metrics.BinaryCrossentropy(
                          from_logits=True, name='binary_crossentropy'),
                          'accuracy'])
    model.summary()

    history = model.fit(
        train_ds.map(lambda x, y: (x, tf.expand_dims(y, axis=-1))),
        steps_per_epoch = STEPS_PER_EPOCH,
        epochs = max_epochs,
        validation_data = validate_ds.map(lambda x, y: (x, tf.expand_dims(y, axis=-1))),
        callbacks = get_callbacks(name),
        verbose = 0
    )
    return history

logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs" 
shutil.rmtree(logdir, ignore_errors=True)

gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')
FEATURES = 28

ds = tf.data.experimental.CsvDataset(gz, [float(),]*(FEATURES+1), compression_type="GZIP")

packed_ds = ds.batch(10000).map(pack_row).unbatch()

for features,label in packed_ds.batch(1000).take(1):
    print(features[0])
    plt.hist(features.numpy().flatten(), bins = 101)
    #you need to add plt.show() on your own, in google collab its done in the background
    #plt.show()

N_VAL = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

validate_ds = packed_ds.take(N_VAL).cache()
train_ds = packed_ds.skip(N_VAL).take(N_TRAIN).cache()

print(train_ds)
validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

LearnRate = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps = STEPS_PER_EPOCH*1000,
    decay_rate = 1,
    staircase=False)

tinyModel = tf.keras.Sequential([
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(1)
])

smallModel = tf.keras.Sequential([
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(16, activation='elu'),
    layers.Dense(1)
])

mediumModel = tf.keras.Sequential([
    layers.Dense(64, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(64, activation='elu'),
    layers.Dense(64, activation='elu'),
    layers.Dense(1)
])
largeModel = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(1)
])
l2Model = tf.keras.Sequential([
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001),
                 input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)             
])
dropoutModel = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)    
])
l2DropoutModel = tf.keras.Sequential([
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001),
                 input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(1)             
])


size_histories = {}
regulizer_histories = {}
dropout_histories = {}
L2Dropout_histories = {}
#size_histories['Tiny'] = compile_fit(tinyModel, 'sizes/Tiny')
#size_histories['Small'] = compile_fit(smallModel, 'size/Small')
#size_histories['Medium'] = compile_fit(mediumModel, 'size/Medium')
#size_histories['Large'] = compile_fit(largeModel, 'size/Large')
#regulizer_histories['l2'] = compile_fit(l2Model, "Regulizers/l2")
#regulizer_histories['Dropout'] = compile_fit(dropoutModel, "Dropout/l2")
regulizer_histories['L2Dropout'] = compile_fit(dropoutModel, "L2Dropout/l2")

