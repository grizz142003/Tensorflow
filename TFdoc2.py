import matplotlib.pyplot as plt
import os 
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

#url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'

#dataset = tf.keras.utils.get_file("SOF", url, untar=True, cache_dir='.',cache_subdir='')
datadir = 'SOF'
print(os.listdir(datadir))
traindir = 'SOF/train'
print(os.listdir(traindir))

batch_size = 32
seed = 42

training = tf.keras.utils.text_dataset_from_directory(
    'SOF/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)
validation = tf.keras.utils.text_dataset_from_directory(
    'SOF/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)
test = tf.keras.utils.text_dataset_from_directory(
    'SOF/test',
    batch_size=batch_size)

def Txt2Num(stringData):
    lowercase = tf.strings.lower(stringData)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                   '[%s]' % re.escape(string.punctuation),
                                   '')
max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=Txt2Num,
    max_tokens=max_features,
    output_mode = 'int',
    output_sequence_length=sequence_length)

train = training.map(lambda x, y: x)
vectorize_layer.adapt(train)

def vectorize_text(text,label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

trainds = training.map(vectorize_text)
valds = validation.map(vectorize_text)
testds = test.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE
trainds = trainds.cache().prefetch(buffer_size=AUTOTUNE)
valds = valds.cache().prefetch(buffer_size=AUTOTUNE)
testds = testds.cache().prefetch(buffer_size=AUTOTUNE)

embedding_dim = 16
model = tf.keras.Sequential([
    layers.Embedding(max_features, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(4, activation='sigmoid')])

model.summary()
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer='adam',
              metrics=['accuracy'])

epochs = 19

history = model.fit(
    trainds,
    validation_data=valds,
    epochs=epochs
)

loss, accuracy =  model.evaluate(testds)

print(loss, accuracy)
