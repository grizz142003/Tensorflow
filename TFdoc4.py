import numpy as np
import os

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print(hub.__version__)

trainData, valData, testData, = tfds.load(name="imdb_reviews",
                                          split=('train[:60%]', 'train[60%:]', 'test'),
                                          as_supervised=True)

#examples used to test text vectorization from tfhub
examples, labelEx = next(iter(trainData.batch(10)))

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
embed = hub.load(embedding)

def embedtxt(text):
    return embed(tf.reshape(tf.cast(text,tf.string),[-1]))

#testing tf text vectorization
#Comment above needs to be removed for this to work 

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Lambda(embedtxt, input_shape=[]))
model.add(tf.keras.layers.Dense(16,activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(trainData.shuffle(10000).batch(512),
                    epochs=10,
                    validation_data=valData.batch(512),
                    verbose=1)

results = model.evaluate(testData.batch(512), verbose=2)

for name, value in zip(model.metrics_name, results):
    print("%s: %.3f" % (name, value))
