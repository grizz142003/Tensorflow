import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

#Uncomment next 5 lines when first running the code to initially install the dataset

#url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
#
#dataset = tf.keras.utils.get_file("aclImdb_v1", url,
#                                    untar=True, cache_dir='.',
#                                    cache_subdir='')

#auto entered directory name copying from google gives file path issues easier to set variable manually
dataset_dir = './aclImdb_v1/aclImdb'

#Uncomment to test if dataset path is set correctly
#print(os.listdir(dataset_dir))

train_dir = os.path.join(dataset_dir, 'train')
#Uncomment to test if dataset path is set correctly
#print(os.listdir(train_dir))

#reads sample file from dataset
#sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
#with open(sample_file) as f:
#  print(f.read())

#next lines of code need to be uncommented at least once before running the final version of the code
#remove_dir = os.path.join(train_dir, 'unsup')
#shutil.rmtree(remove_dir)

batch_size = 32
seed = 42

#next batch of codes path needs to be edited
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    './aclImdb_v1/aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

#test the training batch data make sure you can open it, Note the HTML tags left over from the website
#for text_batch, label_batch in raw_train_ds.take(1):
#  for i in range(3):
#    print("Review", text_batch.numpy()[i])
#    print("Label", label_batch.numpy()[i])
#print("Label 0 corresponds to", raw_train_ds.class_names[0])
#print("Label 1 corresponds to", raw_train_ds.class_names[1])

#next batch of codes path needs to be edited
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    './aclImdb_v1/aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    './aclImdb_v1/aclImdb/train',
    batch_size=batch_size)

#function for cleaning html tags and punctuation in the dataset. 
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label


#text_batch, label_batch = next(iter(raw_train_ds))
#first_review, first_label = text_batch[0], label_batch[0]
#print("Review", first_review)
#print("Label", raw_train_ds.class_names[first_label])
#print("Vectorized Review", vectorize_text(first_review, first_label))

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)          

embedding_dim = 16

model = tf.keras.Sequential([
  layers.Embedding(max_features, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1,activation='sigmoid')])

model.compile(loss=losses.BinaryCrossentropy(),
              optimizer='adam',
              metrics=[tf.metrics.BinaryAccuracy(threshold=0.5)])

epochs = 10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs)

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

history_dict = history.history
history_dict.keys()

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss' )

plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()


model.summary()