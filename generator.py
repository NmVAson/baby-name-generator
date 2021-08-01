import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

LABEL_COLUMN = 'target'
LABELS = [0, 1]
DATASET_SIZE = 2390968

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

def get_dataset(file_path, **kwargs):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=5, # Artificially small to make examples easier to show.
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True, 
      **kwargs)
  return dataset

full_dataset = get_dataset("training.csv")

train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)
val_dataset = test_dataset.skip(val_size)
test_dataset = test_dataset.take(test_size)

train_examples = []
test_examples = []
for feature_batch, label_batch in train_dataset.take(1):
  train_examples = feature_batch['word'].numpy()
  train_labels = label_batch.numpy()

for feature_batch, label_batch in test_dataset.take(1):
  test_examples = feature_batch['word'].numpy()
  test_labels = label_batch.numpy()

print("Training entries: {}, test entries: {}".format(len(train_examples), len(test_examples)))
print(train_examples[:5])
print(train_labels[:5])

model = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(model, input_shape=[], dtype=tf.string, trainable=True)
hub_layer(train_examples[:3])

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))
print(model.summary())

model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])

model.fit(train_dataset, epochs=10, batch_size=512, validation_data=val_dataset, verbose=1)

test_loss, test_accuracy = model.evaluate(test_dataset)

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))