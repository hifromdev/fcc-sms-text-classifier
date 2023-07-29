import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print(tf.__version__)

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 5])
  plt.xlabel('Epoch')
  plt.ylabel('Error [expenses]')
  plt.legend()
  plt.grid(True)

VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 250

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

train_dataset = pd.read_csv(train_file_path, sep="\t", names=['labels', 'text'], header=None)
test_dataset = pd.read_csv(test_file_path, sep="\t", names=['labels', 'text'], header=None)

# PREPROCESSING

vectorize_text = tf.keras.layers.TextVectorization(
  max_tokens=VOCAB_SIZE, 
  output_mode='int'
)

vectorize_text.adapt(train_dataset['text'])

train_labels = np.asarray(train_dataset['labels'].map({'ham': 0, 'spam': 1}), dtype=np.float32)
train_text = np.asarray(sequence.pad_sequences(train_dataset['text'].map(vectorize_text), MAX_SEQUENCE_LENGTH), dtype=np.float32)

test_labels = np.asarray(test_dataset['labels'].map({'ham': 0, 'spam': 1}), dtype=np.float32)
test_text = np.asarray(sequence.pad_sequences(test_dataset['text'].map(vectorize_text), MAX_SEQUENCE_LENGTH), dtype=np.float32)

# MODEL

model = tf.keras.Sequential([
  tf.keras.layers.Embedding(VOCAB_SIZE, 64, input_length=MAX_SEQUENCE_LENGTH),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
  tf.keras.layers.Dense(32, activation='sigmoid'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation='relu')
])

model.compile(
  loss=tf.keras.losses.BinaryCrossentropy(),
  optimizer='rmsprop',
  metrics=['accuracy']
)

model.summary()

history = model.fit(
  train_text, train_labels,
  epochs=10,
  validation_split=0.2
)

print(model.evaluate(test_text, test_labels))

# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
def predict_message(pred_text):

  new_pred_text = np.asarray(sequence.pad_sequences(pd.Series(data=[pred_text]).map(vectorize_text), MAX_SEQUENCE_LENGTH), dtype=np.float32)
  prediction = model.predict(new_pred_text)
  
  return [prediction.item(), "spam"] if prediction > 0.5 else [prediction.item(), "ham"]

pred_text = "how are you doing today?"

prediction = predict_message(pred_text)
print(prediction)