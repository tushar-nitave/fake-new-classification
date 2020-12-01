import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow
import tensorflow_hub as hub
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Dense, Flatten, MaxPooling1D, Input, Activation, concatenate
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Dropout, BatchNormalization, Reshape
from sklearn.utils import resample
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

print("Loading Features and Labels \n\n")

######################
## Train Set ##
######################

train = pd.read_csv("Merged_Train.csv")
print(train.shape)
train = train.sample(frac=1)

features = np.array(train.iloc[:, :1027])
labels = train.iloc[:, 1027:1028]
print(features.shape, labels.shape)

encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.transform(labels)
labels = to_categorical(labels)

train_y = np.array(labels)

######################
## Validation ##
######################

val = pd.read_csv("Merged_Val.csv")

print(val.shape)
val = val.sample(frac=1)

feature_val = val.iloc[:, :1027]


print(feature_val.shape)

val_x_f = np.array(feature_val)

labels_val = val.iloc[:, 1027:1028]

encoder = LabelEncoder()
encoder.fit(labels_val)
labels_val = encoder.transform(labels_val)
labels_val = to_categorical(labels_val)

val_y = np.array(labels_val)


print("Training Model CNN_VARUN \n\n")
model = Sequential()

model.add(Dense(units=256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer="adam", metrics=['accuracy'])

# model = load_model("CNN_VARUN")

train_x = features.reshape((features.shape[0], 1, features.shape[1]))
train_y = train_y.reshape((train_y.shape[0], 1, train_y.shape[1]))
val_x = val_x_f.reshape((val_x_f.shape[0], 1, val_x_f.shape[1]))
val_y = val_y.reshape((val_y.shape[0], 1, val_y.shape[1]))

history = model.fit(train_x, train_y, epochs=5,
                    validation_data=(val_x, val_y), batch_size=128)

print("Model Trained \n\n")

model.save("CNN_VARUN")

print("Model Saved \n\n")
