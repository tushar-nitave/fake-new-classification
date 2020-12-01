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
import tqdm as tqdm


print("Prediction \n\n")
model = Sequential()
model.add(Conv1D(filters=256, kernel_size=1, padding='same',
                 activation='relu', kernel_initializer='he_normal', input_shape=(1, 1024)))
model.add(MaxPooling1D(pool_size=1))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv1D(filters=512, kernel_size=1, padding='same',
                 activation='relu', kernel_initializer='he_normal', input_shape=(1, 1024)))
model.add(MaxPooling1D(pool_size=1))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv1D(filters=256, kernel_size=1, padding='same',
                 activation='relu', kernel_initializer='he_normal', input_shape=(1, 1024)))
model.add(MaxPooling1D(pool_size=1))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer="adam", metrics=['accuracy'])

model = load_model("CNN_OG")

mapping = {0: "agreed", 1: "disagreed", 2: "unrelated"}

processed = pd.read_csv("test_merged.csv", sep=",")
features = processed.iloc[:, :1024]
features = np.array(features)

features = features.reshape((features.shape[0], 1, features.shape[1]))

result = model.predict((features))

predicted_labels = list()

for i in range(result.shape[0]):
    predicted_labels.append(mapping[np.argmax(result[i])])

print(len(predicted_labels))


data = pd.read_csv("test.csv", sep=",")
data["label"] = pd.DataFrame(predicted_labels)
data.to_csv("predicted_CNN_OG.csv", index=False)
