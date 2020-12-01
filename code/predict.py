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

input1 = Input(shape=(None, 512))
inner = Conv1D(128, 1, padding='same', name='conv1',
               kernel_initializer='he_normal')(input1)
inner = BatchNormalization()(inner)
inner = Dropout(0.5)(inner)
inner = Activation('relu')(inner)
inner1 = MaxPooling1D(pool_size=1, name='max1')(inner)

inner = Conv1D(128, 1, padding='same', name='conv1',
               kernel_initializer='he_normal')(inner1)
inner = BatchNormalization()(inner)
inner = Dropout(0.5)(inner)
inner = Activation('relu')(inner)
inner1 = MaxPooling1D(pool_size=1, name='max1')(inner)

inner = Conv1D(512, 1, padding='same', name='conv1',
               kernel_initializer='he_normal')(inner1)
inner = BatchNormalization()(inner)
inner = Dropout(0.5)(inner)
inner = Activation('relu')(inner)
inner1 = MaxPooling1D(pool_size=1, name='max1')(inner)

inner = Conv1D(512, 1, padding='same', name='conv1',
               kernel_initializer='he_normal')(inner1)
inner = BatchNormalization()(inner)
inner = Dropout(0.5)(inner)
inner = Activation('relu')(inner)
inner1 = MaxPooling1D(pool_size=1, name='max1')(inner)

inner = Conv1D(768, 1, padding='same', name='conv1',
               kernel_initializer='he_normal')(inner1)
inner = BatchNormalization()(inner)
inner = Dropout(0.5)(inner)
inner = Activation('relu')(inner)
inner1 = MaxPooling1D(pool_size=1, name='max1')(inner)


input2 = Input(shape=(None, 512))
inner2 = Conv1D(128, 1, padding='same', name='conv1',
                kernel_initializer='he_normal')(input2)
inner2 = BatchNormalization()(inner2)
inner2 = Dropout(0.5)(inner2)
inner2 = Activation('relu')(inner2)
inner2 = MaxPooling1D(pool_size=1, name='max1')(inner2)

inner2 = Conv1D(128, 1, padding='same', name='conv1',
                kernel_initializer='he_normal')(inner2)
inner2 = BatchNormalization()(inner2)
inner2 = Dropout(0.5)(inner2)
inner2 = Activation('relu')(inner2)
inner2 = MaxPooling1D(pool_size=1, name='max1')(inner2)

inner2 = Conv1D(512, 1, padding='same', name='conv1',
                kernel_initializer='he_normal')(inner2)
inner2 = BatchNormalization()(inner2)
inner2 = Dropout(0.5)(inner2)
inner2 = Activation('relu')(inner2)
inner2 = MaxPooling1D(pool_size=1, name='max1')(inner2)

inner2 = Conv1D(512, 1, padding='same', name='conv1',
                kernel_initializer='he_normal')(inner2)
inner2 = BatchNormalization()(inner2)
inner2 = Dropout(0.5)(inner2)
inner2 = Activation('relu')(inner2)
inner2 = MaxPooling1D(pool_size=1, name='max1')(inner2)

inner2 = Conv1D(768, 1, padding='same', name='conv1',
                kernel_initializer='he_normal')(inner2)
inner2 = BatchNormalization()(inner2)
inner2 = Dropout(0.5)(inner2)
inner2 = Activation('relu')(inner2)
inner2 = MaxPooling1D(pool_size=1, name='max1')(inner2)


x = concatenate([input1, input2])

x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)

x = Dense(3, activation='softmax')(x)

model = Model(inputs=[input1, input2], outputs=x)
model.compile(loss='binary_crossentropy',
              optimizer='adam',  metrics=['accuracy'])

model = load_model("TV_Model")

mapping = {0: "agreed", 1: "disagreed", 2: "unrelated"}

processed = pd.read_csv("test_merged.csv", sep=",")
feature1 = processed.iloc[:, 0:512]
feature2 = processed.iloc[:, 512:1024]
feature1 = np.array(feature1)
feature2 = np.array(feature2)

feature1 = feature1.reshape((feature1.shape[0], 1, feature1.shape[1]))
feature2 = feature2.reshape((feature2.shape[0], 1, feature2.shape[1]))

result = model.predict((feature1, feature2))

predicted_labels = list()

for i in range(result.shape[0]):
    predicted_labels.append(mapping[np.argmax(result[i])])

print(len(predicted_labels))


data = pd.read_csv("test.csv", sep=",")
data["label"] = pd.DataFrame(predicted_labels)
data.to_csv("predicted_TV.csv", index=False)
