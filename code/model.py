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

train = pd.read_csv("Merged_Train_OG.csv")
print(train.shape)
train = train.sample(frac=1)

feature1 = train.iloc[:, 0:512]
feature2 = train.iloc[:, 512:1024]

print(feature1.shape, feature2.shape)

train_x_f1 = np.array(feature1)
train_x_f2 = np.array(feature2)


labels = train.iloc[:, 1024:1025]

encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.transform(labels)
labels = to_categorical(labels)

train_y = labels

######################
## Validation ##
######################

val = pd.read_csv("Merged_Val_OG.csv")

print(val.shape)
val = val.sample(frac=1)

feature1_val = val.iloc[:, 0:512]
feature2_val = val.iloc[:, 512:1024]

print(feature1_val.shape, feature2_val.shape)

val_x_f1 = np.array(feature1_val)
val_x_f2 = np.array(feature2_val)

labels_val = val.iloc[:, 1024:1025]

encoder = LabelEncoder()
encoder.fit(labels_val)
labels_val = encoder.transform(labels_val)
labels_val = to_categorical(labels_val)

val_y = labels_val


print("Training Model TALOS Merge \n\n")

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


train_x_f1 = np.array(train_x_f1).reshape(
    (train_x_f1.shape[0], 1, train_x_f1.shape[1]))
train_x_f2 = np.array(train_x_f2).reshape(
    (train_x_f2.shape[0], 1, train_x_f2.shape[1]))
train_y = np.array(train_y).reshape((train_y.shape[0], 1, train_y.shape[1]))
val_x_f1 = np.array(val_x_f1).reshape(
    (val_x_f1.shape[0], 1, val_x_f1.shape[1]))
val_x_f2 = np.array(val_x_f2).reshape(
    (val_x_f2.shape[0], 1, val_x_f2.shape[1]))
val_y = np.array(val_y).reshape((val_y.shape[0], 1, val_y.shape[1]))


history = model.fit([train_x_f1, train_x_f2], train_y,
                    epochs=5, shuffle=True, validation_data=([val_x_f1, val_x_f2], val_y))

print("Model Trained \n\n")

model.save("TV_Model")

print("Model Saved \n\n")
