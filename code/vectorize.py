import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow
import tensorflow_hub as hub
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Dense, Flatten, MaxPooling1D, Input, Activation
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Dropout, BatchNormalization, Reshape
from sklearn.utils import resample
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
from tqdm import tqdm

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

data = pd.read_csv("processed_train.csv", sep=",")
print(data.shape, data.columns)
data = data.dropna()

features1 = data["title1"]
features2 = data["title2"]

total = len(features1)/100 if int(len(features1) /
                                  100) == len(features1)/100 else int((len(features1)/100) + 1)

embedded_features1 = pd.DataFrame()
for i in tqdm(range(total)):
    chunk = features1[i*100:(i+1)*100] if i != total-1 else features1[i*100:]
    x = embed(chunk)
    x = np.array(x)
    x = pd.DataFrame(x)
    embedded_features1 = embedded_features1.append(x, ignore_index=True)

embedded_features1.to_csv("features1.csv", index=False)


total = len(features2)/100 if int(len(features2) /
                                  100) == len(features2)/100 else int((len(features2)/100) + 1)

embedded_features2 = pd.DataFrame()
for i in tqdm(range(total)):
    chunk = features2[i*100:(i+1)*100] if i != total-1 else features2[i*100:]
    x = embed(chunk)
    x = np.array(x)
    x = pd.DataFrame(x)
    embedded_features2 = embedded_features2.append(x, ignore_index=True)

embedded_features2.to_csv("features2.csv", index=False)
