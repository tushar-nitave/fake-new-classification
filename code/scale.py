import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow
import tensorflow_hub as hub
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

data = pd.read_csv("test_proc.csv", sep=",")
print(data.shape, data.columns)
features1 = data["title1"]
features2 = data["title2"]
# label = data["label"]
js = data["js"]
cs = data["cs"]
common_count = data["common_count"]

total = len(features1)/100 if int(len(features1) /
                                  100) == len(features1)/100 else int((len(features1)/100) + 1)

embedded_features1 = pd.DataFrame()
for i in tqdm(range(total)):
    chunk = features1[i*100:(i+1)*100] if i != total-1 else features1[i*100:]
    x = embed(chunk)
    x = np.array(x)
    x = pd.DataFrame(x)
    embedded_features1 = embedded_features1.append(x, ignore_index=True)


total = len(features2)/100 if int(len(features2) /
                                  100) == len(features2)/100 else int((len(features2)/100) + 1)

embedded_features2 = pd.DataFrame()
for i in tqdm(range(total)):
    chunk = features2[i*100:(i+1)*100] if i != total-1 else features2[i*100:]
    x = embed(chunk)
    x = np.array(x)
    x = pd.DataFrame(x)
    embedded_features2 = embedded_features2.append(x, ignore_index=True)


features1 = embedded_features1
features2 = embedded_features2
print("Data Shape")
print(data.shape)
print("Features Shape")
print(features1.shape, features2.shape)

processed = pd.concat([features1, features2, js, cs,
                       common_count], axis=1)

print(processed.shape)
print(processed.isnull().sum())

# c_sim = []

# for i in tqdm(range(len(features1))):
#     c_s = cosine_similarity([features1.iloc[i:i+1, :]],
#                             [features2.iloc[i:i+1, :]])
#     c_sim.append(c_s[0][0])

# processed["CS"] = pd.DataFrame(c_sim)

# df_minority_upsampled = resample(processed,
#                                  replace=True,     # sample with replacement
#                                  n_samples=175000,    # to match majority class
#                                  random_state=123)


# Add agreed label


# labels = pd.DataFrame(["unrelated"]*features1.shape[0])

# processed["label"] = labels

processed.to_csv("unrelated_proc_og.csv", index=False)
