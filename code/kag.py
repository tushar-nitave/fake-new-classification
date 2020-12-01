#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import gensim
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from wordcloud import WordCloud
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# In[4]:


fake = pd.read_csv("train.csv")
fake.head()


# In[5]:


# Counting by Subjects
for key, count in fake.label.value_counts().iteritems():
    print(f"{key}:\t{count}")

# Getting Total Rows
print(f"Total Records:\t{fake.shape[0]}")


# In[7]:


plt.figure(figsize=(8, 5))
sns.countplot("label", data=fake)
plt.show()


fake = fake.drop(["id", "tid1", "tid2"], axis=1)


# In[17]:


data = fake
data.head()


# In[18]:


data["title1_en"] = data["title1_en"] + " " + data["title2_en"]
data = data.drop(["title2_en"], axis=1)
data.head


# In[19]:


y = data["label"].values
# Converting X to format acceptable by gensim, removing annd punctuation stopwords in the process
X = []
stop_words = set(nltk.corpus.stopwords.words("english"))
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
for par in data["title1_en"].values:
    tmp = []
    sentences = nltk.sent_tokenize(par)
    for sent in sentences:
        sent = sent.lower()
        tokens = tokenizer.tokenize(sent)
        filtered_words = [w.strip()
                          for w in tokens if w not in stop_words and len(w) > 1]
        tmp.extend(filtered_words)
    X.append(tmp)

del data


# In[27]:


# Dimension of vectors we are generating
EMBEDDING_DIM = 100

# Creating Word Vectors by Word2Vec Method (takes time...)
w2v_model = gensim.models.Word2Vec(
    sentences=X, size=EMBEDDING_DIM, window=5, min_count=1)
# vocab size
len(w2v_model.wv.vocab)


# In[23]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

X = tokenizer.texts_to_sequences(X)


# In[24]:


X[0][:10]


# In[25]:


# Lets check few word to numerical replesentation
# Mapping is preserved in dictionary -> word_index property of instance
word_index = tokenizer.word_index
for word, num in word_index.items():
    print(f"{word} -> {num}")
    if num == 10:
        break


# In[26]:


# For determining size of input...

# Making histogram for no of words in news shows that most news article are under 700 words.
# Lets keep each news small and truncate all news to 700 while tokenizing
plt.hist([len(x) for x in X], bins=500)
plt.show()

# Its heavily skewed. There are news with 5000 words? Lets truncate these outliers :)


# In[34]:


nos = np.array([len(x) for x in X])
len(nos[nos < 100])
# Out of 256441 news, 256403 have less than 700 words


# In[35]:


# Lets keep all news to 100, add padding to news with less than 100 words and truncating long ones
maxlen = 100

# Making all news of size maxlen defined above
X = pad_sequences(X, maxlen=maxlen)


# In[36]:


# all news has 700 words (in numerical form now). If they had less words, they have been padded with 0
# 0 is not associated to any word, as mapping of words started from 1
# 0 will also be used later, if unknows word is encountered in test set
len(X[0])


# In[37]:


# Adding 1 because of reserved 0 index
# Embedding Layer creates one more vector for "UNKNOWN" words, or padded words (0s). This Vector is filled with zeros.
# Thus our vocab size inceeases by 1
vocab_size = len(tokenizer.word_index) + 1


# In[38]:


# Function to create weight matrix from word2vec gensim model
def get_weight_matrix(model, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        weight_matrix[i] = model[word]
    return weight_matrix


# In[53]:


# Getting embedding vectors from word2vec and usings it as weights of non-trainable keras embedding layer
embedding_vectors = get_weight_matrix(w2v_model, word_index)


# In[54]:

# Defining Neural Network
model = Sequential()
# Non-trainable embeddidng layer
model.add(Embedding(vocab_size, output_dim=EMBEDDING_DIM, weights=[
          embedding_vectors], input_length=maxlen, trainable=False))
# LSTM
model.add(LSTM(units=128))
model.add(BatchNormalization())
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['acc'])

del embedding_vectors


# In[55]:


model.summary()


# In[56]:


encoder = LabelEncoder()
encoder.fit(y)
labels_val = encoder.transform(y)
labels_val = to_categorical(labels_val)

val_y = labels_val
val_y


# In[57]:


# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, val_y)


# In[ ]:


model.fit(X_train, y_train, validation_split=0.3, epochs=2)


model.save("KAG")
