import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
from termcolor import colored


def remove_stopwords(sentence):
    english_stopwords = stopwords.words("english")
    return " ".join(i for i in sentence.split(" ") if i not in english_stopwords)


def stemming(sentence):
    stemmer = PorterStemmer()
    return " ".join(stemmer.stem(word) for word in sentence.split(" "))


def pre_process(data):
    """
    get data from the csv file
    clean the data - lowercase, puncutations, html tags etc.
    normalization - stemming
    :return: title1 (sentences) and labels (sentiment)
    """
    print(colored("1. Preprocessing Data", "yellow"))
   
    REPLACE_NO_SPACE = re.compile("[_.;:!\'?,\"\(\)\[\]<>$0-9]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    # make lower case and remove puncutations
    print(colored("\t1.1 Cleaning data...", "yellow"), end="", flush=True)
    title1 = [REPLACE_NO_SPACE.sub("", line.lower()) for line in data["title1_en"]]
    title1 = [REPLACE_WITH_SPACE.sub(" ", line) for line in title1]

    title2 = [REPLACE_NO_SPACE.sub("", line.lower()) for line in data["title2_en"]]
    title2 = [REPLACE_WITH_SPACE.sub(" ", line) for line in title2]
    print(colored(" [Done]", "green"))

    print(colored("\t1.2 Removing stopwords...", "yellow"), end="", flush=True)
    title1 = [remove_stopwords(i) for i in title1]
    title2 = [remove_stopwords(i) for i in title2]

    print(colored(" [Done]", "green"))

    print(colored("\t1.3 Stemming...", "yellow"), end="", flush=True)
    title1 = [stemming(i) for i in title1]
    title2 = [stemming(i) for i in title2]

    print(colored(" [Done]", "green"))
    labels = data["label"]

    return pd.DataFrame({'title1':title1, 'title2':title2, 'label':labels})
