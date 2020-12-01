from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
import pandas as pd


def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'

    if tag.startswith('V'):
        return 'v'

    if tag.startswith('J'):
        return 'a'

    if tag.startswith('R'):
        return 'r'

    return None


def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None

    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None


def sentence_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))

    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]

    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0

    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        ls = [synset.path_similarity(
            ss) for ss in synsets2 if synset.path_similarity(ss)]

        if ls:
            best_score = max(ls)
        else:
            best_score = None

        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1

    # Average the values
    if score != 0:
        score /= count
        return score
    else:
        return 0.0


# data = pd.read_csv("train.csv", sep=",")
# ag = data.loc[data["label"] == "unrelated"]
# print(ag)

# s1tos2 = []
# s2tos1 = []

# c = 0

# for index, row in ag.iterrows():
#     s1 = row['title1_en']
#     s2 = row['title2_en']
#     s1tos2.append(sentence_similarity(s1, s2))
#     s2tos1.append(sentence_similarity(s2, s1))
#     print(c)
#     c += 1

# col1 = pd.DataFrame({'title1_en': ag['title1_en'].tolist()})
# col2 = pd.DataFrame({'title2_en': ag['title2_en'].tolist()})
# col3 = pd.DataFrame({'s1tos2': s1tos2})
# col4 = pd.DataFrame({'s2tos1': s2tos1})

# agreed = pd.concat([col1, col2, col3, col4], axis=1)
# agreed.to_csv("agreed.csv", index=False)

print(sentence_similarity("100 stranger fujian provinc steal children",
                          "tell rumour student rob stranger suichang fake"))
