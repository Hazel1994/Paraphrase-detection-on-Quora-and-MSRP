import numpy as np
from nltk.corpus import stopwords
import pickle
from scipy.spatial.distance import euclidean
import re
import distance
from bs4 import BeautifulSoup
from scipy import spatial
from fuzzywuzzy import fuzz

STOP_WORDS = stopwords.words("english")

#load the Glove model
def read_model(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

#preprocessing module
def review_to_words(raw_review):
    text = BeautifulSoup(raw_review,'html5lib').get_text()
    text = text.lower().split()
    text = " ".join(text)

    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", "not", text)
    text = re.sub("it's", "it is", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"thnx", "thanks", text)
    text = re.sub(r":\)", "smile", text)
    text = re.sub(r"%", " percent ", text)
    text = re.sub(r"₹", " rupee ", text)
    text = re.sub(r"\$", " dollar ", text)
    text = re.sub(r"€", " euro ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " america ", text)
    text = re.sub(r" u s ", " america ", text)
    text = re.sub(r" uk ", " england ", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r" dms ", "direct messages ", text)
    text = re.sub(r"demonitization", "demonetization", text)
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text)
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text)
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"III", "3", text)
    text = re.sub(r"ya", "you", text)
    text = re.sub(r"coz", "cause", text)
    text = re.sub(r"bc", "because", text)
    text = re.sub(r"b/c", "because", text)
    text = re.sub(r"lol", "laugh", text)
    text = re.sub(r"the US", "america", text)
    text = re.sub(r"the US", "america", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r" j k ", " jk ", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r"wanna", "want to", text)
    text = re.sub(r"gonna", "going to", text)

    text = re.sub(r",000", " thousand", text)
    text = re.sub(r",000,000", "million", text)
    text = re.sub(r"([0-9]+)000000", r"\1million", text)
    text = re.sub(r"([0-9]+)000", r"\1thousand", text)

    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    return text

#compute the cosine similarity between 2 given vectors
def cosine_sim(a, b):
    result = 1 - spatial.distance.cosine(a, b)
    return result

#compute the euclidean distance  between 2 given vectors
def euclidean_distance(a, b):
    return euclidean(a, b)

#feature extraction module
def extract_features(model,sentence_1, sentence_2):

    features = []
    #preprocessing each pair of sentences and tokenize them
    sentence_1 = review_to_words(sentence_1)
    sentence_2 = review_to_words(sentence_2)
    tokens_1 = sentence_1.split()
    tokens_2 = sentence_2.split()

    #compute average of Glove word vectors for each sentence
    vec1 = np.zeros([300], dtype=float)
    vec2 = np.zeros([300], dtype=float)
    counter = 1
    for i in range(len(tokens_1)):
        if tokens_1[i] in model:
            counter += 1
            vec1 += model[tokens_1[i]]
    vec1 = vec1 / counter

    counter = 1
    for i in range(len(tokens_2)):
        if tokens_2[i] in model:
            counter += 1
            vec2 += model[tokens_2[i]]
    vec2 = vec2 / counter

    #add diffferent features
    features.append(cosine_sim(vec1, vec2))
    features.append(euclidean_distance(vec1,vec2))
    features.append(jaccard_similarity(sentence_1, sentence_2))
    features.append(distance_title_len(sentence_1, sentence_2))
    features.append(get_longest_substr_ratio(sentence_1, sentence_2))
    features.append(distance_bigrams_same(sentence_1, sentence_2))

    SAFE_DIV = 0.0001
    token_features = [0.0] * 9

    q1_tokens = sentence_1.split()
    q2_tokens = sentence_2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return features

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    token_features[8] = (len(q1_tokens) + len(q2_tokens)) / 2

    for i in token_features:
        features.append(i)

    #use fuzzy wuzzy library
    features.append(fuzz.token_set_ratio(sentence_1,sentence_2))
    features.append(fuzz.token_sort_ratio(sentence_1, sentence_2))
    features.append(fuzz.QRatio(sentence_1, sentence_2))
    features.append(fuzz.partial_ratio(sentence_1, sentence_2))

    return features

#LCS similarity metric
def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)

#jaccard similarity metric
def jaccard_similarity(query, document):
    query=query.split()
    document=document.split()

    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))

    return (len(intersection)) / len(union) + .00001

#difference in lenght of each pair
def distance_title_len(t1, t2):
    return abs(len(t1) - len(t2))

#bigram overlap
def distance_bigrams_same(t1, t2):
    """Bigram distance metric, term frequency is ignored,
       0 if bigrams are identical, 1.0 if no bigrams are common"""
    t1_terms = make_terms_from_string(t1)
    t2_terms = make_terms_from_string(t2)
    terms1 = set(ngrams(t1_terms, 2))
    terms2 = set(ngrams(t2_terms, 2))
    shared_terms = terms1.intersection(terms2)
    all_terms = terms1.union(terms2)
    dist = 1.0
    if len(all_terms) > 0:
        dist = (len(shared_terms) / float(len(all_terms)))
    return dist

def make_terms_from_string(s):
    """turn string s into a list of unicode terms"""
    u = s
    return u.split()

def ngrams(sequence, n):
    """Create ngrams from sequence, e.g. ([1,2,3], 2) -> [(1,2), (2,3)]
       Note that fewer sequence items than n results in an empty list being returned"""
    # credit: http://stackoverflow.com/questions/2380394/simple-implementation-of-n-gram-tf-idf-and-cosine-similarity-in-python
    sequence = list(sequence)
    count = max(0, len(sequence) - n + 1)
    return [tuple(sequence[i:i + n]) for i in range(count)]
