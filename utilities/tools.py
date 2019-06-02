import numpy as np
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import pickle
import re

from keras.layers import Dense, Input, LSTM, Embedding, Dropout,Bidirectional,GRU,dot
from keras.layers.core import Lambda
from keras.layers.merge import concatenate, multiply,subtract
from keras.layers.normalization import BatchNormalization
from keras.models import Model

STOP_WORDS = set(stopwords.words('english'))

GloVe_embedding_dim = 300
max_sentence_length = 30
embedding_model_file='GloVe/glove.pkl'


def read_model(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

def is_numeric(s):
    return any(i.isdigit() for i in s)

def prepare(q,top_words):
    new_q = []
    surplus_q = []
    numbers_q = []
    unknown = True
    for w in q.split()[::-1]:
        if w in top_words:
            new_q = [w] + new_q

            unknown = True
        elif w not in STOP_WORDS:
            if unknown:
                new_q = ["unknown"] + new_q
                unknown = False
            if is_numeric(w):
                numbers_q = [w] + numbers_q
            else:
                surplus_q = [w] + surplus_q
        else:
            unknown = True
        if len(new_q) == max_sentence_length:
            break
    new_q = " ".join(new_q)
    return new_q, set(surplus_q), set(numbers_q)

def extract_features(df,top_words):
    q1s = np.array([""] * len(df), dtype=object)
    q2s = np.array([""] * len(df), dtype=object)
    features = np.zeros((len(df),4))

    for i, (q1, q2) in enumerate(list(zip(df["question1"], df["question2"]))):
        q1s[i], surplus1, numbers1 = prepare(q1,top_words)
        q2s[i], surplus2, numbers2 = prepare(q2,top_words)
        features[i, 0] = len(surplus1.intersection(surplus2))
        features[i, 1] = len(surplus1.union(surplus2))
        features[i, 2] = len(numbers1.intersection(numbers2))
        features[i, 3] = len(numbers1.union(numbers2))

    return q1s, q2s,features

def review_to_words(raw_review):
    text = BeautifulSoup(raw_review,'html5lib').get_text()

    text = text.lower().split()

    text = " ".join(text)

    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"cannot", "can not ", text)
    text = re.sub(r"can't", "can not ", text)
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
    text = re.sub(r"doesnot", " does not ", text)
    text = re.sub(r"donot", "do not ", text)
    text = re.sub(r"wouldnot", "would not ", text)
    text = re.sub(r"hasnot", "has not ", text)
    text = re.sub(r"isnot", "is not ", text)

    text = re.sub(r"wasnot", "was not ", text)
    text = re.sub(r"werenot", "were not ", text)
    text = re.sub(r"wonot", "will not ", text)
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

    text = re.sub(r",000", " k", text)
    text = re.sub(r",000,000", "m", text)
    text = re.sub(r"([0-9]+)000000", r"\1 m", text)
    text = re.sub(r"([0-9]+)000", r"\1 k", text)

    text = re.sub(r"([0-9]+)([a-zA-Z]+)", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z]+)([0-9]+)", r"\1 \2", text)

    text = re.sub(r"[^A-Za-z0-9]", " ", text)

    return (text)

def get_embedding(top_words):
    embeddings_dict = {}
    embeddings_matrix = read_model(embedding_model_file)
    for word in top_words:
        if word in embeddings_matrix:
            embeddings_dict[word] = embeddings_matrix[word]
    del embeddings_matrix
    return embeddings_dict

def load_model(number,nb_words,n_handcrafted_features):

    embedding_matrix = np.zeros((nb_words, GloVe_embedding_dim))

    embedding_layer = Embedding(nb_words, GloVe_embedding_dim, weights=[embedding_matrix],
                                input_length=max_sentence_length, trainable=False)
    lstm_layer = Bidirectional(LSTM(100, recurrent_dropout=0.4, return_sequences=False), merge_mode='mul')

    # sentence 1
    sequence_1_input = Input(shape=(max_sentence_length,), dtype="int32")
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    s1 = lstm_layer(embedded_sequences_1)
    s1 = BatchNormalization()(s1)

    # sentence 2
    sequence_2_input = Input(shape=(max_sentence_length,), dtype="int32")
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    s2 = lstm_layer(embedded_sequences_2)
    s2 = BatchNormalization()(s2)

    # handcrafted features
    nlp_input = Input(shape=(n_handcrafted_features,), dtype="float32")
    features_dense = BatchNormalization()(nlp_input)
    features_dense = Dense(100, activation="relu")(features_dense)
    features_dense = BatchNormalization()(features_dense)

    # computing cosine similarity
    csd = dot([s1, s2], axes=-1, normalize=True)
    # computng  multiplication between the 2 vectors
    mul_v = multiply([s1, s2])
    # compute the absolute difference
    x_y = subtract([s1, s2])
    merged = Lambda(lambda x: abs(x))(x_y)

    # merge the features
    merged = concatenate([merged, mul_v])
    merged = Dropout(0.3)(merged)

    merged = concatenate([merged, features_dense, csd])
    merged = BatchNormalization()(merged)

    merged = Dense(200, activation="relu")(merged)
    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)

    out = Dense(2, activation="softmax")(merged)
    model = Model(inputs=[sequence_1_input, sequence_2_input, nlp_input], outputs=out)
    model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=['acc'])
    best_model_path = "Kfold/best_model_" + str(number)
    model.load_weights(best_model_path)

    return model
