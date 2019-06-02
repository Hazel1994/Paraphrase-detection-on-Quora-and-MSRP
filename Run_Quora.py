from utilities.tools import review_to_words,get_embedding,extract_features
from utilities.classify_ensemble import predict_Quora_test_data

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,Bidirectional,dot
from keras.layers.core import Lambda
from keras.layers.merge import concatenate, multiply,subtract
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
from keras.utils import to_categorical,plot_model
from keras.models import Model

import pandas as pd
import numpy as np

min_word_frequency = 20
out_v_w = "unknown"
max_sentence_length = 30
GloVe_embedding_dim = 300
n_folds = 10
batch_size = 125
n_epochs=30

#load all the handcrafted features
train_nlp_features = np.load("handcrafted_features/quora_train.npy")
test_nlp_features = np.load("handcrafted_features/quora_test.npy")

#load the dataset
train = pd.read_csv("data/quora_train.csv", header=0,delimiter=",",error_bad_lines=False)

print("\npreprocessing training data")
train["question1"] = train["question1"].fillna("").apply(review_to_words)
train["question2"] = train["question2"].fillna("").apply(review_to_words)

print("creating vocabulary of words occured more than", min_word_frequency)
all_questions = pd.Series(train["question1"].tolist() + train["question2"].tolist()).unique()
vectorizer = CountVectorizer(lowercase=False, token_pattern="\S+", min_df=min_word_frequency)
vectorizer.fit(all_questions)
top_words = set(vectorizer.vocabulary_.keys())
top_words.add(out_v_w)

#create an ebbedding dictionary out of the created vocabulary
embeddings_index = get_embedding(top_words)
top_words = embeddings_index.keys()

print("preparing training data")
q1s_train, q2s_train,train_f_extra = extract_features(train,top_words)

#use tokenizer to convert each sentence into its words' indecies
tokenizer = Tokenizer(filters="")
tokenizer.fit_on_texts(np.append(q1s_train, q2s_train))
word_index = tokenizer.word_index
data_1 = pad_sequences(tokenizer.texts_to_sequences(q1s_train), maxlen=max_sentence_length)
data_2 = pad_sequences(tokenizer.texts_to_sequences(q2s_train), maxlen=max_sentence_length)
labels = np.array(train["is_duplicate"])

print("initilizae the embedding layer with GloVe word vectors")
nb_words = len(word_index) + 1
print('size of the embedding matrix : ',nb_words)
embedding_matrix = np.zeros((nb_words, GloVe_embedding_dim))

for word, i in word_index.items():
    embedding_vector = embeddings_index[word]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


#divide the whole dataset into n_folds parts
labels_temp=labels
labels = to_categorical(labels,num_classes=2)

skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
model_count = 1

train_nlp_features=np.concatenate((train_f_extra,train_nlp_features),axis=1)

#loop over each validation/train section
for idx_train, idx_val in skf.split(data_1,labels_temp):

    print("model number :", model_count)

    #select train and validation samples
    data_1_train = data_1[idx_train]
    data_2_train = data_2[idx_train]
    labels_train = labels[idx_train]
    f_train = train_nlp_features[idx_train]

    data_1_val = data_1[idx_val]
    data_2_val = data_2[idx_val]
    labels_val = labels[idx_val]
    f_val = train_nlp_features[idx_val]

    ######################___defining the model___#############################

    #define an embedding layer initilized with Glove weights
    embedding_layer = Embedding(nb_words, GloVe_embedding_dim,weights=[embedding_matrix],
                                input_length=max_sentence_length, trainable=False)

    #define a bi-directional lstm neural network
    lstm_layer = Bidirectional(LSTM(100, recurrent_dropout=0.4,input_shape=(nb_words,GloVe_embedding_dim), return_sequences=False), merge_mode='mul')

    #encode sentence 1
    sequence_1_input = Input(shape=(max_sentence_length,), dtype="int32")
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    s1 = lstm_layer(embedded_sequences_1)
    s1=BatchNormalization()(s1)

    #encode sentence 2
    sequence_2_input = Input(shape=(max_sentence_length,), dtype="int32")
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    s2 = lstm_layer(embedded_sequences_2)
    s2=BatchNormalization()(s2)

    #define an input for handcrafted features
    nlp_input = Input(shape=(train_nlp_features.shape[1],), dtype="float32")
    features_dense = BatchNormalization()(nlp_input)
    features_dense = Dense(100, activation="relu")(features_dense)
    features_dense = BatchNormalization()(features_dense)

    #computing cosine similarity
    csd=dot([s1,s2],axes=-1, normalize=True)

    #computng  multiplication between the 2 vectors
    mul_v = multiply([s1, s2])

    #compute the absolute difference
    x_y = subtract([s1, s2])
    merged=Lambda(lambda x:abs(x))(x_y)

    #merge the features
    merged = concatenate([merged, mul_v])
    merged = Dropout(0.3)(merged)

    #final features for each pair of sentences
    merged = concatenate([merged, features_dense,csd])
    merged = BatchNormalization()(merged)

    merged = Dense(200, activation="relu")(merged)
    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)

    #using a softmax classifer
    out = Dense(2, activation="softmax")(merged)

    #define the input to feed the model
    model = Model(inputs=[sequence_1_input, sequence_2_input,nlp_input], outputs=out)

    #set up the optimizer, loss function and the evlauation metric
    model.compile(loss="binary_crossentropy",optimizer="nadam",metrics=['acc'])

    best_model_path = "Kfold/best_model_" + str(model_count)

    #stop the training if validation accuracy does not imporve after 6 consecutive epochs
    early_stopping = EarlyStopping(monitor="val_loss", patience=6)
    #save the best model based on the validation accuracy
    model_checkpoint = ModelCheckpoint(best_model_path,monitor='val_acc', save_best_only=True, save_weights_only=True)
    #save the graph for each model while training
    tensorboard = TensorBoard(log_dir="graph",write_graph=True, write_images=True,write_grads=True)
    #model.summary()

    #train the model
    hist = model.fit([data_1_train, data_2_train,f_train], labels_train,
                     validation_data=([data_1_val, data_2_val,f_val], labels_val),
                     epochs=n_epochs, batch_size=batch_size, shuffle=True,
                     callbacks=[ tensorboard,model_checkpoint], verbose=2)

    #load the best model achieved and report the best accuracy on validation data
    model.load_weights(best_model_path)
    print('model number ',model_count, ", best validation accuracy :",max(hist.history["val_acc"]))

    model_count=model_count+1

print("\ndone training")
print("preprocessing testing data")

test = pd.read_csv("data/quora_test.csv", header=0, delimiter=",",error_bad_lines=False)
valid_ids =[type(x)==int for x in test.test_id]
test = test[valid_ids].drop_duplicates()

test["question1"] = test["question1"].fillna("").apply(review_to_words)
test["question2"] = test["question2"].fillna("").apply(review_to_words)

q1s_test, q2s_test, test_f_extra = extract_features(test, top_words)
test_data_1 = pad_sequences(tokenizer.texts_to_sequences(q1s_test), maxlen=max_sentence_length)
test_data_2 = pad_sequences(tokenizer.texts_to_sequences(q2s_test), maxlen=max_sentence_length)
test_nlp_features=np.concatenate((test_f_extra,test_nlp_features),axis=1)


print("using all the trained models and predict the test data")
predict_Quora_test_data(n_folds,nb_words,test_nlp_features,test_data_1,test_data_2)
