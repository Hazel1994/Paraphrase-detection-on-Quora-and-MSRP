from sklearn.metrics import f1_score,accuracy_score
import numpy as np
from utilities.tools import load_model
import pandas as pd

def predict_MSRP_test_data(n_models,nb_words,nlp_f,test_data_1,test_data_2,test_labels):

    models=[]
    n_h_features=nlp_f.shape[1]
    print('loading the models...')
    for i in range(n_models):
        models.append(load_model(i+1,nb_words,n_h_features))

    preds=[]
    print('predicting the test data...\n')
    i=0
    for m in models:
      i+=1
      preds_prob=m.predict([test_data_1, test_data_2,nlp_f], batch_size=64, verbose=0)
      preds.append(preds_prob[:,1])

    preds=np.asarray(preds)
    final_labels=np.zeros(len(test_data_1),dtype=int)

   #average the predicttion
    for i in range(len(test_data_1)):
        final_labels[i]=round(np.mean(preds[:,i]))
        if i%100==0:
            print(i ,' out of ',len(test_data_1))

    print("test data accuracy: ", accuracy_score(final_labels,test_labels))
    print("test data f_measure: ", f1_score(final_labels, test_labels))

    submission = pd.DataFrame({"Quality": final_labels})
    submission.to_csv("predictions/MSRP.tsv", index=True,index_label='test_id')


def predict_Quora_test_data(n_models,nb_words,nlp_f,test_data_1,test_data_2):

    models=[]
    n_h_features=nlp_f.shape[1]
    print('loading the models...')
    for i in range(n_models):
        models.append(load_model(i+1,nb_words,n_h_features))

    preds=[]
    print('predicting the test data...\n')
    i=0
    for m in models:
      i+=1
      preds_prob=m.predict([test_data_1, test_data_2,nlp_f], batch_size=125, verbose=0)
      preds.append(preds_prob[:,1])

    preds=np.asarray(preds)
    final_labels=np.zeros(len(test_data_1),dtype=float)

   #average the predicttion
    for i in range(len(test_data_1)):
        final_labels[i]=np.mean(preds[:,i])
        if i%10000==0:
            print(i ,' out of ',len(test_data_1))

    print('making the sumbission file')

    submission = pd.DataFrame({"is_duplicate": final_labels})
    submission.to_csv("predictions/Quora.tsv", index=True,index_label='test_id')