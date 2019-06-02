from utilities.f_extractor import read_model,extract_features
import pandas as pd
import numpy as np

#load the GloVe word embedding model as a pickle file
model = read_model("GloVe/glove.pkl")

#load the dataset
train = pd.read_csv("data/quora_train.csv", header=0,delimiter=",",error_bad_lines=False)
test = pd.read_csv("data/quora_test.csv", header=0, delimiter=",",error_bad_lines=False)
valid_ids =[type(x)==int for x in test.test_id]
test = test[valid_ids].drop_duplicates()

#separate train and test text
train_1,train_2 = train["question1"],train["question2"]
test_1,test_2 = test["question1"], test["question2"]

print("extracting features for train reviews...\n")
train_reviews_vec=[]
for i in range(len(train_1)):
    r1,r2 = train_1[i],train_2[i]
    train_reviews_vec.append(extract_features(model,r1,r2))
    if (i+1) %10000==0:
     print(i+1 ," out of " ,len(train_1) )

print("extracting features for test reviews...\n")
test_reviews_vec=[]
for i in range(len(test_1)):
    r1,r2 = test_1[i],test_2[i]
    test_reviews_vec.append(extract_features(model,r1,r2))
    if (i+1) %10000==0:
        print(i+1 ," out of " ,len(test_1) )

#release the space taken by the GloVe dictionary
del model

#convert list of features into arrays and save them
X_train = np.asarray(train_reviews_vec)
X_test = np.asarray(test_reviews_vec)

print('number of the extracted features')
print(X_train.shape[1])

np.save('handcrafted_features/quora_train',X_train)
np.save('handcrafted_features/quora_test',X_test)
