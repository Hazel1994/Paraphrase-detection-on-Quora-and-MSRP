# Quora Question Pairs

> this is my solution for [QQP competition](https://www.kaggle.com/c/quora-question-pairs/overview) launched by kaggle.
my model scored 0.27333 which outperforms the benchmark by 0.43801.
I also tested my model on [MSRP](https://www.microsoft.com/en-us/download/details.aspx?id=52398) dataset which has the exact structure as Quora but it is smaller in size.

## Main requirements:
-keras
-tensorflow
-scikit-learn
-pandas
-NLTK
-fuzzywuzzy
-bs4

## How to use 
- download training and testing data from [here](https://www.kaggle.com/c/quora-question-pairs/data) and copy them into the data
folder under the names "quora_train" and "quora_test". The MSRP dataset is already available there.
-download the 6B token version of Glove from [here](https://nlp.stanford.edu/projects/glove/), then create a dictionary out of it ( term -> vector) and dump it into a pickle file.
Don't know how to do this? check out [here](https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict). This way is much faster than 
reading the whole Glove text file every time you run the program. copy the pickle file into the "GloVe" folder under the name "glove.pkl".
- Run "Quora_features.py" to extract the hand-crafted features for Quora dataset. It will store the features into the "handcrafted_features" folder. run
"MSRP_features.py" for MSRP.
-Run "Run_Quora.py", it will train the model and make a submission file in the "predictions" folder. I will also save the best model in each fold and their progress during training as as [graph](https://www.tensorflow.org/guide/graph_viz).
Run "Run_MSRP.py" for runing the model on MSRP.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* most of the hand-crafted features are based on [this](https://github.com/ianozsvald/string_distance_metrics) and [this](https://github.com/seatgeek/fuzzywuzzy).
