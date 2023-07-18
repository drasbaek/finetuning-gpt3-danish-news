'''
Script for training and evaluating the simple classifier BOW to distinguish between GPT-3 generated 

Written for the paper titled "Fine-tuning GPT-3 for Synthetic Danish News Generation" (Almasi & Schiønning, 2023)
'''

import pathlib 

# data & results wrangling 
import pandas as pd
import numpy as np

# vectoriser and model 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# evaluate classifier 
from sklearn import metrics

# plot confusion matrix 
import matplotlib.pyplot as plt

# functions 
def prepare_data(train_path:pathlib.Path, test_path:pathlib.Path):
    '''
    Prepare data for TF-IDF or BOW classifier

    Args
        train_path: path to train data
        test_path: path to test data 

    Returns: 
        train_data, test_data: preprocessed pandas dataframes
    '''
    
    # read in data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # remove punctuations 
    train_data["text"] = train_data["text"].str.replace(r'[^\w\s]+', '', regex=True)
    test_data["text"] = test_data["text"].str.replace(r'[^\w\s]+', '', regex=True)
    
    # make lower case 
    train_data["text"] = train_data["text"].str.lower()
    test_data["text"] = test_data["text"].str.lower()

    return train_data, test_data
    

def BOW_vectorize(train_data:pd.DataFrame, test_data:pd.DataFrame):
    '''
    Vectorise clean data using a bag-of-words (BOW) vectoriser 

    Args
        train_data: dataframe containing training data
        test_data: dataframe containing test data

    Returns: 
        X_train, X_test: vectorised training and test data (features)
        Y_train, Y_test: vector with true labels for training and test data (0 or 1)
    '''

    # initialise vectoriser 
    vectorizer = CountVectorizer(lowercase=False) # lowercase = false as text is already lower-cased

    # vectorise train and test 
    X_train = vectorizer.fit_transform(train_data["text"]).toarray()

    X_test = vectorizer.transform(test_data["text"]).toarray()

    # extract labels
    Y_train = train_data["label"].values
    Y_test = test_data["label"].values 

    return X_train, X_test, Y_train, Y_test

def evaluate_LR(lr_model, X_test, Y_test):
    '''
    Evaluate a fitted logistic regression, extracting accuracy, f1, precision, recall

    Args:
        lr_model: logistic regression model 
        X_test = test data 
        Y_test = true labels for test data 
    '''

    # extract predictions
    Y_predict = lr_model.predict(X_test)

    # calculate metrics 
    accuracy = metrics.accuracy_score(Y_test, Y_predict)
    f1_score = metrics.f1_score(Y_test, Y_predict)
    precision = metrics.precision_score(Y_test, Y_predict)
    recall = metrics.recall_score(Y_test, Y_predict)

    # make into dictionary 
    metrics_dict = {"Accuracy": round(accuracy, 3),
                "F1": round(f1_score, 3),
                "Precision": round(precision, 3),
                "Recall": round(recall, 3)
                } 

    return metrics_dict 
    
def main(): 
    # define paths
    path = pathlib.Path(__file__) 
    path_train = path.parents[2] / "data" / "labelled_data_for_classifier.csv"
    path_test =  path.parents[2] / "data" / "test_data_classifier.csv"

    # prepare data 
    print("preparing data ...")
    train_data, test_data = prepare_data(path_train, path_test)

    # vectorise data
    print("vectorising data ...")
    X_train, X_test, Y_train, Y_test = BOW_vectorize(train_data, test_data)

    # intialise logistic regression
    lr = LogisticRegression(solver="lbfgs", C=10, random_state=10, max_iter=250)

    # train
    print("fitting model ...")
    lr.fit(X_train, Y_train)

    # evaluate 
    print("evaluating model ...")
    metrics = evaluate_LR(lr, X_test, Y_test)

    print(metrics)

if __name__ == "__main__":
    main()