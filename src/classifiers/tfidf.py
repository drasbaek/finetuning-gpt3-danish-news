'''
Script for training and evaluating the simple classifier TF-IDF to distinguish between GPT-3 generated articles and human written articles.

Written for the paper titled "Fine-tuning GPT-3 for Synthetic Danish News Generation" (Almasi & Schiønning, 2023)
'''

import pathlib 

# data & results wrangling 
import pandas as pd
import numpy as np

# vectoriser and model 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# plot confusion matrix 
import matplotlib.pyplot as plt

# custom modules for loading data, evaluating LR 
from modules.simple_fns import prepare_data, evaluate_LR

# functions 
def TFIDF_vectorize(train_data:pd.DataFrame, test_data:pd.DataFrame):
    '''
    Vectorise clean data using a TF-IDF vectorizer

    Args
        train_data: dataframe containing training data
        test_data: dataframe containing test data

    Returns: 
        X_train, X_test: vectorised training and test data (features)
        Y_train, Y_test: vector with true labels for training and test data (0 or 1)
    '''

    # initialise vectoriser 
    vectorizer = TfidfVectorizer(lowercase=False) # lowercase = false as text is already lower-cased

    # vectorise train and test 
    X_train = vectorizer.fit_transform(train_data["text"]).toarray()

    X_test = vectorizer.transform(test_data["text"]).toarray()

    # extract labels
    Y_train = train_data["label"].values
    Y_test = test_data["label"].values 

    return X_train, X_test, Y_train, Y_test
    
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
    X_train, X_test, Y_train, Y_test = TFIDF_vectorize(train_data, test_data)

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