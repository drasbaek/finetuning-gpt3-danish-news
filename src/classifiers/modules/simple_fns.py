'''
Script for functions used for the simple classifiers BOW and TFID

Written for the paper titled "Fine-tuning GPT-3 for Synthetic Danish News Generation" (Almasi & Schiønning, 2023).
'''
import pathlib  

# data wrangling
import pandas as pd 

# evaluate classifier 
from sklearn import metrics

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

def evaluate_LR(lr_model, X_test, Y_test):
    '''
    Evaluate a fitted logistic regression, extracting accuracy, f1, precision, recall

    Args
        lr_model: logistic regression model 
        X_test: test data 
        Y_test: true labels for test data 

    Returns 
        metrics_dict: dictionary of metrics (accuracy, F1, precision, recall)
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
