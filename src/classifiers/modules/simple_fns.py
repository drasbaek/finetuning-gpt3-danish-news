'''
Script for functions used for the simple classifiers BOW and TFID

Written for the paper titled "Fine-tuning GPT-3 for Synthetic Danish News Generation" (Almasi & Schiønning, 2023).
'''
import pathlib  

# data wrangling
import pandas as pd 
import numpy as np

# evaluate classifier 
from sklearn import metrics

# plot confusion matrix 
import matplotlib.pyplot as plt

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

def get_predictions(lr_model, X_test, Y_test):
    '''
    Get predictions and probability scores from LR model 

    Args 
        lr_model: logistic regression model 
        X_test: test data 
        Y_test: true labels for test data 

    Returns 
        Y_predict: predicted labels 
        Y_probability: predicted scores for the predicted labels
    '''
    # extract predictions
    Y_predict = lr_model.predict(X_test)

    # extract probabilities (returns probability score for each class)
    probabilities = lr_model.predict_proba(X_test)

    # get only the highest probability score (the score for the predicted label)
    Y_probability = [max(probabilities[i]) for i in range(len(probabilities))]

    return Y_predict, Y_probability

def get_metrics(Y_test, Y_predict):
    '''
    Extract accuracy, f1, precision, recall from predicted and true labels 

    Args
        Y_test: true labels for test data 
        Y_predict: predicted labels for test data 

    Returns 
        metrics_dict: dictionary of metrics (accuracy, F1, precision, recall)
    '''

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

def create_confusion_matrix(lr_model, Y_test, Y_predict, savepath):
    '''
    Create confusion matrix

    Get predictions and probability scores from LR model 

    Args 
        lr_model: logistic regression model 
        X_test: test data 
        Y_test: true labels for test data 
        savepath: defaults to None. If not None, the path where the .csv path should be saved

    Returns 
        cm: confusion matrix 
    '''

    # create confusion matrix
    cm = metrics.confusion_matrix(Y_test, Y_predict, labels=lr_model.classes_)

    # create plot 
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lr_model.classes_)
    disp.plot()

    # save plot 
    plt.savefig(savepath, dpi=300)

    return cm 

def create_predictions_data(test_data, Y_predict, Y_probability, classifier_name:str, savepath=None):
    '''
    Create predictions dataframe with a prediction and probability column

    Args: 
        test_data: data with texts, labels etc. 
        Y_predict: predicted labels for test data 
        Y_probability: predicted scores for the predicted labels
        classifier_name: string name as suffix for prediction and probability columns (e.g., "bow")
        savepath: defaults to none (if not none, saves CSV file to specified path)
    
    Returns 
        test_data: dataframe with predicted labels and probability scores 
    '''
    
    # create predictions column
    test_data[f"prediction_{classifier_name}"] = Y_predict

    # create probability columns 
    test_data[f"probability_{classifier_name}"] = Y_probability

    # save if savepath is defined
    if savepath is not None:
        test_data.to_csv(savepath)

    return test_data 

