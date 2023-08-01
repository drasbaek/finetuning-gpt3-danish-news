'''
Script with functions used for fine-tuning and evaluating the BERT classifier. Used in the "bert.py" script. 

Written for the paper titled "Fine-tuning GPT-3 for Synthetic Danish News Generation" (Almasi & Schiønning, 2023).
'''
import pathlib
from functools import partial 

# data wrangling 
from datasets import Dataset, DatasetDict
import pandas as pd 
from sklearn.model_selection import train_test_split

# transformers tokenizers, models 
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                        Trainer, DataCollatorWithPadding, EarlyStoppingCallback)

# for compute_metrics function used during training
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# for evaluation, getting predictions
from sklearn.metrics import classification_report
import torch 

def prepare_data(train_path:pathlib.Path, test_path:pathlib.Path):
    '''
    Prepare data for BERT classifier  

    Args
        train_path: path to train data
        test_path: path to test data 

    Returns: 
        dataset: HF dataset dictionary containing train, eval and test ds 
    '''
    
    # read in data
    data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # select only relevant columns 
    test_data = test_data[["text", "label"]]

    # ensure that data is in the right format 
    data['text'] = data['text'].astype(str)
    test_data['text'] = test_data['text'].astype(str)

    data['label'] = data['label'].astype(int)
    test_data['label'] = test_data['label'].astype(int)

    # split "data" into train_data and eval_data 
    train_data, eval_data = train_test_split(data)

    # convert pandas dataframes into huggingface datasets
    train_ds = Dataset.from_pandas(train_data, preserve_index = False) # removing past indices with preserve_index = False
    eval_ds = Dataset.from_pandas(eval_data, preserve_index = False)
    test_ds = Dataset.from_pandas(test_data, preserve_index = False)

    # combine all three datasets into one dataset dict 
    dataset = DatasetDict({"train":train_ds,"eval":eval_ds, "test":test_ds})

    return dataset 

def tokenize(example, tokenizer, text_col:str="text"):
    '''
    Tokenize an example in HF dataset.

    Args:
        - example: dataset dict with text column
        - tokenizer: intialized tokenizer
        - text_col: name of text column in dataset dict
    
    Returns:
        - tokenized example
    '''
    return tokenizer(example[text_col], truncation=True)
    
def tokenize_dataset(dataset, tokenizer, text_col:str="text"): 
    '''
    Tokenize dataset using tokenizer function.

    Args:
        - dataset: dataset dict (HF dataset)
        - tokenizer: intialized tokenizer
        - text_col: name of text column in dataset dict
        
    Returns:
        - tokenized dataset
    '''
    # prepare tokenize func with multiple arguments to be passed to "map"
    tokenize_func = partial(tokenize, tokenizer=tokenizer, text_col=text_col)
    
    # tokenize entire dataset, using multiprocessing
    tokenized_dataset = dataset.map(tokenize_func, batched=True)

    return tokenized_dataset

def compute_metrics(pred):
    '''
    Compute metrics for training and validation data.

    Args:
        - pred: predictions from trainer.predict() (or to be used within trainer when training)
    
    Returns:
        - metrics_dict: dictionary with metrics (accuracy, f1, precision, recall)
    '''
    # get labels 
    labels = pred.label_ids

    # get predictions
    preds = pred.predictions.argmax(-1)

    # calculate metrics 
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)

    # return dict 
    metrics_dict = {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    return metrics_dict 

def get_loss(trainer_history):
    '''
    Get train and eval loss from trainer history. Useful for diagnostics in fine-tune pipeline.

    Args:
        - trainer_history: trainer.history (list of dicts)

    Returns 
        - train_loss: dictionary with epoch (key) and train loss (value)
        - eval_loss: dictionary with epoch (key) and eval loss (value)
    '''
    train_loss = {}
    eval_loss = {}

    # get train and eval loss for each epoch by looping through trainer history
    for item in trainer_history:
        epoch = item['epoch'] 
        # get train loss per epoch
        if "loss" in item:
            train_loss[epoch] = item["loss"] 

        # get eval loss per epoch
        if "eval_loss" in item:
            eval_loss[epoch] = item["eval_loss"]

    # get total number of epochs by getting the length of the train loss dict
    total_epochs = len(train_loss.keys())

    return train_loss, eval_loss, total_epochs

def plot_loss(train_loss, val_loss, epochs, savepath, filename): # adapted from class notebook
    '''
    Plot train and validation loss for a single figure. Useful for diagnostics in fine-tune pipeline. 

    Args:
        - train_loss: dictionary with epoch (key) and train loss (value)
        - val_loss: dictionary with epoch (key) and val loss (value)
        - epochs: total number of epochs
        - savepath: directory where folder is created to save plot
        - filename: filename of plot

    Outputs: 
        - .png of train and validation loss
    '''

    # define theme 
    plt.style.use("seaborn-colorblind")

    # define figure size 
    plt.figure(figsize=(8,6))

    # create plot of train and validation loss, defined as two subplots on top of each other ! 
    plt.plot(np.arange(1, epochs+1), train_loss.values(), label="Train Loss") # plot train loss 
    plt.plot(np.arange(1, epochs+1), val_loss.values(), label="Val Loss", linestyle=":") # plot val loss
    
    # text description on plot !!
    plt.title("Loss curve") 
    plt.xlabel("Epoch") 
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    # make folder in save path 
    fullpath = savepath / "loss_curves"
    fullpath.mkdir(parents=True, exist_ok=True)

    # save fig 
    plt.savefig(fullpath / filename, dpi=300)

def finetune(dataset, model_name:str, n_labels:int, training_args, early_stop_patience:int=3): 
    '''
    Fine-tune model on dataset. Tokenizes dataset, defines datacollator, earlystop, and trainer.

    Args:
        - dataset: dataset dict (HF dataset)
        - model_name: name of model to be used (e.g. "distilbert-base-uncased")
        - n_labels: number of labels in dataset
        - training_args: training arguments (from transformers)
        - early_stop_patience: number of epochs to wait before stopping training if no improvement in eval loss (default=3)
    
    Returns:
        - trainer: trainer object (from transformers)
        - tokenized_data: tokenized dataset
    '''

    # import tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # tokenize 
    tokenized_data = tokenize_dataset(dataset, tokenizer, "text")

    # define datacollator 
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # define earlystop
    early_stop = EarlyStoppingCallback(early_stopping_patience = early_stop_patience)

    # initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_labels,
    )

    # initialize trainer 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["eval"], 
        tokenizer=tokenizer, 
        data_collator=data_collator, 
        compute_metrics = compute_metrics, 
        callbacks = [early_stop],
    )

    # train model
    trainer.train()

    return trainer, tokenized_data