#!/bin/bash

# activate virtual environment 
source ./env/bin/activate

# run BOW
echo -e "[INFO:] Fine-tuning BERT classifier on dummy data ..."
python src/classifiers/finetune_bert.py

# run BERT 
echo -e "[INFO:] Running Fine-Tuned BERT on the test data"
python src/classifiers/inference_bert.py

# deactivate virtual env
deactivate