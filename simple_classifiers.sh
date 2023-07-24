#!/bin/bash

# activate virtual environment 
source ./env/bin/activate

# run BOW
echo -e "[INFO:] Running bag-of-words classifier ..."
python src/classifiers/bow.py

# run TF-IDF
echo -e "[INFO:] Running TF-IDF classifier ..."
python src/classifiers/TF-IDF.py

# deactivate virtual env
deactivate