#!/bin/bash

# activate virtual environment 
source ./env/bin/activate

# run BOW
echo -e "[INFO:] Running bag-of-words classifier ..."
python src/classifiers/simple_bow.py

# run TF-IDF
echo -e "[INFO:] Running TF-IDF classifier ..."
python src/classifiers/simple_tfid.py

# deactivate virtual env
deactivate