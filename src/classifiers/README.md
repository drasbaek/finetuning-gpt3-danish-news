# Reproducing the Classifier Pipeline (Experiment B)
To reproduce the construction of the machine classifiers (Experiment B in the paper) on dummy data, please follow the steps below. This will only work if you have previously run the `setup.sh` as explained in the main [README.md](https://github.com/drasbaek/finetuning-gpt3-danish-news#setup).

⚠️ `NOTE!`  Everything is done on `dummy data` except inference in `inference_bert.py` which is done with the fine-tuned model [MinaAlmasi/dknes-NB-BERT-AI-classifier](https://huggingface.co/MinaAlmasi/dknews-NB-BERT-AI-classifier) on actual data.

## Overview of Files 
The `classifiers` folder contains scripts which do the following:
| <div style="width:120px"></div>| Description |
|---------|:-----------|
| `finetune_bert.py`  | Fine-tune [NbAiLab/nb-bert-large](https://huggingface.co/NbAiLab/nb-bert-large) on dummy data. Optionally, you can push the model to the Hugging Face Hub.              |
| `inference_bert.py`  | Perform inference with [MinaAlmasi/dknes-NB-BERT-AI-classifier](https://huggingface.co/MinaAlmasi/dknews-NB-BERT-AI-classifier) on `ACTUAL` test data.              |
| `simple_bow.py`  | Construct a logistic regression using a `bag-of-words` representation on dummy data.               |
| `simple_tfid.py`  | Construct a logistic regression using a `TF-IDF` representation on dummy data.        |


## Constructing the Simple Classifiers 
To run BOW and TF-IDF classifiers on dummy data, please run `simple_classifiers.sh` in the terminal:
```
bash simple_classifiers.sh
```

## Fine-Tuning BERT
To run the BERT fine-tuning pipeline, type: 
```
bash bert_classifier.sh
```
⚠️ `NOTE!` While the fine-tuning of [NbAiLab/nb-bert-large](https://huggingface.co/NbAiLab/nb-bert-large) is done on dummy data, the inference is done with the `actual` fine-tuned classifier on the real `test_data`.

The fine-tuned `BERT` can be accessed from the Hugging Face Hub: [MinaAlmasi/dknews-NB-BERT-AI-classifier](https://huggingface.co/MinaAlmasi/dknews-NB-BERT-AI-classifier).


## [OPTIONAL] Pushing to HF Hub
Pushing models to the [Hugging Face Hub](https://huggingface.co/models) is disabled by default in all scripts. If you wish to push models to the Hugging Face Hub, you need to firstly save a [Hugging Face token](https://huggingface.co/docs/hub/security-tokens) in a .txt file called ```hf_token.txt``` in the `tokens` folder.

Then please run the lines in the chunk below. 

```
# activate env
source ./env/bin/activate

# run BOW with -hub flag to push to HF hub
echo -e "[INFO:] Fine-tuning BERT classifier on dummy data ..."
python src/classifiers/finetune_bert.py -hub

# deactivate env
deactivate 
```
