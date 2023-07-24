# Fine-tuning GPT-3 for Synthetic Danish News Generation
This repository contains the code written for the paper titled, **"Fine-tuning GPT-3 for Synthetic Danish News Generation" (Almasi & Schiønning, 2023)**. 

The project involves fine-tuning GPT-3 to produce synthetic news articles in Danish and evaluating the model in binary classification tasks. The evaluation relied on both human participants (A) and machine classifiers (B).

To read the details of this evaluation, please refer to our paper. 

## Structure 
The repository is structured as such:

| <div style="width:120px"></div>| Description |
|---------|:-----------|
| ```dummy_data``` | Dummy data to run GPT-3 fine-tuning pipeline, reproduce plots from `experiment A`` (human participants) and reproduce technical pipelines from ?`experiment B`` (machine classifiers)|
| ```data``` | ... |
| ```plots``` | Plots used in paper|
| ```results``` | Results from machine classifiers |
| ```src``` | All code organised in folders `article-preprocessing`, `gpt3` and `classifiers` |
| ```tokens``` | Empty folder to place `openai-token.txt` and `huggingface-token.txt`|
| ```setup.sh``` | Run to install necessary requirements, packages in virtual environment|
| ```simple_classifier.sh``` | Run to reproduce classifier pipelines|
| ```bert_classifier.sh``` | Run to reproduce BERT pipeline|


## Reproducibility  
Due to constraints with copyright and GDPR, only the test data and the synthetically generated GPT-3 data is uploaded to the GitHub. For all other purposes, `dummy data` is provided. To run the pipelines with the dummy data, follow the instructions in the `Pipeline` section. 

For any other questions regarding the project, please contact the authors. 

## Pipeline 
For this project, Python (version 3.10) and R was used. Python's [venv](https://docs.python.org/3/library/venv.html) needs to be installed for the setup to work.

### Setup 
To install necessary requirements in a virtual environment (`env`), please run the `setup.sh` in the terminal: 
```
bash setup.sh
```

### [1] Fine-Tuning and Text Generation with GPT-3
To fine-tune and/or generate text with GPT-3, please follow the instructions in the [README.md](https://github.com/drasbaek/finetuning-gpt3-danish-news/blob/main/src/gpt3/README.md) located in `src/gpt3`. 

### [2] Experiment A: Analysis of Human Participants  
To run the analysis, please refer to the Rmarkdown `exp-a-analysis.Rmd` in the src folder. 

### [3] Experiment B: Constructing Machine Classifiers
To run BOW and TF-IDF classifiers on dummy data, please run `simple_classifiers.sh` in the terminal:
```
bash simple_classifiers.sh
```

To run the BERT fine-tuning pipeline, type:
```
bash bert_classifier.sh
```
⚠️ `NOTE!` While the fine-tuning of [NbAiLab/nb-bert-large](https://huggingface.co/NbAiLab/nb-bert-large) is done on dummy data, the inference is done with the `actual` fine-tuned classifier on the real `test_data`.

The fine-tuned `BERT` can be accessed from the Hugging Face Hub: [MinaAlmasi/dknes-NB-BERT-AI-classifier](https://huggingface.co/MinaAlmasi/dknews-NB-BERT-AI-classifier). 

## Authors 
For any questions regarding the paper or reproducibility of the project, you can contact us:
<ul style="list-style-type: none;">
  <li><a href="mailto:drasbaek@post.au.dk">drasbaek@post.au.dk</a>
(Anton Drasbæk Schiønning)</li>
    <li><a href="mailto: mina.almasi@post.au.dk"> mina.almasi@post.au.dk</a>
(Mina Almasi)</li>
</ul>
