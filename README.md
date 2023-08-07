# Fine-tuning GPT-3 for Synthetic Danish News Generation
This repository contains the code written for the paper titled, **"Fine-tuning GPT-3 for Synthetic Danish News Generation" (Almasi & Schiønning, 2023)**. 

The project involved fine-tuning GPT-3 to produce synthetic news articles in Danish and evaluating the model in binary classification tasks. The evaluation relied on both human participants (A) and machine classifiers (B).

To read the details of this evaluation, please refer to (Almasi & Schiønning, 2023). 

## Reproducibility  
Due to constraints with copyright and GDPR, only the test data and the synthetically generated GPT-3 data is uploaded to this GitHub repository. For all other purposes, `dummy` data is provided to reproduce the pipelines (see also `Project Structure`). To run any of the pipelines, follow the instructions in the `Pipeline` section. 

For any other questions regarding the project, please [contact](https://github.com/drasbaek/finetuning-gpt3-danish-news/tree/main#authors) the authors. 

## Pronect Structure 
The repository is structured as such: 

| <div style="width:120px"></div>| Description |
|---------|:-----------|
| ```dummy_data``` | Dummy data to run GPT-3 fine-tuning pipeline, reproduce plots from `experiment A` (human participants) and reproduce technical pipelines from `experiment B` (machine classifiers)|
| ```data``` | Contains the `96` test articles used in both Experiment A and B (i.e., for evaluating both human participants and machine detectors) and the `609` articles generated by GPT-3 for fine-tuning BERT. |
| ```plots``` | Plots used in paper|
| ```results``` | ACTUAL Results from machine classifiers |
| ```src``` | All code organised in folders `article-preprocessing`, `gpt3` and `classifiers` |
| ```tokens``` | Empty folder to place `openai_token.txt` (for fine-tuning and generating text with GPT-3) and `hf_token.txt` (for pushing model to Hub, OPTIONAL!!!) |
| ```setup.sh``` | Run to install necessary requirements, packages in virtual environment|
| ```simple_classifier.sh``` | Run to reproduce classifier pipelines|
| ```bert_classifier.sh``` | Run to reproduce BERT pipeline|

Please note again that `results`, `plots` and `data` contains actual data pertaining to (Almasi & Schiønning, 2023) while `dummy_results` and `dummy_data` are not real data files. 

## Pipeline 
For this project, Python (version 3.10) and R was used. Python's [venv](https://docs.python.org/3/library/venv.html) needs to be installed for the setup to work.

### Setup 
To install necessary requirements in a virtual environment (`env`), please run the `setup.sh` in the terminal: 
```
bash setup.sh
```

### [0] Article Preprocessing
To reproduce the article preprocessing on dummy data, follow the instructions in ___ 

### [1] Fine-Tuning and Text Generation with GPT-3
To fine-tune and/or generate text with GPT-3, follow the instructions in the [README.md](https://github.com/drasbaek/finetuning-gpt3-danish-news/blob/main/src/gpt3/README.md) located in `src/gpt3`. 

### [2] Experiment A: Analysis of Human Participants  
To run the analysis, please refer to the Rmarkdown `exp-a-analysis.Rmd` in the src folder. 

### [3] Experiment B: Constructing Machine Classifiers
To construct the machine classifiers (`BOW`, `TF-IDF`, `fine-tuned BERT`), follow the instructions in the [README.md](https://github.com/drasbaek/finetuning-gpt3-danish-news/blob/main/src/classifiers/README.md) located in `src/classifiers`.

⚠️ `NOTE!` While the fine-tuning of [NbAiLab/nb-bert-large](https://huggingface.co/NbAiLab/nb-bert-large) is done on dummy data, the inference is done with the `actual` fine-tuned classifier on the real `test_data`.

The fine-tuned `BERT` can be accessed from the Hugging Face Hub: [MinaAlmasi/dknews-NB-BERT-AI-classifier](https://huggingface.co/MinaAlmasi/dknews-NB-BERT-AI-classifier).

## Authors 
For any questions regarding the paper or reproducibility of the project, you can contact us:
<ul style="list-style-type: none;">
  <li><a href="mailto:drasbaek@post.au.dk">drasbaek@post.au.dk</a>
(Anton Drasbæk Schiønning)</li>
    <li><a href="mailto: mina.almasi@post.au.dk"> mina.almasi@post.au.dk</a>
(Mina Almasi)</li>
</ul>
