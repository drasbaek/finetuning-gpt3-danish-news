# Fine-tuning GPT-3 for Synthetic Danish News Generation
This repository contains the code written for the paper titled, **"Fine-tuning GPT-3 for Synthetic Danish News Generation" (Almasi & Schiønning, 2023)**. 

The project involves fine-tuning GPT-3 to produce synthetic news articles and evaluating the model using both human participants and machine classifiers in a binary classification task. To read more, please refer to our paper. 

## Structure 
The repository is structured as such:


## Reproducibility  
Due to constraints with copyright and GDPR, only the test data is uploaded to the GitHub. For the participant data and the training data for classifiers, **dummy data** is provided.

To run the pipelines with the dummy data, follow the instructions in the `Pipeline` section. 

Contact the authors for more information. 

## Pipeline 
For this project, Python (version 3.7) and R was used. Python's [venv](https://docs.python.org/3/library/venv.html) needs to be installed for the setup to work.

### Setup 
To install necessary requirements in a virtual environment (`env`), please run the `setup.sh` in the terminal: 
```
bash setup.sh
```

### Fine-Tuning GPT-3

### Experiment A: Analysis of Human Participants  
To run the analysis, please refer to the R markdowns in the `results` folder. 

### Experiment B: Constructing Machine Classifiers
To run BOW and TFIDF classifiers, please run `simple_classifiers.sh` in the terminal:
```

```

To train the BERT classifier ... 
```

```

## Authors 
For any questions regarding the paper or reproducibility of the project, you can contact us:
<ul style="list-style-type: none;">
  <li><a href="mailto:drasbaek@post.au.dk">drasbaek@post.au.dk</a>
(Anton Drasbæk Schiønning)</li>
    <li><a href="mailto: mina.almasi@post.au.dk"> mina.almasi@post.au.dk</a>
(Mina Almasi)</li>
</ul>
