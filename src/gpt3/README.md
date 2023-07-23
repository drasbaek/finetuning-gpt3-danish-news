# Reproducing the GPT-3 Pipeline
Follow the steps below to reproduce the GPT-3 pipeline with `dummy data`. 

⚠️ `NOTE!` Both the fine-tuning and text generation costs money. The pricing is available at ___, and we encourage you to set usage limits to avoid unecessary spending. 

## SETUP: OPENAI TOKEN
Before being able to fine-tune and generate text, you need to acquire an API access token: 
1. Generate the token at ____
2. Create a `txt` called `openai-token.txt` in the `tokens` folder
3. Paste the generated token within this `txt` file 

## Fine-Tuning GPT-3 
To fine-tune GPT-3 with `dummy data`, type in the terminal:
```
python src/gpt3/finetune_gpt3.py
```
⚠️ `NOTE!`  The code is designed to only fine-tune ONE model ONCE, but run the code with care regardless!! Check your `usage` each time you run the code to avoid spending money unintentionally.

## Generating text with GPT-3 
To generate text with GPT-3, type in the terminal: 
```
python src/gpt3/generate_gpt3.py
```
⚠️ `NOTE!` Each time you run the code to generate text, you will be charged a small fee!







