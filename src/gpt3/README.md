# Reproducing the GPT-3 Pipeline
Follow the steps below to reproduce the GPT-3 pipeline with `dummy data`. Please note that the current script finetunes  "text-davinci-002", but this will be deprecated on the 4th of January 2024.
You can read more on about this at https://openai.com/blog/gpt-4-api-general-availability.

⚠️ `NOTE!` Both the fine-tuning and text generation costs money. The pricing is available at https://openai.com/pricing. We encourage you to set usage limits within your OpenAI account to avoid unecessary spending. 

## SETUP: OPENAI TOKEN
Before being able to fine-tune and generate text, you need to acquire an API access token: 
1. Generate the token at https://platform.openai.com/account/api-keys 
2. Create a `txt` called `openai-token.txt` in the `tokens` folder
3. Paste the generated token within this `txt` file 

## Fine-Tuning GPT-3 
To fine-tune GPT-3 with `dummy data`, type in the terminal:
```
python src/gpt3/finetune_gpt3.py
```
⚠️ `NOTE!`  The code is designed to only fine-tune ONE model ONCE, but run the code with care regardless!! Check your `usage` each time you run the code to avoid spending money unintentionally.

## Generating text with GPT-3 
To generate text with GPT-3 with `dummy prompts`, type in the terminal: 
```
python src/gpt3/generate_gpt3.py
```
⚠️ `NOTE!` Each time you run the code to generate text, you will be charged a small fee!







