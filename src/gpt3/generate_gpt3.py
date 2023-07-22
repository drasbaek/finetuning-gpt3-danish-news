'''
Script for generating Danish News Articles using a fine-tuned GPT-3

Written for the paper titled "Fine-tuning GPT-3 for Synthetic Danish News Generation" (Almasi & Schiønning, 2023).
'''

# utils 
import pathlib

# data wrangling
import pandas as pd 

# gpt3
import openai

# custom functions 
from modules.gpt3_helpers import import_token, find_existing_finetune_id

def prepare_data(data_path):
    # read data
    data = pd.read_csv(data_path)

    # create prompt column (according to OPENAI recs)
    data["prompt"] = data["headers"].astype(str) +". "+ data["sub_header"] +" ->"

    return data 

def generate_text_from_prompt(finetune_mdl, prompt):
    '''
    Generate text from a single prompt with a fine-tuned model. 

    Args
        finetune_mdl: NAME of the fine-tuned model 
        prompt: input text (the text that the model should continue)

    Returns
        completion_txt: output text 
    '''

    completion = openai.Completion.create(
        model=finetune_mdl,
        prompt=prompt, 
        stop = "\n", 
        temperature = 0,
        max_tokens=400,
        best_of=1,
        frequency_penalty=0.2,
        presence_penalty=0.2,
    )

    # return only the text of the completion
    completion_txt = completion.choices[0].text

    return completion_txt

def generate_text_from_data(finetune_mdl, data):
    '''
    Generate text from data with a "prompt" column. 

    Args
        finetune_mdl: NAME of the fine-tuned model 
        data: data with a prompt column

    Returns 
        data: original dataframe with a new "completions" column

    '''
    completion_lst = []

    for prompt in data["prompt"]:
        completion = generate_text_from_prompt(finetune_mdl, prompt)
        completion_lst.append(completion)

    # Add the completions list as a new column "completions" in the original dataframe
    data["completions"] = completion_lst

    return data

def main(): 
    # define paths 
    path = pathlib.Path(__file__) 
    path_token = path.parents[2] / "tokens" / "openai_token.txt"
    path_data = path.parents[2] / "dummy_data" / "generate_gpt3_dummy_prompts.csv"

    # setup token
    import_token(path_token)

    # Retrieve a list of existing fine-tunes
    existing_finetunes = openai.FineTune.list()

    # Find the ID, MDL of the existing fine-tune with the specified suffix
    existing_finetune_id, existing_finetune  = find_existing_finetune_id(existing_finetunes, "finetune-dummy")

    # load data with prompts
    data = prepare_data(path_data)

    # create completions 
    completions_data = generate_text_from_data(existing_finetune, data)

    # save data
    path_outfile = path.parents[0] / "dummy_synthetic_generations.csv"
    completions_data.to_csv(path_outfile, index=False)

    # print 
    print(completions_data)

if __name__ == "__main__":
    main()