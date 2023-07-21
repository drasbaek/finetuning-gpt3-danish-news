'''
Script for generating Danish News Articles using a fine-tuned GPT-3

Written for the paper titled "Fine-tuning GPT-3 for Synthetic Danish News Generation" (Almasi & Schiønning, 2023).
'''

# utils 
import pathlib

# gpt3
import openai

# custom functions 
from modules.gpt3_helpers import import_token, find_existing_finetune_id

def generate_text(finetune_id, prompt):
    pass

def main(): 
    # define paths 
    path = pathlib.Path(__file__) 
    path_token = path.parents[2] / "tokens" / "openai_token.txt"

    # setup token
    import_token(path_token)

    # Retrieve a list of existing fine-tunes
    existing_finetunes = openai.FineTune.list()

    # Find the ID of the existing fine-tune with the specified suffix
    existing_finetune_id = find_existing_finetune_id(existing_finetunes, "finetune-dummy")

    print(existing_finetune_id)


if __name__ == "__main__":
    main()