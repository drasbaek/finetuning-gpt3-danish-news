'''
Script for fine-tuning GPT-3 for generating Danish news articles


Written for the paper titled "Fine-tuning GPT-3 for Synthetic Danish News Generation" (Almasi & Schiønning, 2023).
'''

import openai 

def import_token(token_path):
        # get token from txt
        with open(token_path) as f:
            openai.api_key = f.read()

def main(): 
    # define paths 
    path = pathlib.Path(__file__) 
    path_dummydata = path.parents[2] / "dummy_data" / "finetune_gpt_3_dummy.jsonl" 

    # create file 
    openai.File.create(
        file=open(path_dummydata),
        purpose="fine-tune"
    )

    # retrieve file id 
    files = openai.File.list()

    # print 
    print(files)

if __name__ == "__main__":
    main()