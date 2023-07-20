'''
Script for fine-tuning GPT-3 for generating Danish news articles


Written for the paper titled "Fine-tuning GPT-3 for Synthetic Danish News Generation" (Almasi & Schiønning, 2023).

NB! Please note that the current script finetunes "text-davinci-002", but that this will be deprecated on the 4th of January 2024
Read more on: https://openai.com/blog/gpt-4-api-general-availability
'''

# utils 
import pathlib
import json 

# gpt3 finetuning
import openai 

def import_token(token_path:pathlib.Path):
    '''
    Import OPENAI token from token.txt in specified path.

    Args
        token_path: path of token.txt file 
    '''
    # get token from txt
    with open(token_path) as f:
        openai.api_key = f.read()

def create_file(file_path:pathlib.Path(), target_filename:str):
    '''
    Create finetune file IF the file (with the target_filename) does not exist already. 
    Note that this function ONLY works if an OPENAI API TOKEN has been loaded prior to calling the function

    Args
        file_path: path of the file you want to create 
        target_filename: the name of the file that you want to create
    
    Returns 
        file_id: id of the file (used within fine-tuning)
    '''

    # retrieve files 
    files_OPENAI = openai.File.list()

    # parse the JSON data into a dictionary
    files_dict = json.loads(str(files_OPENAI))

    # iterate over files dict 
    matching_file = None
    for file_info in files_dict["data"]:
        if file_info["filename"] == target_filename:
            matching_file = file_info
            break # break when matching file is found ! 
        
    # extract 
    if matching_file is not None:
        file_id = matching_file["id"]
   
    else:
        print("Creating File ...")
        file = openai.File.create(
            file = open(str(file_path), "rb"),
            purpose = "fine-tune", 
                user_provided_filename = target_filename
            )

        file_id = file["id"]


    return file_id

def create_finetune(training_file_id, target_finetune, n_epochs, model):
    '''
    Create a fine-tune IF a fine-tune with the specified suffix does not exist already.

    Args:
        training_file_id: The ID of the training file to use in the fine-tune.
        target_finetune: The name to identify the specific fine-tune.
        n_epochs: The number of fine-tuning epochs.
        model: The model to use for fine-tuning.

    Returns:
        fine_tune_id: The ID of the existing or newly created fine-tune.
    '''

    # Retrieve a list of existing fine-tunes
    existing_finetunes = openai.FineTune.list()

    # parse the data into a dictionary
    existing_finetunes_dict = dict(existing_finetunes)

    # Check if a fine-tune with the specified suffix exists
    existing_finetune_id = None
    for finetune_info in existing_finetunes_dict["data"]:
        try:
            if finetune_info["fine_tuned_model"].endswith(target_finetune):
                existing_finetune_id = finetune_info["id"]
                break
        except:
            print("Error, finetune has no name")

    # If the fine-tune with the specified suffix does not exist, create a new one
    if existing_finetune_id is None:
        print("Creating Fine-Tune ...")
        finetune = openai.FineTune.create(
            training_file=training_file_id,
            n_epochs=n_epochs,
            model=model,
            suffix=target_finetune, 
            batch_size=2,
            learning_rate_multiplier=0.2,
            prompt_loss_weight=0.01, 
        )
        finetune_id = finetune["id"]
    else:
        finetune_id = existing_finetune_id
    
    return finetune_id

def main(): 
    # define paths 
    path = pathlib.Path(__file__) 
    path_token = path.parents[2] / "tokens" / "openai_token.txt"

    # setup token
    import_token(path_token)

    # define filename, filepath 
    path_dummydata = path.parents[2] / "dummy_data" / "finetune_gpt3_dummy.jsonl" 
    target_filename = "finetune_gpt3_dummy.jsonl"

    # create file, if file does not exist already 
    file_id = create_file(path_dummydata, target_filename)

    # create finetune, if finetune with the specific suffix does not exist
    finetune_id = create_finetune(file_id, "finetune_dummy", 2, "ada")

    # retrieve exact finetune 
    finetune = openai.FineTune.retrieve("ft-CTJaj0GmC2SZcdJTyGGbt9dE")

    # print 
    print(finetune)


if __name__ == "__main__":
    main()
