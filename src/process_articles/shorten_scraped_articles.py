'''
Script which shortens scraped articles ! Outputs a jsonl for fine-tuning gpt-3, and a 

Written for the paper titled "Fine-tuning GPT-3 for Synthetic Danish News Generation" (Almasi & Schiønning, 2023).
'''
import pathlib

# custom fns 
from modules.preprocessing_fns import shorten_all_articles, jsonl_article_data

# data wrangling
import pandas as pd

def prep_data(datapath, shorten:bool=True): 
    '''
    Prepare scraped data for data pipeline.
    '''

    # read data
    data = pd.read_csv(datapath)

    # randomise data, seed for reproducibility
    data = data.sample(frac=1, random_state=2502) 

    # shorten 
    if shorten == True:
        data["short_text"] = shorten_all_articles(data["tekst"])

    return data 

def main():
    # define paths 
    path = pathlib.Path(__file__) 

    # scraped articles, split into three purposes 
    path_data_1 = path.parents[2] / "dummy_data" / "scraped_data" /  "dummy_scraped_data_1.csv" # note that these three dummy files contain the exact same articles for demonstration purposes only!
    path_data_2 = path.parents[2] / "dummy_data" / "scraped_data" / "dummy_scraped_data_2.csv" # in the actual data, they are splits of the scraped data: used for fine-tuning gpt3, used for generating text with gpt3, used for classifier_data. Note that NO article is used for two purposes i.e., no double-dipping!
    path_data_3 = path.parents[2] / "dummy_data" / "scraped_data" / "dummy_scraped_data_3.csv"

    # read, randomise, shorten data (for fine-tuning + for training data for classifiers)
    data_1, data_2 = prep_data(path_data_1, shorten=True), prep_data(path_data_2, shorten=True)

    # read, randomise data (for creating dataframe with headers and subheaders for a fine-tuned gpt-3 to generate text from) 
    data_3 = prep_data(path_data_3)

    # convert into jsonl for gpt-3 finetune 
    path_out_1 = path.parents[2] / "dummy_data" / "gpt3_data" / "finetune_gpt3_dummy.jsonl"
    jsonl_article_data(data_1, "headers", "sub_header", "short_text", path_out_1)

    # write to csv for classifier 
    path_out_2 = path.parents[2] / "dummy_data" / "scraped_data" / "dummy_scraped_shortened_articles.csv" 
    data_2.to_csv(path_out_2, index=False)

    # write out to csv for prompts that gpt3 can generate from 
    path_out_3 = path.parents[2] / "dummy_data" / "gpt3_data" / "generate_gpt3_dummy_prompts.csv" 
    data_3 = data_3[["headers","sub_header","nwords_headers","nwords_subheader"]] # select only relevant cols 
    data_3.to_csv(path_out_3, index=False)
    
if __name__ == "__main__":
    main()