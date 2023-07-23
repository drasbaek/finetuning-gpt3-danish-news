'''
Script which shortens scraped articles 

Written for the paper titled "Fine-tuning GPT-3 for Synthetic Danish News Generation" (Almasi & Schiønning, 2023).
'''
# custom fns 
from modules.preprocessing_fns import shorten_all_articles, jsonl_article_data

# data wrangling
import pandas as pd

def main():
    # define paths 
    path = pathlib.Path(__file__) 
    path_data = path.parents[2] / "dummy_data" / "dummy_scraped_data.csv"

    # read data 
    data = pd.read_csv(path_data)

    # sample data (randomise)
    data = data.sample(frac=1)

    # shorten
    data["short_text"] = shorten_all_articles(data["tekst"])

    # convert into jsonl for gpt-3 finetune 
    path_out = path.parents[2] / "dummy_data" / "finetune_gpt3_dummy.jsonl"
    jsonl_article_data(data, "headers", "sub_header", "short_text", path_out)