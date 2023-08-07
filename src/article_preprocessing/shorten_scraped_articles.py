'''
Script which shortens scraped articles 

Written for the paper titled "Fine-tuning GPT-3 for Synthetic Danish News Generation" (Almasi & Schiønning, 2023).
'''
import pathlib

# custom fns 
from modules.preprocessing_fns import shorten_all_articles, jsonl_article_data

# data wrangling
import pandas as pd

def main():
    # define paths 
    path = pathlib.Path(__file__) 
    path_data_1 = path.parents[2] / "dummy_data" / "scraped_data" /  "dummy_scraped_data_1.csv" # note that these two dummy files contain the exact same articles for demonstration purposes only!
    path_data_2 = path.parents[2] / "dummy_data" / "scraped_data" / "dummy_scraped_data_2.csv" # in the actual data, they are splits of the scraped data: used for fine-tuning gpt3, used for generating text with gpt3, used for classifier_data. Note that NO article is used for two purposes i.e., no double-dipping!

    # read data 
    data_1 = pd.read_csv(path_data_1)
    data_2 = pd.read_csv(path_data_2)

    # randomise data
    data_1 = data_1.sample(frac=1)
    data_2 = data_2.sample(frac=1)

    # shorten
    data_1["short_text"] = shorten_all_articles(data_1["tekst"])
    data_2["short_text"] = shorten_all_articles(data_2["tekst"])

    # convert into jsonl for gpt-3 finetune 
    path_out_1 = path.parents[2] / "dummy_data" / "finetune_gpt3_dummy.jsonl"
    jsonl_article_data(data_1, "headers", "sub_header", "short_text", path_out_1)

    # write to csv for classifier 
    path_out_2 = path.parents[2] / "dummy_data" / "scraped_data" / "scraped_shortened_articles.csv" 
    data_2.to_csv(path_out_2, index=False)
    
if __name__ == "__main__":
    main()