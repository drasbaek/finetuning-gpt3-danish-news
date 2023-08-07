'''
Script which shortens the DaNewsRoom articles. 

Note that this file cannot be run without downloading the DaNewsRoom dataset 
(by contacting the owner of the repo: https://github.com/danielvarab/da-newsroom)

Written for the paper titled "Fine-tuning GPT-3 for Synthetic Danish News Generation" (Almasi & Schiønning, 2023).
'''

# utils
import pathlib 
import re 

# to load data
import gzip 

# data wrangling 
import pandas as pd 
from nltk import sent_tokenize
from modules.preprocessing_fns import shorten_all_articles

def load_data(datapath:pathlib.Path): 
    '''
    Load and filter DaNewsRoom data to only include TV2 articles with the SAMFUND topic 

    Args
        datapath: path to data 

    Returns
        samf_data: dataframe with filtered DaNewsRoom data
    '''

    # read JSONL files 
    with gzip.open(datapath) as file:
        data = pd.read_json(file, lines=True)

    # filter data only to include TV2 
    data = data[data["site"]=="nyhederne.tv2.dk"]
    data = data.reset_index(drop=True)

    # create urls 
    urls = data["url"]

    # filter only "samfund" articles
    samfund_urls = []

    for i in range(len(urls)):
        match = re.search(r"\bsamfund\b", urls[i])
        if match is not None: 
            samfund_urls.append(urls[i])

    # create dataframe for samf articles URLS 
    samfund_urls_df = pd.DataFrame(samfund_urls, columns = ["url"])

    # create samf_data 
    samf_data = data.copy()
    samf_data = samf_data.merge(samfund_urls_df, on="url", how="inner")

    # select columns
    samf_data = samf_data[["url", "title", "text"]]

    # rename title column
    samf_data = samf_data.rename(columns = {'title':'header'})

    return samf_data 


def clean_data(samf_data, savepath=None): 
    '''
    Clean (already filtered) DaNewsRoom data 

    Args
        samf_data: dataframe with TV2 "samfund" articles 
        savepath: defaults to none (if not none, saves CSV files to specified path)

    Returns
        samf_data: cleaned dataframe 
    '''

    # remove \n\n characters
    for i in range(len(samf_data["text"])):
        samf_data["text"][i] = samf_data["text"][i].replace("\n\n", " ")

    # add create sub_header column from the first sentences in text  
    sub_headers = []

    for i in range(len(samf_data["text"])):
        sentence_tokenize = sent_tokenize(samf_data["text"][i]) # tokenize but in sentences
        sub_headers.append(sentence_tokenize[0]) # append the first sentence to sub_headers lst 

    # add to dataframe 
    samf_data["sub_header"] = sub_headers

    # create short_text column (article body by removing the sub_header from DaNewsRoom's text col )
    short_text = []

    for i in range(len(samf_data["text"])):
        n_remove = len(samf_data["sub_header"][i]) + 1 #remove sub_header from text
        short_text.append(samf_data["text"][i][n_remove:])

    # add article body (without heading) to overall dataframe 
    samf_data["short_text"] = short_text

    # shorten articles 
    samf_data["short_text"] = shorten_all_articles(samf_data["short_text"])

    # save if savepath is not None
    if savepath is not None: 
        # subset for both to use both for GPT-3 purposes (fine-tune/generate) and for training classifiers (human category)
        samf_data_gpt3 = samf_data[:428]
        samf_data_human = samf_data[428:856]

        # save 
        samf_data_human.to_csv(savepath/"clean_danewsroom_human.csv")
        samf_data_gpt3.to_csv(savepath/"clean_danewsroom_gpt3.csv")
    
    return samf_data

def main(): 
    # define paths 
    path = pathlib.Path(__file__) 
    path_data = path.parents[2] / "data" / "danewsroom.jsonl.gz"
    path_save = path.parents[2] / "data" 

    # load data 
    print("Loading data ...")
    data = load_data(path_data)

    # clean data 
    print("Cleaning data ...")
    cleaned_data = clean_data(data, path_save)


if __name__ == "__main__":
    main()