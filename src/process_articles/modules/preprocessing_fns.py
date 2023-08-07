'''
Functions for shortening articles used for fine-tuning GPT-3 and classification task.

Written for the paper titled "Fine-tuning GPT-3 for Synthetic Danish News Generation" (Almasi & Schiønning, 2023).
'''

# data wrangling
import pandas as pd 

# to shorten
from nltk.tokenize import word_tokenize, TreebankWordDetokenizer

def shorten_article(article: str) -> str:
    '''
    Function to shorten an article to roughly 120 words using nltk (stops at the first natural punctuation after 120 words) 

    Parameters
    article (text to be shortened): str
    '''

    tokens = word_tokenize(article) #use nltk to tokenize the text
    punctuations = [".", "!", "?", ","]
    stops = [".", "!", "?",]

    counter = 0 
    
    short_article = []
    
    for i in range(len(tokens)): 
        if counter > 120 and tokens[i] in stops: # if counter is over 120 & token is a stop, break the loop (to not cut in middle of a sentence)
            short_article.append(tokens[i]) 
            break
        else: 
            if tokens[i] not in punctuations: # count the number of words (not punctuations)
                short_article.append(tokens[i])
                counter += 1
            else: 
                short_article.append(tokens[i]) # add punctuation to the list but not count it

    detokenize_article = TreebankWordDetokenizer().detokenize(short_article) # use nltk to detokenize the text
    detokenize_article = detokenize_article.replace(" .", ".") # fix detokenization errors

    return detokenize_article

def shorten_all_articles(text_column:pd.Series)-> list:
    '''
    Function to shorten all articles in a text column using the shorten_article function

    Parameters
    text_column (column with articles to be shortened): pd.Series
    '''
    short_articles = []
    for article in text_column:
        short_articles.append(shorten_article(article))
    return short_articles

def jsonl_article_data(data, header_col, sub_header_col, short_text_col, outpath):
    ''' 
    Function to create a jsonl file with the following structure:
    prompt: <header> <subheader>
    completion: <short_text>

    Parameters:
    data (pandas dataframe): dataframe with the data
    header_col (str): name of the column with the header
    sub_header_col (str): name of the column with the subheader
    short_text_col (str): name of the column with the shortened text
    outpath (pathlib.Path): full path of outfile 
    '''

    # create prompt column
    data["prompt"] = data[header_col].astype(str) +". "+ data[sub_header_col]

    # rename short_text to completion
    data = data.rename(columns={short_text_col: "completion"})

    # rearrange columns for jsonl format
    data = data[["prompt", "completion"]]


    # prepare for jsonl fine-tuning (prompts ending with "->" and completitons starting with white space " " & ending with "\n")
    data["prompt"] = data["prompt"].astype(str) + " ->"
    data["completion"] = " "+ data["completion"].astype(str) + " \n"

    short_json = data.to_json(outpath, orient='records', lines=True, force_ascii=False) #needs force_ascii=False to keep danish characters