# Article Preprocessing
To run the article preprocessing, including the cleaning and shortening of dummy scraped articles, please run the code (with `env` activated):

```
python src/process_articles/shorten_scraped_articles.py
```

To shorten and prepare `DaNewsRoom` corpora, you can request the data by contacting the authors (see [danielvarab/da-newsroom](https://github.com/danielvarab/da-newsroom)). Then place the data in the `data` folder and run the script by typing (with `env` activated);

```
python src/process_articles/shorten_danewsroom.py
```
