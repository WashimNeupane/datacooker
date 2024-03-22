# datacooker
Curates a mix of datasets for llm training. 

# Dataset Subset Downloader

This script allows you to download a subset of a Hugging Face dataset based on a specified language (e.g., English). It utilizes the Dolma library for processing Wikipedia data and filtering the dataset during the loading process.

## Requirements

- [wikiextractor](https://github.com/santhoshtr/wikiextractor.git)
- requests
- smart_open
- tqdm
- dolma

You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```

##Dolma Scripts to Run
First, you need to extract Wikipedia data and create a Wikipedia mix using Dolma:

```bash
python make_wikipedia.py \
  --output wikipedia \
  --date 20240320 \
  --lang simple \
  --processes 16 \
  --overwrite
````

Then, use the Dolma script to mix the Wikipedia data:

```bash
dolma -c wikipeida-mix.yaml mix --processes 16
```

Datasets
For certain private datasets, you may need to log in via the Hugging Face CLI before downloading.



