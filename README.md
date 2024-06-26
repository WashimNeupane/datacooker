# datacooker
Curates a mix of datasets for llm training. Based on dolma library (See https://arxiv.org/abs/2402.00159)

# Dataset Subset Downloader

This script allows you to download a subset of a Hugging Face dataset based on a specified language (e.g., English). It utilizes the Dolma library for processing Wikipedia data and filtering the dataset during the loading process.

## Requirements

- [wikiextractor](https://github.com/santhoshtr/wikiextractor.git)
- requests
- smart_open
- tqdm
- dolma
- datasets

You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```

## Dolma Scripts to Run

First, you need to extract Wikipedia data and create a Wikipedia mix using Dolma:

```bash
python make_wikipedia.py \
  --output data/wikipedia \
  --date {latestDate in YYYYMMDD format}  \
  --lang simple \     
  --processes 16 \
  --overwrite
````

Then, use the Dolma script to mix the Wikipedia data:

```bash
dolma -c config/wikipeida-mix.yaml mix --processes 16
```

You can subsample or oversample with the probability 0-1 and >1 respectively. For oversample, the oversampled_p= 1/p. Run the code as follows

```bash
python sampling.py  -s 'data/dir/location1/*.gz' 'data/dir/location2/*.gz'   -p 1.65   -d data/mixed   -n 16
```

## Notes

For private datasets, you may need to log in via the Hugging Face CLI before downloading.




