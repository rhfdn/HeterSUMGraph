# HeterSUMGraph
This git repository presents and compares HeterSUMGraph and variants using GATConv, GATv2Conv and a combination of HeterSUMGraph and SummaRuNNer (using HeterSUMGraph as a sentence encoder).

The datasets are CNN-DailyMail and NYT50.  
  
paper: [HeterSUMGraph](https://arxiv.org/pdf/2004.12393.pdf)  

## Clone project
```bash
git clone https://github.com/Baragouine/HeterSUMGraph.git
```

## Enter into the directory
```bash
cd HeterSUMGraph
```

## Create environnement
```bash
conda create --name HeterSUMGraph python=3.9
```

## Activate environnement
```bash
conda activate HeterSUMGraph
```

## Install dependencies
```bash
pip install -r requirements.txt
```

## Install nltk data
To install nltk data:
  - Open a python console.
  - Type ``` import nltk; nltk.download()```.
  - Download all data.
  - Close the python console.

## Convert NYT zip to NYT50 json and preprocessing it
  - Download raw NYT zip from [https://catalog.ldc.upenn.edu/LDC2008T19](https://catalog.ldc.upenn.edu/LDC2008T19) to `data/`  
  - Run `00-00-convert_nyt_to_json.ipynb` (convert zip to json).
  - Run `00-01-nyt_filter_short_summaries.ipynb` (keep summary with 50 distinct word only).
  - Run `00-02-compute_nyt_labels.ipynb` (comput labels).
  - Run ```python scripts/compute_tfidf_dataset.py -input data/nyt_corpus_LDC2008T19_50.json -output data/nyt50_dataset_tfidf.json``` (compute tfidfs for whole dataset).
  - Run ```python scripts/compute_tfidf_dataset.py -input data/nyt_corpus_LDC2008T19_50.json -output data/compute_tfidf_sent_dataset.json``` (compute tfidfs for each document).
  - Run `00-03-split_NYT50.ipynb` (split NYT50 to train, val, test).
  
## CNN-DailyMail preprocessing
  - Follow CNN-DailyMail preprocessing instruction on: [https://github.com/Baragouine/SummaRuNNer/tree/master](https://github.com/Baragouine/SummaRuNNer/tree/master).
  - After labels computed, run ```00-03-merge_cnn_dailymail.ipynb``` to merge CNN-DailyMail to one json file.
  - Run python scripts/compute_tfidf_dataset.py -input data/cnn_dailymail.json -output data/cnn_dailymail_dataset_tfidf.json``` (compute tfidfs for whole dataset).
  - Run python scripts/compute_tfidf_dataset.py -input data/cnn_dailymail.json -output data/cnn_dailymail_sent_tfidf.json``` (compute tfidfs for each document).






