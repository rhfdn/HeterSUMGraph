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
  - Run ```python scripts/compute_tfidf_dataset.py -input data/cnn_dailymail.json -output data/cnn_dailymail_dataset_tfidf.json``` (compute tfidfs for whole dataset).
  - Run ```python scripts/compute_tfidf_dataset.py -input data/cnn_dailymail.json -output data/cnn_dailymail_sent_tfidf.json``` (compute tfidfs for each document).

## Embeddings
For training you must use glove 300 embeddings, they must have the following path: `data/glove.6B/glove.6B.300d.txt`

## Training
For CNN-DailyMail max doc len is 100 sentences, not 50 as in the paper (same max doc len as SummaRuNNer to compare both).
  - `01-train_HeterSUMGraph_CNN_DailyMail.ipynb`: paper model on CNN-DailyMail
  - `02-train_HeterSUMGraph_NYT50.ipynb`: paper model on NYT50
  - `03-train_HeterSUMGraph_CNN_DailyMail_TG_GATConv.ipynb`: HeterSUMGraph with torch_geometric GATConv layer on CNN-DailyMail.
  - `04-train_HeterSUMGraph_NYT50_TG_GATConv.ipynb`: HeterSUMGraph with torch_geometric GATConv layer on NYT50.
  - `05-train_HeterSUMGraph_CNN_DailyMail_TG_GATv2Conv.ipynb`: HeterSUMGraph with torch_geometric GATv2Conv layer on CNN-DailyMail.
  - `06-train_HeterSUMGraph_NYT50_TG_GATv2Conv.ipynb`: HeterSUMGraph with torch_geometric GATv2Conv layer on NYT50.
  - `07-train_HSGRNN_CNN_DailyMail_TG_GATv2Conv.ipynb`: HeterSUMGraph with torch_geometric GATv2Conv layer + SummaRuNNer on CNN-DailyMail.
  - `08-train_HSGRNN_NYT50_TG_GATv2Conv.ipynb`: HeterSUMGraph with torch_geometric GATv2Conv layer + SummaRuNNer on NYT50.

## Result
### NYT50 (limited-length ROUGE Recall)
| model | ROUGE-1 | ROUGE-2 | ROUGE-L |  
|:-:    |:-:      |:-:      |:-:      |  
| HeterSUMGraph (Wang) | 46.89 | 26.26 | 42.58 |
| HeterSUMGraph (ours) | 45.5 &plusmn; 0.0 | 24.2 &plusmn; 0.0 | 34.1 &plusmn; 0.0 |
| HSG GATConv | 45.4 &plusmn; 0.0 | 24.2 &plusmn; 0.0 | 34.0 &plusmn; 0.0 |
| HSG GATv2Conv | **47.2 &plusmn; 0.0** | **26.5 &plusmn; 0.0** | **35.5 &plusmn; 0.0\*** |
| HSGRNN GATv2Conv | 46.9 &plusmn; 0.0 | 26.3 &plusmn; 0.0 | 35.3 &plusmn; 0.0 |
  
*: maybe the ROUGE-L have changed in the rouge library I use.


