# %%
from torch_geometric.data import Dataset, Data
import torch
import pandas as pd
import argparse
import re

from utils.GloveMgr import GloveMgr
from utils.preprocess_text import preprocess_text
from utils.preprocess_df import preprocess_df

# %%
# Parse args if script mode
def create_graph(doc, tfidf_sent, glovemgr, pad_sent, word_blacklist = []):
  words = list(set([id for line in doc for id in line if glovemgr.i2w(id) not in word_blacklist]))
  sents = doc

  edge_index_src = []
  edge_index_dst = []
  edge_attr = []

  for s in range(len(doc)):
    for w_in_s in range(len(doc[s])):
      for w in range(len(words)):
        if doc[s][w_in_s] == words[w]:
          edge_index_src.append(s)
          edge_index_dst.append(w)
          target_word = glovemgr.i2w(words[w])
          if target_word in tfidf_sent:
            edge_attr.append(tfidf_sent[s][w])
          else:
            edge_attr.append(0)

  # pad sentences
  #max_sent_len = max([len(sent) for sent in doc])
  sents_padded = [sent + [0 for i in range(pad_sent - len(sent))] for sent in sents]

  words = torch.tensor(words, dtype=torch.float)
  sents = torch.tensor(sents_padded, dtype=torch.float)
  edge_index = torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
  edge_attr = torch.tensor(edge_attr, dtype=torch.float)

  data = Data(x=[words, sents], edge_index=edge_index, edge_attr=edge_attr, undirected=True)

  return data

#%%
class GraphDataset(Dataset):
  def __init__(self, dataset) -> None:
    super().__init__()
    self.dataset = dataset

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    return self.dataset[index]

# %%
def create_graph_dataset(df, tfidfs_sent, glovemgr, word_blacklist = [], doc_column_name="docs", labels_column_name="labels", is_sep_n=False, remove_stop_word = True, stemming=True, trunc_sent=-1, padding_sent=-1, trunc_doc=-1):
  res = []

  df = preprocess_df(df=df, glovemgr=glovemgr, doc_column_name=doc_column_name, labels_column_name=labels_column_name, is_sep_n = is_sep_n, remove_stop_word = remove_stop_word, stemming=stemming, trunc_sent=trunc_sent, padding_sent=-padding_sent, trunc_doc=trunc_doc)
  
  max_sent_len = max([len(s) for t in df for s in t["docs"]])

  for i in range(len(df)):
    res.append({"idx": df[i]["idx"], "doc_lens": len(df[i]["docs"]), "docs": create_graph(df[i]["docs"], tfidfs_sent[i], word_blacklist=word_blacklist, pad_sent=max_sent_len, glovemgr=glovemgr), "labels": df[i]["labels"]})
  
  return GraphDataset(res)
