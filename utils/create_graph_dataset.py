# %%
from torch_geometric.data import Dataset, Data
import torch
import pandas as pd
import argparse
import re

from utils import GloveMgr
from utils import preprocess_text
from utils import preprocess_df

# %%
# Parse args if script mode
def create_graph(doc, tfidf_sent, glovemgr):
  words = list(set([(id for id in line) for line in doc]))
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

  words = torch.tensor(words, dtype=torch.float)
  sents = torch.tensor(sents, dtype=torch.float)
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
def create_graph_dataset(df, tfidfs_sent, glovemgr, doc_column_name="docs", labels_column_name="labels", is_sep_n=False, remove_stop_word = True, stemming=True, trunc_sent=-1, padding_sent=-1, trunc_doc=-1):
  res = []
  df2 = preprocess_df(df=df, glovemgr=glovemgr, doc_column_name=doc_column_name, labels_column_name=labels_column_name, is_sep_n = is_sep_n, remove_stop_word = remove_stop_word, stemming=stemming, trunc_sent=trunc_sent, padding_sent=-padding_sent, trunc_doc=trunc_doc)
  for i in range(len(df2)):
    res.append(create_graph(df2[i]["docs"], tfidfs_sent[i], glovemgr))
