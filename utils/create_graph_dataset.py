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
def create_graph(doc, tfidf_sent, glovemgr, pad_sent, word_blacklist = [], remove_unkn_words = False, self_loop=False):
  words = list(set([id for line in doc for id in line if glovemgr.i2w(id) not in word_blacklist]))
  sents = doc

  # remove unknown word and padding
  if remove_unkn_words:
    words = [w for w in words if w != 1 and w != 0]
    sents = [[w for w in sent if w != 1 and w != 0] for sent in sents]

  edge_index_src = []
  edge_index_dst = []
  edge_attr = []

  for s in range(len(sents)):
    list_w_in_s = list(set(sents[s]))
    for w_in_s in range(len(list_w_in_s)):
      for w in range(len(words)):
        if list_w_in_s[w_in_s] == words[w]:
          edge_index_src.append(len(words) + s)
          edge_index_dst.append(w)
          edge_index_src.append(w)
          edge_index_dst.append(len(words) + s)
          target_word = glovemgr.i2w(words[w])
          if target_word in tfidf_sent:
            edge_attr.append(tfidf_sent[s][w])
            edge_attr.append(tfidf_sent[s][w])
          else:
            edge_attr.append(0)
            edge_attr.append(0)

  if self_loop:
    edge_index_src = edge_index_src + list(range(len(words) + len(sents)))
    edge_index_dst = edge_index_dst + list(range(len(words) + len(sents)))
    edge_attr = edge_attr + [0 for _ in range(len(words) + len(sents))]

  # pad sentences
  #max_sent_len = max([len(sent) for sent in doc])
  sents_padded = [sent + [0 for i in range(pad_sent - len(sent))] for sent in sents]

  words = torch.tensor(words, dtype=torch.long)
  sents = torch.tensor(sents_padded, dtype=torch.long)
  edge_index = torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
  edge_attr = torch.tensor(edge_attr, dtype=torch.float)

  data = Data(x=[words, sents], edge_index=edge_index, edge_attr=edge_attr, undirected=True)

  assert len(data.x[0]) + len(data.x[1]) >= max(edge_index[0], default=0) + 1
  assert len(data.x[0]) + len(data.x[1]) >= max(edge_index[1], default=0) + 1

  return data

#%%
class GraphDataset:#(Dataset):
  def __init__(self, dataset) -> None:
    #super(GraphDataset, self).__init__()
    self.dataset = dataset

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    return self.dataset[index]

#%%
def create_graph_dataset(df, tfidfs_sent, glovemgr, word_blacklist = [], remove_unkn_words = False, self_loop=False, doc_column_name="docs", labels_column_name="labels", is_sep_n=False, remove_stop_word = True, stemming=True, trunc_sent=-1, padding_sent=-1, trunc_doc=-1):
  res = []

  df = preprocess_df(df=df, glovemgr=glovemgr, doc_column_name=doc_column_name, labels_column_name=labels_column_name, is_sep_n = is_sep_n, remove_stop_word = remove_stop_word, stemming=stemming, trunc_sent=trunc_sent, padding_sent=padding_sent, trunc_doc=trunc_doc)
  
  max_sent_len = max([len(s) for t in df for s in t["docs"]])

  for i in range(len(df)):
    idx = df[i]["idx"]
    docs = create_graph(df[i]["docs"], tfidfs_sent["tfidf"][idx], word_blacklist=word_blacklist, remove_unkn_words=remove_unkn_words, self_loop=self_loop, pad_sent=max_sent_len, glovemgr=glovemgr)

    if len(docs.x[0]) == 0 or len(docs.x[1]) == 0:
      continue
    
    res.append({"idx": idx, "doc_lens": len(df[i]["docs"]), "docs": docs, "labels": df[i]["labels"]})
  
  return GraphDataset(res)
