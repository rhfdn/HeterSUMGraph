import random
import multiprocessing
import numpy as np
from torch_geometric.data import Data
import torch

from .DataLoader import DataLoader
from .preprocess_df import preprocess_df
from .create_graph_dataset import create_graph


def merge_graphs(gs):
    x0 = []
    x1 = []
    edge_index_src = []
    edge_index_dst = []
    edge_attr = []
    ptr0 = []
    ptr1 = []

    d_x0 = 0
    start_x0 = 0

    d_x1 = sum([len(g.x[0]) for g in gs])
    start_x1 = d_x1

    ptr0.append(d_x0)
    ptr1.append(d_x1)

    for g in gs:
        x0 = x0 + g.x[0].tolist()
        x1 = x1 + g.x[1].tolist()
        
        eis = g.edge_index[0].clone()
        eid = g.edge_index[1].clone()

        eis[g.edge_index[0] < len(g.x[0])] = d_x0 + eis[g.edge_index[0] < len(g.x[0])]
        eis[g.edge_index[0] >= len(g.x[0])] = d_x1 + eis[g.edge_index[0] >= len(g.x[0])] - len(g.x[0])

        eid[g.edge_index[1] < len(g.x[0])] = d_x0 + eid[g.edge_index[1] < len(g.x[0])]
        eid[g.edge_index[1] >= len(g.x[0])] = d_x1 + eid[g.edge_index[1] >= len(g.x[0])] - len(g.x[0])
        
        edge_index_src = edge_index_src + eis.tolist()
        edge_index_dst = edge_index_dst + eid.tolist()

        assert start_x1 + len(x1) >= max(edge_index_dst) + 1
        
        edge_attr = edge_attr + g.edge_attr.tolist()
        
        d_x0 = start_x0 + len(x0)
        d_x1 = start_x1 + len(x1)

        ptr0.append(d_x0)
        ptr1.append(d_x1)

    x = [torch.tensor(x0, dtype=torch.long),torch.tensor(x1, dtype=torch.long)]
    edge_index = torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    ptr = [torch.tensor(ptr0, dtype=torch.long), torch.tensor(ptr1, dtype=torch.long)]

    assert len(x[0]) + len(x[1]) >= max(edge_index[0]) + 1
    assert len(x[0]) + len(x[1]) >= max(edge_index[1]) + 1

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, ptr=ptr, undirected=True)


def merge_batch(batch):
    idxs = [e["idx"] for e in batch]
    doc_lens = [e["doc_lens"] for e in batch]
    docs = [e["docs"] for e in batch]
    docs = merge_graphs(docs)
    labels = [lbl for e in batch for lbl in e["labels"]]
    return {"idx": idxs, "doc_lens": doc_lens, "docs": docs, "labels": labels}


# Worker (processus)
def worker_data_loader(id_worker, nb_worker, i_queue, o_queue, dataset, batch_size,
                       tfidfs_sent, word_blacklist, remove_unkn_words, self_loop,
                       max_sent_len, glovemgr):
    input = i_queue.get()
    buffer_size = input
    i = 0
    indexes = []
    while input != "stop":
        if input == "new_indexes":
            indexes = i_queue.get()
            i = 0
            input = buffer_size

        while input > 0 and (i * nb_worker + id_worker) * batch_size < len(indexes):
            first = (i * nb_worker + id_worker) * batch_size
            idxs = indexes[first:first+batch_size]
            batch = []

            for id in idxs:
                idx = dataset[id]["idx"]
                docs = create_graph(dataset[id]["docs"], tfidfs_sent["tfidf"][idx],
                                    word_blacklist=word_blacklist,
                                    remove_unkn_words=remove_unkn_words, self_loop=self_loop,
                                    pad_sent=max_sent_len, glovemgr=glovemgr)
                batch.append({"idx": idx, "doc_lens": len(dataset[id]["docs"]), "docs": docs,
                              "labels": dataset[id]["labels"]})
            o_queue.put(merge_batch(batch))
            i += 1
            input -= 1

        input = i_queue.get()
    o_queue.put("end")


class MultiProcessusDataLoader():
    def __init__(self, buffer_size, dataset, batch_size, shuffle,
                 tfidfs_sent, glovemgr, word_blacklist = [], remove_unkn_words = False,
                 self_loop=False, doc_column_name="docs", labels_column_name="labels",
                 is_sep_n=False, remove_stop_word = True, stemming=True, trunc_sent=-1,
                 padding_sent=-1, trunc_doc=-1) -> None:
        assert batch_size > 0
        self.nb_worker = batch_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.dataset = preprocess_df(df=dataset, glovemgr=glovemgr, doc_column_name=doc_column_name,
                                labels_column_name=labels_column_name, is_sep_n = is_sep_n,
                                remove_stop_word = remove_stop_word, stemming=stemming,
                                trunc_sent=trunc_sent, padding_sent=padding_sent, trunc_doc=trunc_doc)
        max_sent_len = max([len(s) for t in self.dataset for s in t["docs"]])
        self.indexes = list(range(len(self.dataset)))

        # Padding last batch if necessary
        if len(self.indexes) % self.batch_size != 0:
            self.indexes = self.indexes + random.sample(self.indexes, self.batch_size - (len(self.indexes) % self.batch_size))

        # Shuffle if necessary
        if self.shuffle:
            random.shuffle(self.indexes)

        self.workers = []
        for num_worker in range(self.nb_worker):
            i_queue = multiprocessing.Queue()
            o_queue = multiprocessing.Queue()
            worker = multiprocessing.Process(target=worker_data_loader,
                                             daemon=True,
                                             args=(num_worker, self.nb_worker, o_queue, i_queue,
                                                   self.dataset, self.batch_size, tfidfs_sent,
                                                   word_blacklist, remove_unkn_words, self_loop,
                                                   max_sent_len, glovemgr))
            worker.start()
            o_queue.put(self.buffer_size)
            self.workers.append((worker, i_queue, o_queue))

    def __del__(self):
        for p in self.workers:
            p[0].terminate()

    def __getitem__(self, idx):
        assert idx >= 0

        if idx == 0:
            self.indexes = list(range(len(self.dataset)))

            # Padding last batch if necessary
            if len(self.indexes) % self.batch_size != 0:
                self.indexes = self.indexes + random.sample(self.indexes, self.batch_size - (len(self.indexes) % self.batch_size))

            # Shuffle if necessary
            if self.shuffle:
                random.shuffle(self.indexes)

            for p in self.workers:
                p[2].put("new_indexes")
                p[2].put(self.indexes)

        if idx >= len(self.indexes) // self.batch_size:
            return self.dataset[:1][2]

        batch = self.workers[idx % self.nb_worker][1].get()
        self.workers[idx % self.nb_worker][2].put(1)

        return batch

    def __len__(self):
        return int(len(self.indexes) / self.batch_size)
