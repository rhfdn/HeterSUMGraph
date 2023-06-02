import random
import numpy as np
from torch_geometric.data import Data
import torch

class DataLoader():
    def __init__(self, dataset, batch_size=1, shuffle=False):
        assert batch_size > 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lidx = list(range(len(self.dataset)))

        # Padding last batch if necessary
        if len(self.lidx) % self.batch_size != 0:
            self.lidx = self.lidx + random.sample(self.lidx, self.batch_size - (len(self.lidx) % self.batch_size))

        # Shuffle if necessary
        if self.shuffle:
            random.shuffle(self.lidx)

    def __getitem__(self, idx):
        assert idx >= 0
        if idx == 0:
            self.lidx = list(range(len(self.dataset)))

            # Padding last batch if necessary
            if len(self.lidx) % self.batch_size != 0:
                self.lidx = self.lidx + random.sample(self.lidx, self.batch_size - (len(self.lidx) % self.batch_size))

            # Shuffle if necessary
            if self.shuffle:
                random.shuffle(self.lidx)
        if (idx >= len(self.lidx) / self.batch_size):
            return self.dataset[len(self.dataset)]
        idxs = self.lidx[idx*self.batch_size:idx*self.batch_size+self.batch_size]
        batch = [self.dataset[i] for i in idxs]

        batch = self.merge(batch)

        return batch

    def __len__(self):
        return int(len(self.lidx) / self.batch_size)

    def merge(self, batch):
        idxs = [e["idx"] for e in batch]
        doc_lens = [e["doc_lens"] for e in batch]
        docs = [e["docs"] for e in batch]
        docs = self.merge_graphs(docs)
        labels = [lbl for e in batch for lbl in e["labels"]]
        return {"idx": idxs, "doc_lens": doc_lens, "docs": docs, "labels": labels}

    def merge_graphs(self, gs):
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