import torch
import numpy as np

class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per, args):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.unlabeled = args.unlabeled

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        #distractive learning
        # for i_batch in range(self.n_batch):
        #     batch = []
        #     classes = torch.randperm(len(self.m_ind))[:self.n_cls+1]
        #     for c in classes[:-1]:
        #         l = self.m_ind[c]
        #         pos = torch.randperm(len(l))[:self.n_per]
        #         if c==classes[-2]:
        #             # get distractive sample
        #             l_d = self.m_ind[classes[-1]]
        #             pos_d =  torch.randperm(len(l_d))[0]
        #             pos[self.unlabeled] = pos_d
        #         batch.append(l[pos])
        #     batch = torch.stack(batch).t().reshape(-1)
        #     yield batch
        
        # inductive learning
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
            