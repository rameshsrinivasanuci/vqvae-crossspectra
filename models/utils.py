from torch.utils.data import Dataset
import torch


class UnsupervisedEEGDataset(Dataset):
    def __init__(self, X): # X: Input data of shape (B, C, T)

        # lets just keep it as numpy for now lol
        self.X = X

    def __len__(self): # to obtain the number of x in X.
        return len(self.X) 

    def __getitem__(self, idx): # to index an x in X.
        return torch.from_numpy(self.X[idx]).float() 