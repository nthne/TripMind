import torch
from torch.utils.data import Dataset
from src.utils import encode

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab):
   
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = encode(self.texts[idx], self.vocab)
        y = self.labels[idx]
        return torch.tensor(x), torch.tensor(y)
