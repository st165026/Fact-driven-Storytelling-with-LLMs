import torch
from torch.utils.data import Dataset

class SufficiencyDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.embeddings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item