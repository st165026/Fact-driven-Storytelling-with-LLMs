import pandas as pd
from torch.utils.data import Dataset


class ArgEvi(Dataset):

    # read data and convert it to list format

    def __init__(self, file_path):

        super().__init__()
        data = pd.read_csv(file_path)

        data = data.values.tolist()
        self.data = data
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

#file_path = 'pos_neg_pairs_train.csv'
#dataset = ArgEvi(file_path)
