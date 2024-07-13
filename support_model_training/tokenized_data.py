import torch
from torch.utils.data import Dataset

class TokenizedDataset(Dataset):

    """
    Parameters
    ----------
    arg_tokens: Dictionary containing 2D tensors for 'input_ids' and 'attention_mask' for arguments
    pos_tokens: Dictionary containing 2D tensors for 'input_ids' and 'attention_mask' for positive evidences
    neg_tokens: Dictionary containing 3D tensors for 'input_ids' and 'attention_mask' for negative evidences (num_samples, num_neg_for_each=5, 512)
    """

    def __init__(self, arg_tokens, pos_tokens, neg_tokens):

        super().__init__()
        self.arg_tokens = arg_tokens
        self.pos_tokens = pos_tokens
        self.neg_tokens = neg_tokens

    def __len__(self):
        return len(self.arg_tokens['input_ids'])

    def __getitem__(self, idx):
        
        arg_item = {key: val[idx] for key, val in self.arg_tokens.items()}
        pos_item = {key: val[idx] for key, val in self.pos_tokens.items()}
        neg_items = {key: val[idx] for key, val in self.neg_tokens.items()} 

        return arg_item, pos_item, neg_items