import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):

    """
    Compute the adapted NT-Xent loss for contrastive learning to pull the positive (argument, evidence) pair closer while push other negative pairs apart

    Parameters
    ----------
    temperature: scalar value
                A temperature scaling factor
    anchor: Tensor of shape (batch_size, feature_dim)
            The argument embeddings
    positive: Tensor of shape (batch_size, feature_dim)
            The embeddings of positive evidences for the argument
    negatives: Tensor of shape (batch_size, num_negatives=5, feature_dim)
            The embeddings of negative evidences. Each positive evidence embedding matches 5 negative evidence embeddings
    """


    def __init__(self, temperature):

        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    
    def forward(self, anchor, positive, negatives):

        # normalization to eliminate the effect of embedding length
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negatives = F.normalize(negatives, dim=2)

        # tensor dimensions are aligned to compare positive and negative similarity
        positive_similarity = F.cosine_similarity(anchor, positive, dim=1).unsqueeze(1) 
        # bf: batch_size + feature_dim, bnf: batch_size + num_negatives + feature_dim, bn: batch_size(row) * num_negatives(col)
        # dot product between anchor and each negative in the batch
        negatives_similarity = torch.einsum('bf,bnf->bn', anchor, negatives)

        # concatenate positive and negative similarities to one tensor
        logits = torch.cat([positive_similarity, negatives_similarity], dim=1) / self.temperature
        # create target labels for cross-entropy loss
        # the positive class for each example in the batch is at first column (index 0) of the logits tensor
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=anchor.device)
        
        # computes the probability of the class at index 0 being the correct class. 
        loss = F.cross_entropy(logits, targets)
        
        return loss
        