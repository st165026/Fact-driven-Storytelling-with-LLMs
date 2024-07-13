import torch
from torch import nn
from transformers import RobertaModel
import torch.nn.functional as F


class ContrastiveRoberta(nn.Module):

    def __init__(self, freeze_layers=True):

        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')

        if freeze_layers:
            # freeze all layers
            for param in self.roberta.parameters():
                param.requires_grad = False

            # unfreeze the last 3 layers
            for param in self.roberta.encoder.layer[-3:].parameters():
                param.requires_grad = True

    def forward(self, arg_tokens, pos_tokens, neg_tokens):

        # extract the embeddings for [CLS] token of the argument, positive, and negative evidences
        arg_emb = self.roberta(**arg_tokens).last_hidden_state[:, 0, :]
        pos_emb = self.roberta(**pos_tokens).last_hidden_state[:, 0, :]

        batch_size, num_negatives, seq_length = neg_tokens['input_ids'].shape

        # flatten 3D tensor for neg_evidences --> (batch_size * num_negatives, seq_length)
        neg_input_ids = neg_tokens['input_ids'].view(-1, seq_length)
        neg_attention_mask = neg_tokens['attention_mask'].view(-1, seq_length)

        # extract the embeddings for [CLS] token and reshape back
        neg_embs = self.roberta(input_ids=neg_input_ids, attention_mask=neg_attention_mask).last_hidden_state[:, 0, :].view(batch_size, num_negatives, -1)

        return arg_emb, pos_emb, neg_embs
    

class SupportScoreModel(nn.Module):

    def __init__(self, contrastive_model):

        super().__init__()
        self.contrastive_model = contrastive_model

    def forward(self, arg_tokens, evidence_tokens):
        
        arg_emb = self.contrastive_model.roberta(**arg_tokens).last_hidden_state[:, 0, :]
        evi_emb = self.contrastive_model.roberta(**evidence_tokens).last_hidden_state[:, 0, :]
        
        # compute cosine similarity of the arg_emb and evi_emb as the support score
        support_score = F.cosine_similarity(arg_emb, evi_emb, dim=-1)

        return support_score