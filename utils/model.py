import torch.nn as nn

from utils.constants import EMBED_DIMENSION, EMBED_MAX_NORM

class CBOW_Model(nn.Module):
    def __init__(self, vocab_size:int):
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = EMBED_DIMENSION,
            max_norm = EMBED_MAX_NORM,
        )

        self.linear = nn.Linear(
            in_features = EMBED_DIMENSION,
            out_features = vocab_size,
        )
    
    def forward(self, inputs_):
        x = self.embeddings(inputs_) # [batch_size, vocab_size, EMBED_DIM]
        x = x.mean(axis=1) # [batch_size, EMBED_DIM]
        x = self.linear(x)
        return x

class SkipGram_Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = EMBED_DIMENSION,
            max_norm = EMBED_MAX_NORM,
        )

        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features = vocab_size,
        )
    
    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x