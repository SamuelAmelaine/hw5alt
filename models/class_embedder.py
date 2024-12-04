import torch
import torch.nn as nn
import math

class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, cond_drop_rate=0.1):
        super().__init__()
        
        # Implement class embedding layer
        self.embedding = nn.Embedding(n_classes, embed_dim)
        
        self.cond_drop_rate = cond_drop_rate
        self.num_classes = n_classes

    def forward(self, x):
        b = x.shape[0]
        
        if self.cond_drop_rate > 0 and self.training:
            mask = torch.bernoulli(torch.ones(b) * self.cond_drop_rate).to(x.device)
            null_ids = torch.zeros_like(x) + self.num_classes  # last embedding is reserved for null token
            x = torch.where(mask.bool().unsqueeze(1), null_ids, x)
        
        c = self.embedding(x)
        return c