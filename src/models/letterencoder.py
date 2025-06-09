import torch
import torch.nn as nn
import torch.nn.functional as F

# given a letter, produces rich embeddings with the given dimension
# this is useful for computing word embeddings for the policy network
# as well as for encoding the state by taking embeddings for each guessed letter (and concat with feedback)
# note although this is esssentially an nn.Embedding, it is still its own class in case one would like to
# produce word embeddings using letter embeddings
class LetterEncoder(nn.Module):
    
    # initializes a LetterEncoder with the given letter embedding dimension
    def __init__(self, letter_embed_dim=16):
        super().__init__()
        self.letter_embed_dim = letter_embed_dim

        # letter embeddings (learnable)
        self.letter_embed = nn.Embedding(26, letter_embed_dim)

    # given a letter index ('a' being 0, 'z' being 25), produces its embeddings representation as a torch tensor of dim [self.letter_embed_dim]
    def forward(self, letter_idx):
        device = self.letter_embed.weight.device # fetch device of embedding weights for consistency
        letter_vec = self.letter_embed(letter_idx.to(device))
        
        return letter_vec