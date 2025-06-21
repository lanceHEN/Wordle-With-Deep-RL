import torch.nn as nn


# Implementation of the policy head that works with batching. Takes in batched latent vectors from the Shared Encoder and 
# produces a query vector for each batch item, which can be used to obtain probability distributions.
# This was chosen over a simpler output head immediately producing logits over every word because embeddings allow for a finer-grained comparison between words,
# particularly words having the same prefixes or suffixes being treated similarly.
class PolicyHead(nn.Module):
    
    # Initializes a PolicyHead with the given input latent dimension (should be same as SharedEncoder output).
    def __init__(self, input_dim=256):
        super().__init__()
        self.linear = nn.Linear(input_dim, 5*26) # linear

    # Given a batch of latent vectors from the shared encoder, produces a query vector for each batch item, which can then be
    # multiplied with all word embeddings for all guess words (via dot prod.) to produce logits (masked for valid word indices),
    # which one can softmax over to get prob. dists.
    def forward(self, h):
        query = self.linear(h)

        return query
