import torch
import torch.nn as nn


# given batched latent vectors from the shared encoder, produces predicted word embeddings, which can then be
# compared with all given word embeddings for all guess words (via dot prod.) to produce logits (masked for valid word indices),
# which one can softmax over to get prob. dists.
# this was chosen over a simpler output head immediately producing logits over every word because embeddings allow for a finer-grained comparison between words
class PolicyHead(nn.Module):
    
    # Initializes a PolicyHead with the given input latent dimension (should be same as SharedEncoder output), and output dimension
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 5*26) # linear

    # given batched latent vectors from the shared encoder, produces predicted word embeddings, which can then be
    # compared with all word embeddings for all guess words (via dot prod.) to produce logits (masked for valid word indices),
    # which one can softmax over to get prob. dists.
    # made in part with generative AI
    def forward(self, h, valid_indices_batch):
        # word_encodings: [130, vocab_size]
        # valid_indices_batch: list of list of valid indices per environment
        device = h.device
        
        pred_embed = self.linear(h)

        return pred_embed