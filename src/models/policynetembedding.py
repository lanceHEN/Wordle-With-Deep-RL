import torch
import torch.nn as nn

# given batched latent vectors from the shared encoder, produces predicted word embeddings, which are then
# compared with all word embeddings for all guess words (via dot prod.) to produce logits (masked for valid word indices),
# which one can softmax over to get prob. dists.
# this was chosen over a simpler output head immediately producing logits over every word because embeddings allow for a finer-grained comparison between words
class PolicyHead(nn.Module):
    
    # Initializes a PolicyHead with the given input latent dimension (should be same as SharedEncoder output), word embedding dimension, and device
    def __init__(self, hidden_dim=128, word_embed_dim=16, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(hidden_dim, word_embed_dim) # linear

    # given batched latent vectors from the shared encoder, produces predicted word embeddings, which are then
    # compared with all word embeddings for all guess words (via dot prod.) to produce logits (masked for valid word indices),
    # which one can softmax over to get prob. dists.
    # this was chosen over a simpler output head immediately producing logits over every word because embeddings allow for a finer-grained comparison between words
    # made in part with generative AI
    def forward(self, h, valid_indices_batch, word_embeddings):
        query = self.linear(h)
        query = query.clone().detach().requires_grad_(True)  # break all ties, re-enable grad

        word_embeddings_T = word_embeddings.transpose(0, 1).contiguous().clone().detach()
        scores = (query @ word_embeddings_T)

        batch_size, vocab_size = scores.shape
        mask = torch.ones(batch_size, vocab_size, dtype=torch.bool, device=scores.device)
        for i, valid_idx in enumerate(valid_indices_batch):
            mask[i, valid_idx] = False

        mask_float = torch.where(mask, float('-inf'), 0.0)
        masked_logits = scores + mask_float

        return masked_logits