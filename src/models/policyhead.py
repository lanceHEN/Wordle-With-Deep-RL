import torch
import torch.nn as nn


# given batched latent vectors from the shared encoder, produces predicted word embeddings, which are then
# compared with all given word embeddings for all guess words (via dot prod.) to produce logits (masked for valid word indices),
# which one can softmax over to get prob. dists.
# this was chosen over a simpler output head immediately producing logits over every word because embeddings allow for a finer-grained comparison between words
class PolicyHead(nn.Module):
    
    # Initializes a PolicyHead with the given input latent dimension (should be same as SharedEncoder output), and output dimension
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 5*26) # linear

    # given batched latent vectors from the shared encoder, produces predicted word embeddings, which are then
    # compared with all word embeddings for all guess words (via dot prod.) to produce logits (masked for valid word indices),
    # which one can softmax over to get prob. dists.
    # made in part with generative AI
    def forward(self, h, valid_indices_batch, word_encodings):
        # word_encodings: [130, vocab_size]
        # valid_indices_batch: list of list of valid indices per environment
        device = h.device
        
        query = self.linear(h)

        # Compute logits via dot product with all word embeddings
        logits = query @ word_encodings.T  # [B, vocab_size]

        # Create a mask for invalid indices
        batch_size, vocab_size = logits.shape
        mask = torch.ones(batch_size, vocab_size, dtype=torch.bool, device=device)
        for i, valid_idx in enumerate(valid_indices_batch):
            mask[i, valid_idx] = False  # these are VALID indices, so mask should be False here

        # Apply mask
        mask_float = torch.where(mask, float('-inf'), 0.0)
        masked_logits = logits + mask_float

        return masked_logits