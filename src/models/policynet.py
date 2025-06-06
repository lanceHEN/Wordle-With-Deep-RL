import torch
import torch.nn as nn

# given batched latent vectors from the shared encoder, produces logits over every single word (action)
# which one can softmax over to get prob. dists.
# this was chosen over a more fine-grained embedding-based output because that would require training word embeddings simultaneously, which can be expensive
class PolicyHead(nn.Module):
    
    # Initializes a PolicyHead with the given input latent dimension (should be same as SharedEncoder output), and output dimension
    def __init__(self, hidden_dim=128, vocab_size=14855):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size) # linear

    # given batched latent vectors from the shared encoder, produces logits over every single word (action)
    # which one can softmax over to get prob. dists.
    # made in part with generative AI
    def forward(self, h, valid_indices_batch):
        logits = self.linear(h)
        
        device = h.device # automatically fetch device of hidden vector

        batch_size, vocab_size = logits.shape
        mask = torch.ones(batch_size, vocab_size, dtype=torch.bool, device=device)
        for i, valid_idx in enumerate(valid_indices_batch):
            mask[i, valid_idx] = False

        mask_float = torch.where(mask, float('-inf'), 0.0)
        masked_logits = logits + mask_float

        return masked_logits