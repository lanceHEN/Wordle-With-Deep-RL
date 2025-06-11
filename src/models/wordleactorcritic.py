import torch
import torch.nn as nn
from models.obssharedwrapper import ObservationSharedWrapper

# Wrapper that takes in every component of the model architecture into one class
# Given an observation batch, and word encodings, produces two outputs:
# 1. Logits over each action (word), i.e. the policy output
# 2. Value prediction, i.e. the value output
class WordleActorCritic(nn.Module):
    def __init__(self, observation_encoder, shared_encoder, policy_head, value_head):
        super().__init__()
        self.obs_shared = ObservationSharedWrapper(observation_encoder, shared_encoder)
        self.policy_head = policy_head
        self.value_head = value_head

    # Given an observation batch, and word encodings, produces two outputs:
    # 1. Logits over each action (word), i.e. the policy output
    # 2. Value prediction, i.e. the value output
    def forward(self, obs_batch, word_encodings):
        device = next(self.parameters()).device
        
        valid_indices_batch = [obs["valid_indices"] for obs in obs_batch]
        h = self.obs_shared(obs_batch)  # [B, hidden_dim]
        query = self.policy_head(h, valid_indices_batch)  # [B, V]
        
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
        
        values = self.value_head(h).squeeze(-1)  # [B]
        return masked_logits, values