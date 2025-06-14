import torch
import torch.nn as nn
from models.obssharedwrapper import ObservationSharedWrapper
from models.policyhead import PolicyHead
from models.valuehead import ValueHead

# Wrapper that combines the functionality of all individual components into one, such that given an observation, it will produce:
# 1. Logits over each action (word), i.e. the policy output
# 2. Value prediction, i.e. the value output
# For convenience, none of the components need to be given on construction, i.e. any component set to none will be instantiated with
# default parameters
class WordleActorCritic(nn.Module):
    def __init__(self, observation_encoder=None, shared_encoder=None, policy_head=None, value_head=None):
        super().__init__()
            
        self.obs_shared = ObservationSharedWrapper(observation_encoder, shared_encoder)
        
        if policy_head is not None:
            self.policy_head = policy_head
        else:
            self.policy_head = PolicyHead()
        
        if value_head is not None:
            self.value_head = value_head
        else:
            self.value_head = ValueHead()

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
