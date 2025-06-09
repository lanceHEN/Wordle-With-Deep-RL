import torch
import torch.nn as nn
from models.obssharedwrapper import ObservationSharedWrapper

# Wrapper that takes in every component of the model architecture into one class
# Given an observation batch, valid indices, and word encodings, produces two outputs:
# 1. Logits over each action (word), i.e. the policy output
# 2. Value prediction, i.e. the value output
class WordleActorCritic(nn.Module):
    def __init__(self, observation_encoder, shared_encoder, policy_head, value_head):
        super().__init__()
        self.obs_shared = ObservationSharedWrapper(observation_encoder, shared_encoder)
        self.policy_head = policy_head
        self.value_head = value_head

    def forward(self, obs_batch, valid_indices_batch, word_matrix):
        h = self.obs_shared(obs_batch)  # [B, hidden_dim]
        logits = self.policy_head(h, valid_indices_batch, word_matrix)  # [B, V]
        values = self.value_head(h).squeeze(-1)  # [B]
        return logits, values