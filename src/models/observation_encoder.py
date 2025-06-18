import torch
import torch.nn as nn
import torch.nn.functional as F

# Batching implementation of the Observation Encoder. Given batched observations (as a list), produces numerical representations friendly for inputs to a neural network
# in particular, produces:
# 1. grid tensor: a [B x 6 x 5 x letter_embed_dim + 3] tensor, storing letter (embeddings from the given LetterEncoder)
# and feedback (one hot) data for every position in the game (filled or unfilled), for each in the batch
# 2. meta vector: a [B x 2] tensor storing the current turn and number of candidate words remaining (divided by total vocab size of word/guess list) for the particular
# game in the batch
class ObservationEncoder(nn.Module):
    
    # Initializes an ObservationEncoder with the given LetterEncoder and vocab size
    def __init__(self, letter_encoder, vocab_size=14855):
        super().__init__()
        self.vocab_size = vocab_size

        # letter embeddings (learnable)
        self.letter_encoder = letter_encoder

        # feedback will be one-hot, size 3: ("gray", "yellow", "green")
        self.feedback_dim = 3

        # final per-cell embedding size = letter + feedback dim
        self.embed_dim = self.letter_encoder.letter_embed_dim + self.feedback_dim

    # given batched observations (as a list), produces numerical representations friendly for inputs to a neural network
    # in particular, produces:
    # 1. grid tensor: a [B x 6 x 5 x letter_embed_dim + 3] tensor, storing letter (embeddings from the given LetterEncoder)
    # and feedback (one hot) data for every position in the game (filled or unfilled), for each in the batch
    # 2. meta vector: a [B x 2] tensor storing the current turn and number of candidate words remaining (divided by total vocab size of word/guess list) for the particular
    # game in the batch
    # made in part with generative AI
    def forward(self, obs_batch):
        device = next(self.parameters()).device
        batch_size = len(obs_batch)
    
        letter_tensor = torch.zeros((batch_size, 6, 5), dtype=torch.long, device=device)
        feedback_tensor = torch.zeros((batch_size, 6, 5, self.feedback_dim), dtype=torch.float32, device=device)
        meta_vector = torch.zeros((batch_size, 2), dtype=torch.float32, device=device)
    
        fb_map = {"gray": 0, "yellow": 1, "green": 2}
    
        for b, obs in enumerate(obs_batch):
            for t, (word, feedback) in enumerate(obs["feedback"]):
                for i, (char, fb) in enumerate(zip(word, feedback)):
                    letter_tensor[b, t, i] = ord(char) - ord('a')
                    feedback_tensor[b, t, i, fb_map[fb]] = 1.0
            meta_vector[b, 0] = obs["turn_number"]
            meta_vector[b, 1] = len(obs["valid_indices"]) / self.vocab_size

        letter_embs = self.letter_encoder(letter_tensor)  # [B, 6, 5, embed_dim]
        grid_tensor = torch.cat([letter_embs, feedback_tensor], dim=-1)  # [B, 6, 5, embed_dim + 3]
        return grid_tensor, meta_vector
