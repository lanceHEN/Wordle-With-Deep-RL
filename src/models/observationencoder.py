import torch
import torch.nn as nn
import torch.nn.functional as F

# given a particular observation, produces numerical representations friendly for input to a neural network
# in particular, produces:
# 1. a [6 x 5 x letter_embed_dim + 3] tensor, storing letter (embeddings from the given LetterEncoder)
# and feedback (one hot) data for every position in the game (filled or unfilled)
# 2. a [2] vector storing the current turn and number of candidate words remaining (divided by total vocab size of word (not guesses) list)
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

    # given a particular observation, produces numerical representations friendly for input to a neural network
    # in particular, produces:
    # 1. a [6 x 5 x letter_embed_dim + 3] tensor, storing letter (embeddings from the given LetterEncoder)
    # and feedback (one hot) data for every position in the game (filled or unfilled)
    # 2. a [2] vector storing the current turn and number of candidate words remaining (divided by total vocab size of word (not guesses) list)
    # made in part with generative AI
    def forward(self, obs):
        device = next(self.parameters()).device # automatically get from params
        
        # guesses - [6, 5, D]
        encoded_grid = []  # build as a list first to avoid in-place ops
 
        for turn, (word, feedback) in enumerate(obs["feedback"]): # over each guess
            row = []
            for i, (letter, fb_str) in enumerate(zip(word, feedback)): # over each letter
                fb_idx = {"gray": 0, "yellow": 1, "green": 2}[fb_str]

                letter_vec = self.letter_encoder(letter) # letter embedding
                fb_one_hot = F.one_hot(torch.tensor(fb_idx, device=device), num_classes=3).float() # one hot fb embedding

                row.append(torch.cat([letter_vec, fb_one_hot]))  # [embed_dim]
                
            # Pad the row to length 5 if necessary
            while len(row) < 5:
                row.append(torch.zeros(self.embed_dim, device=device))

            encoded_grid.append(torch.stack(row))  # [5, D]
            
        # Pad to 6 guesses if fewer
        while len(encoded_grid) < 6:
            encoded_grid.append(torch.zeros(5, self.embed_dim, device=device))

        encoded_grid = torch.stack(encoded_grid)  # [6, 5, D]

        # additional features
        turn_scalar = obs["turn_number"]
        remaining_scalar = len(obs["valid_indices"]) / self.vocab_size # divide by vocab size for scalability

        encoded_meta = torch.tensor(
            [turn_scalar, remaining_scalar],
            dtype=torch.float32,
            device=device
        )  # shape: [2]

        return encoded_grid, encoded_meta  # [6, 5, D], [2]