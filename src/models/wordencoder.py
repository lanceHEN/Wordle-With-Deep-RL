import torch
import torch.nn as nn

# given a word, produces a rich embedding of shape [letter_embed_dim] representing both the letters and their positions
# this is useful since the policy network produces a word embedding of equivalent length when predicting a word to choose
# class made in part with generative AI
class WordEncoder(nn.Module):
    def __init__(self, letter_encoder, use_positional=True):
        super().__init__()
        self.letter_encoder = letter_encoder  # a LetterEncoder
        self.use_positional = use_positional
        self.word_embed_dim = self.letter_encoder.letter_embed_dim

        if self.use_positional:
            self.position_encoder = nn.Embedding(5, self.word_embed_dim)

    # given a word, produces a rich embedding of shape [letter_embed_dim] representing both the letters and their positions
    # this is useful since the policy network produces a word embedding of equivalent length when predicting a word to choose
    def forward(self, word):  # word: str
        
        # Determine device from letter_encoder parameters
        device = next(self.parameters()).device
        
        letter_embeds = [self.letter_encoder(c) for c in word]  # list of [letter_embed_dim] tensors
        letter_embeds = torch.stack(letter_embeds, dim=0)  # [5, letter_embed_dim]

        if self.use_positional:
            position_ids = torch.arange(5, device=device)  # [5]
            pos_embeds = self.position_encoder(position_ids)  # [5, letter_embed_dim]
            embeddings = letter_embeds + pos_embeds
        else:
            embeddings = letter_embeds

        word_embed = embeddings.sum(dim=0)  # [letter_embed_dim]
        return word_embed