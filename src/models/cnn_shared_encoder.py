import torch
import torch.nn as nn
from models.ffn_shared_encoder import FFNSharedEncoder

'''
cnn_shared_encoder.py:
    Variation of Shared Encoder that uses a small 2D CNN to summarize the Wordle board. Using a CNN lets us share parameters spatially
    so the network learns local patterns regardless of their absolute location. Serves as an alternate method to the standalone FFN.
    This in particular treats each guess as its own channel.
    We do not pool because the per-channel dimensions are too small.
    expects output of ObservationEncoder [B, 6, 5, per_cell_dim]
    expects meta [B, 2]
    returns [B, shared_output_dim]
'''
class CNNSharedEncoder(nn.Module):
    '''
    Encodes Wordle observations via Conv Net --> MLP fusion
    Given:
        grid [B, 6, 5, embed_dim]
        meta [B, 2]
    Return:
        A fused representation using a set number of convolutions on the grid with specified channels 
        with relu activation and layer normalization at each step, before flattening and concatenating with the meta vector
        and passing thru an FFN with 2 hidden layers. Because the FFN is of equivalent design to that in the FFN shared encoder,
        we actually reuse the FFN forward method here.
    '''
    def __init__(self,
                in_channels: int = 6,
                conv_channels: tuple = (32, 64),
                hidden_dim: int = 512,
                shared_output_dim: int = 256):
        super().__init__()
        # Convolutional stack over 5 x 19 img
        in_ch = in_channels
        layers = []
        for out_ch in conv_channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        
        self.flat = nn.Flatten()
        
        input_dim = in_ch * 5 * 19 + 2
        
        self.ffn = FFNSharedEncoder(input_dim, hidden_dim, shared_output_dim)

    # Given a batched tensor representing the game grids (word and feedback) of shape [B, max_guesses, word_length, embed_dim]
    # and a batched tensor representing the turn and number of candidates remaining for each batch item of shape [B, 2],
    # produces a batch of latent vector representations (for each batch item) with the given output dimension, shared_output_dim, via CNN. This is done
    # by applying convolutions on the grid, before flattening, concatenating with the meta vector, and passing through an FFN.
    def forward(self, grid, meta):
        '''
        Returns a fused latent vector for each batch element.
        '''
        # grid: [B, 6, 5, D]
        
        conv = self.conv(grid)

        return self.ffn(conv, meta) # [B, shared_output_dim]