import torch
import torch.nn as nn
'''
cnn_shared_encoder.py:
    Variation of Shared Encoder that uses a small 2D CNN to summarize the Wordle board. Using a CNN lets us share parameters spatially
    so the network learns local patterns regardless of their absolute location. Serves as an alternate method to the FFN.
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
        with relu activation at each step, before flattening and concatenating with the meta vector and passing
        thru an FFN with 2 hidden layers.
    '''
    def __init__(self,
                per_cell_dim: int = 19, # = letter_embed_dim + 3
                conv_channels: tuple = (32, 64),
                hidden_dim: int = 512,
                shared_output_dim: int = 256):
        super().__init__()
        # Convolutional stack over 6 x 5 img
        in_ch = per_cell_dim
        layers = []
        for out_ch in conv_channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        
        self.flatten = nn.Flatten()
        
        # fuses with meta tensor via Multilayer Perceptron (to combine conv features with meta)
        self.fuse = nn.Sequential(
            nn.Linear(in_ch * 6 * 5 + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, shared_output_dim),
            nn.LayerNorm(shared_output_dim),
            nn.ReLU(inplace=False),
        )

    # Given batched 3d tensors representing the game grids (word and feedback) of shape [B, max_guesses, word_length, embed_dim]
    # and batched 1d tensors representing the turn and number of candidates remaining for each batch item of shape [B, 2],
    # produces latent vector representations (for each batch item) with the given output dimension, shared_output_dim, via CNN. This is done
    # by applying convolutions on the grid, before flattening, concatenating with the meta vector, and passing through an FFN.
    def forward(self, grid, meta):
        '''
        Returns a fused latent vector for each batch element.
        '''
        # grid: [B, 6, 5, D]

        # rearrange so that guesses become channels (like color channels in images)
        x = grid.permute(0, 3, 2, 1).contiguous()  # [B, D, 5, 6]
        
        flat = self.flatten(self.conv(x)) # in_ch * 6 * 5

        fused = torch.cat([flat, meta], dim = -1) # in_ch * 6 * 5 + 2
        return self.fuse(fused) # [B, shared_output_dim]
                    
                     



        
