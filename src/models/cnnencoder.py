import torch
import torch.nn as nn

class CNNSharedEncoder(nn.Module):
    '''
    expects output of ObservationEncoder [B, 6, 5, per_cell_dim]
    expects [B, 2]
    returns [B, output_dim]
    '''
    def __init__(self,
                 per_cell_dim: int = 19, # = letter_embed_dim + 3
                 conv_channels: tuple = (32, 64, 128),
                 board_hidden_dim: int = 256,    # after flattening convmap
                 first_hidden_dim: int = 512,
                 output_dim: int = 256):
        super().__init__()

        # ConvNet [C, H = 6, W = 5]
        in_ch = per_cell_dim
        layers = []
        for out_ch in conv_channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers) # 6×5 resolution

        # readout to the fixed board embedding
        self.readout = nn.Sequential(
            nn.Flatten(), # [B, in_ch * 6 * 5]
            nn.Linear(in_ch * 6 * 5, board_hidden_dim),
            nn.LayerNorm(board_hidden_dim),
            nn.ReLU(inplace=True),
        )
                     



        
