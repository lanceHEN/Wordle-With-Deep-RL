import torch
import torch.nn as nn

class CNNSharedEncoder(nn.Module):
    '''
    encodes a batch of Wordle observations w/ a convolutional variant of ObservationEncoder
    output:
        board_embeds: [B, hidden_dim]: latent board embedding of 6 x 5 grid
        meta_tensor: [B, 2]: (turn_num, valid_frac)
    '''
    def __init__(self,
        letter_encoder: nn.Embedding, # learned letter embedder fron observationencoder
        conv_channel = (32, 64, 128), # output channels per conv blocks
        hidden_dim: int = 256, # final board embedding size
        vocab_size: int = 14_855, # answer vocab size
    ):
        super().__init__()
        self.letter_encoder = letter_encoder
        self.feedback_dim = 3 # one-hot size for gray, yellow, green
        self.vocab_size = vocab_size

        # ConvNet [C, H = 6, W = 5]
        in_ch = self.letter_encoder.embedding_dim + self.feedback_dim
        layers = []
        for out_ch in conv_channel:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size = 3, padding = 1),
                nn.ReLU(inplace = True),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers) # keeps shape ([B, last_ch, 6, 5])

        # self read_out that fixes hidden_dim
        self.readout = nn.Sequential(
            nn.Flatten(), # [B, last_ch*6*5]
            nn.Linear(in_ch * 6 * 5, hidden_dim),
            nn.ReLU(inplace=True),
        )
    def forward(self, obs_batch):
        
