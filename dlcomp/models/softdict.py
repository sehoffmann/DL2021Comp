import torch
from torch import nn

from .layers import SoftDictionary
from .autoencoder import EncoderBlock, DecoderBlock


class DictionaryAutoencoder(nn.Module):

    def __init__(self, **kwargs):
        super(DictionaryAutoencoder, self).__init__()

        self.kernel = 3
        self.activation = nn.ReLU() #activation_from_config(kwargs.pop('activation'))

        self.bn = True
        self.residual = True

        blocks = [64, 128, 256,256,256]#kwargs.pop('blocks')
        layers_per_block = 2#kwargs.pop('layers_per_block')


        in_c = 3
        out_c = 3

        # Input Shape: 3 x 96 x 96
        size = 96
        n_blocks = len(blocks)
        self.hidden_res = size // (2**n_blocks)
        self.hidden_dim = blocks[-1]
        self.hidden_features = self.hidden_dim * self.hidden_res * self.hidden_res


        # Encoder
        in_channels = in_c
        self.encoders = nn.ModuleList()
        for n_features in blocks:
            block = EncoderBlock(
                in_channels,
                n_features,
                layers_per_block,
                self.kernel,
                self.activation,
                residual=self.residual,
                bn=self.bn
            )
            self.encoders.append(block)
            in_channels = n_features

        self.dictionary = SoftDictionary(6*6*128, 3*3*256, 1024)

        # Decoder
        self.decoders = nn.ModuleList()
        for n_features, out_features in zip([128, 128, 128, 64], [128, 128, 64, 3]):
            block = DecoderBlock(
                n_features, 
                out_features,
                layers_per_block,
                self.kernel,
                self.activation,
                use_skip_convs=False,
                residual=self.residual,
                bn=self.bn
            )
            self.decoders.append(block)

        self.out_act = nn.Sigmoid()


    def forward(self, x):
        out = x

        # Encoder
        for enc_layer in self.encoders:
            out, skips = enc_layer(out)
        
        out = self.dictionary(out.flatten(start_dim=1))
        out = out.view(-1, 128,6,6)

        # Decoder
        for dec_layer in self.decoders:            
            out = dec_layer(out, None)
        
        out = self.out_act(out)

        return out
        


