import torch
from torch import nn

from dlcomp.config import activation_from_config
from .layers import ConvBnAct, LinearBnAct, UpsamplingConv, SelfAttention2D, ResidualBlock

class SimpleAutoencoder(nn.Module):

    def __init__(self, **kwargs):
        super(SimpleAutoencoder, self).__init__()

        self.single_channel = kwargs.pop('single_channel', False)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 64 * 12 * 12)

        self.decoder = nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1 if self.single_channel else 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        out = self.encoder(x)
        out = out.view(-1, 64 * 12 * 12 )
        out = self.fc1(out)
        out = self.fc2(out)
        out = out.view(-1, 64,12,12)
        out = self.decoder(out)

        if self.single_channel:
            out = out.repeat(1,3,1,1)

        return out



class Autoencoder(nn.Module):

    def __init__(self, **kwargs):
        super(Autoencoder, self).__init__()

        self.kernel = kwargs.pop('kernel', 3)
        self.activation = activation_from_config(kwargs.pop('activation'))

        self.skip_connections = kwargs.pop('skip_connections')
        self.bn = kwargs.pop('bn')
        self.grayscale = kwargs.pop('grayscale')
        self.residual = kwargs.pop('residual')

        blocks = kwargs.pop('blocks')
        layers_per_block = kwargs.pop('layers_per_block')

        bottleneck_dim = kwargs.pop('bottleneck_dim')

        in_c = 3
        out_c = 1 if self.grayscale else 3

        # Input Shape: 3 x 96 x 96
        size = 96
        n_blocks = len(blocks)
        self.hidden_res = size // (2**n_blocks)
        self.hidden_dim = blocks[-1]
        self.hidden_features = self.hidden_dim * self.hidden_res * self.hidden_res


        # Encoder
        last_channels = in_c
        self.encoders = nn.ModuleList()
        for i, n_channels in enumerate(blocks):
            for j in range(layers_per_block):
                if j == 0:
                    layer = self.encoder_layer(last_channels, n_channels, stride=2)
                else:
                    layer = self.encoder_layer(n_channels, n_channels, stride=1)
                self.encoders.append(layer)
            
            last_channels = n_channels
            size = size // 2

        # Bottleneck
        self.bottleneck = nn.Sequential(
            self.linear_bn_act(self.hidden_features, bottleneck_dim),
            self.linear_bn_act(bottleneck_dim, self.hidden_features)
        )

        # Decoder
        self.decoders = nn.ModuleList()
        last_channels = blocks[-1]
        decoder_channels = list(reversed(blocks[:-1])) + [out_c] 
        for n_channels in decoder_channels:
            for j in range(layers_per_block):
                if j == layers_per_block - 1:
                    layer = self.upsample_conv(last_channels, n_channels)
                else:
                    layer = self.conv_bn_act(last_channels, last_channels)
                self.decoders.append(layer)
            last_channels = n_channels

        self.out_act = nn.Sigmoid()


    def forward(self, x):
        out = x

        # Encoder
        enc_outputs = []
        for enc_layer in self.encoders:
            out = enc_layer(out)
            enc_outputs.append(out)
        
        # Bottleneck
        out = out.reshape(-1, self.hidden_features)
        out = self.bottleneck(out)
        out = out.reshape(-1, self.hidden_dim, self.hidden_res, self.hidden_res)

        # Decoder
        for i, dec_layer in enumerate(self.decoders):
            if self.skip_connections:
                out = out + enc_outputs[-i-1]
            out = dec_layer(out)
        
        out = self.out_act(out)

        # Grayscale -> RGB
        if self.grayscale:
            out = out.repeat(1,3,1,1)

        return out


    def encoder_layer(self, in_c, out_c, stride):
        return self.conv_bn_act(in_c, out_c, stride=stride)


    def upsample_conv(self, in_c, out_c, kernel=None):
        kernel = kernel if kernel else self.kernel
        return UpsamplingConv(
            in_c, 
            out_c, 
            kernel, 
            activation = self.activation,
            residual=self.residual,
            bn=self.bn, 
            track_running_stats=False
        )


    def conv_bn_act(self, in_c, out_c, kernel=None, activation=None, stride=1):
        kernel = kernel if kernel else self.kernel
        activation = activation if activation else self.activation
        
        if self.residual:
            conv = ResidualBlock(
                in_c, 
                out_c, 
                kernel, 
                stride=stride,
                padding=(kernel-1) // 2,
                activation=self.activation,
                bn=self.bn,
                track_running_stats=False
            )
        else:
            conv =  ConvBnAct(
                in_c, 
                out_c, 
                kernel, 
                stride=stride,
                padding=(kernel-1) // 2,
                activation=self.activation,
                bn=self.bn,
                track_running_stats=False
            )

        return conv

    
    def linear_bn_act(self, in_c, out_c):
        return LinearBnAct(
            in_c, 
            out_c, 
            self.activation,
            bn=self.bn,
            track_running_stats=False
        )