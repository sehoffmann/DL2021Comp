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


def conv_bn_act(in_c, out_c, kernel, stride, activation, residual=True, bn=True):
    if residual:
        conv = ResidualBlock(
            in_c, 
            out_c, 
            kernel, 
            stride=stride,
            padding=(kernel-1) // 2,
            activation=activation,
            bn=bn,
            track_running_stats=False
        )
    else:
        conv =  ConvBnAct(
            in_c, 
            out_c, 
            kernel, 
            stride=stride,
            padding=(kernel-1) // 2,
            activation=activation,
            bn=bn,
            track_running_stats=False
        )

    return conv


class EncoderBlock(nn.Module):

    def __init__(self, in_c, features, n_layers, kernel, activation, residual=True, bn=True):
        super(EncoderBlock, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                in_channels = in_c
                stride = 2
            else:
                in_channels = features
                stride = 1
            
            layer = conv_bn_act(
                in_channels, 
                features, 
                kernel, 
                stride=stride,
                activation=activation,
                residual=residual,
                bn=bn
            )
            self.layers.append(layer)


    def forward(self, x):
        out = x
        activations = []
        for layer in self.layers:
            out = layer(out)
            activations.append(out)

        return out, activations


class DecoderBlock(nn.Module):

    def __init__(self, features, out_c, n_layers, kernel, activation, residual=True, bn=True, use_skip_convs=True):
        super(DecoderBlock, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i < n_layers-1:
                layer = conv_bn_act(
                    features,
                    features,
                    kernel,
                    stride=1,
                    activation=activation,
                    residual=residual,
                    bn=bn
                )
            else:
                layer = UpsamplingConv(
                    features, 
                    out_c,
                    kernel, 
                    activation=activation,
                    residual=residual,
                    bn=bn, 
                    track_running_stats=False
                )
            
            self.layers.append(layer)

        if use_skip_convs:
            self.skip_convs = nn.ModuleList()
            for i in range(n_layers):
                layer = conv_bn_act(
                    features,
                    features,
                    kernel,
                    stride=1,
                    activation=activation,
                    residual=residual,
                    bn=bn
                )
                self.skip_convs.append(layer)
        else:
            self.skip_convs = None


    def forward(self, x, skip_cons):
        out = x
        for i, layer in enumerate(self.layers):
            if skip_cons and skip_cons[i] is not None:
                s = skip_cons[i]
                if self.skip_convs:
                    s = self.skip_convs[i](s)
                out = out + s
            out = layer(out)

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
        self.less_skips = kwargs.pop('less_skips', False)

        blocks = kwargs.pop('blocks')
        layers_per_block = kwargs.pop('layers_per_block')

        use_skip_convs = kwargs.pop('use_skip_convs')
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

        # Bottleneck
        self.bottleneck = nn.Sequential(
            self.linear_bn_act(self.hidden_features, bottleneck_dim),
            self.linear_bn_act(bottleneck_dim, self.hidden_features)
        )

        # Decoder
        self.decoders = nn.ModuleList()
        out_channels = reversed([out_c] + blocks[:-1]) 
        for n_features, out_features in zip(reversed(blocks), out_channels):
            block = DecoderBlock(
                n_features, 
                out_features,
                layers_per_block,
                self.kernel,
                self.activation,
                use_skip_convs=use_skip_convs,
                residual=self.residual,
                bn=self.bn
            )
            self.decoders.append(block)

        self.out_act = nn.Sigmoid()


    def forward(self, x):
        out = x

        # Encoder
        skip_connections = []
        for enc_layer in self.encoders:
            out, skips = enc_layer(out)
            if self.less_skips:
                skips[:-1] = [None] * (len(skips)-1)
            skip_connections.append(skips)
        
        # Bottleneck
        out = out.reshape(-1, self.hidden_features)
        out = self.bottleneck(out)
        out = out.reshape(-1, self.hidden_dim, self.hidden_res, self.hidden_res)

        # Decoder
        for dec_layer, skips in zip(self.decoders, reversed(skip_connections)):
            if self.skip_connections:
                skips = list(reversed(skips))
                out = dec_layer(out, skips)
            else:
                out = dec_layer(out, None)
        
        out = self.out_act(out)

        # Grayscale -> RGB
        if self.grayscale:
            out = out.repeat(1,3,1,1)

        return out


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