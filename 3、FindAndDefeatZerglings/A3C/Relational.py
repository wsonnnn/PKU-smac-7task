import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

'''Implememted from the paper '''

# debug flags
debug = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

#TODO : adding LayerNorm/BatchNorm

#TODO : adding Conv2DLSTM for The agent
class PositionalEncoding(nn.Module):

    def __init__(self):

        super(PositionalEncoding, self).__init__()
    
    def forward(self, x):
        return self._add_embed2d(x)
    
    def _add_embed2d(self, x):
        shape = x.shape
        H, W = shape[-2], shape[-1]

        x_space = torch.linspace(-1, 1, H)
        y_space = torch.linspace(-1, 1, W).view(-1, W)

        x_repeat = x_space.repeat(shape[0], W, 1).view(-1, 1, W, H).transpose(3, 2).to(device)
        y_repeat = y_space.repeat(shape[0], 1, H).view(-1, 1, W, H).transpose(2, 3).to(device)

        if debug:
            print("shape of input ", shape)
            print("shape of x ", x_repeat.shape)
            print("shape of y ", y_repeat.shape)

        x_out = torch.cat([x, x_repeat, y_repeat], dim=1)
        return x_out

class ResBlock(nn.Module):

    def __init__(self, 
                 input_channel,
                 hidden_channel=32):

        super(ResBlock, self).__init__()
        self.embedding = nn.Sequential(nn.Conv2d(input_channel, hidden_channel, 4, 2, 1))

        self.conv = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(hidden_channel, hidden_channel, 3, 1, 1),
                                  nn.ReLU(),
                                  nn.Conv2d(hidden_channel, hidden_channel, 3, 1, 1))
    
    def forward(self, x):
        out = self.embedding(x)
        out = F.relu(self.conv(out) + out)
        return out

# (batch_size, channel, 64, 64) -> (batch_size, 32, 32, 32) -> (batch_size, 32, 16, 16)
# then downsampling(using average pooling) : (batch_size, 32, 8, 8)

# transpose to (batch_size, 64, 32)
# query, key, value in shape : (batch_size, 64, xxx)

# layer norm over(64, xxx)

# matmul for the batch
# (batch_size, 64, xxx) matmul (batch_size, xxx, 64) -> (batch_size, 64, 64)

# add into attention head
# get (batch_size, 8x8), to softmax and view as (batch_size, 8, 8)

# spatial arguments
# deconvolution : (batch_size, 64, 64)

# non-spatial arguments
# (batch_size, arg_sizes)
'''
class AttentionHead(nn.Module):

    def __init__(self, input_dims=(64, 32), hidden_dim=32):


        super(AttentionHead, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim

        self.qurey = nn.Sequential(nn.Linear(input_dims[1], hidden_dim),
                                   nn.LayerNorm(hidden_dim))

        self.key = nn.Sequential(nn.Linear(input_dims[1], hidden_dim),
                                 nn.LayerNorm(hidden_dim))

        self.value = nn.Sequential(nn.Linear(input_dims[1], hidden_dim),
                                   nn.LayerNorm(hidden_dim))

    def forward(self, x):

        y = x.transpose(2, 1)

        query  = self.query(y)
        key = self.key(y).transpose(2, 1)
        value = self.value(y)

        atten_weights = F.softmax(torch.matmul(query, key)/np.sqrt(self.hidden_dim), dim = -1)

        attention = torch.matmul(atten_weights, value)
        return attention
'''

'''
input (batch_size, H, W, channels)
after MHA (batch_size, H*W, channels)
MLP (channels, hidden) -> (hidden, channels)
'''
# input as shape (batch_size, 64, channels)
# a bit weired if following MHA written by torch
class AttentionBlock(nn.Module):

    def __init__(self, n_features, n_heads, n_hidden, dropout = 0.0):

        super(AttentionBlock, self).__init__()
        self.norm1 = nn.LayerNorm(n_features * n_heads)
        self.norm2 = nn.LayerNorm(n_features)
        # shared attention and MLP between blocks
        self.n_heads = n_heads
        self.attention = nn.MultiheadAttention(n_features * n_heads, n_heads, dropout=dropout)
        self.MLP = nn.Sequential(nn.Linear(n_features * n_heads, n_hidden),
                                 nn.LayerNorm(n_hidden),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(n_hidden, n_features))
    
    def forward(self, x, mask=None):
        
        x_input = torch.cat([x for _ in range(self.n_heads)], dim=-1)
        atten_out, atten_weights = self.attention(x_input, x_input, x_input, key_padding_mask=mask)
        # attention : add and norm
        x_norm = self.norm1(atten_out + x_input)
        y_norm = self.norm2(x + self.MLP(x_norm))

        return y_norm

'''
RelationalModuke:
    shared attention block, shared MLP between blocks
'''

#TODO : adding Conv2DLSTM
class MemoryModule(nn.Module):...

class RelationalModule(nn.Module):

    def __init__(self, n_features, n_heads=3, n_hidden=384, n_blocks=3, dropout = 0.0):

        super(RelationalModule, self).__init__()

        embedding = AttentionBlock(n_features, n_heads, n_hidden, dropout)

        self.attentionBlocks = nn.Sequential(*[embedding for _ in range(n_blocks)])

    def forward(self, x):
        out = self.attentionBlocks(x)
        return out

# relational spatial : (Batch, H, W, C)

# relational nonspatial : flatten tensor as (Batch, H*W*C), process using relu

class SC2Net(nn.Module):

    def __init__(self, minimap_channel, screen_channel, nonspatial_dim, n_actions):

        super(SC2Net, self).__init__()

        # state encoding
        # can switch to average pooling anyhow
        nonspatial_dim = nonspatial_dim + 16

        self.action_embedding = nn.Embedding(n_actions, 16)
        self.mini_conv = nn.Sequential(PositionalEncoding(),
                                       ResBlock(minimap_channel+2, 32),
                                       ResBlock(32, 32),
                                       nn.MaxPool2d(kernel_size=2))

        self.screen_conv = nn.Sequential(PositionalEncoding(),
                                         ResBlock(screen_channel+2, 32),
                                         ResBlock(32, 32),
                                         nn.MaxPool2d(kernel_size=2))

        self.nonspatial = nn.Sequential(nn.Linear(nonspatial_dim, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 64))
        
        # features for relational module : (B, 2xC, H, W)
        self.relational = RelationalModule(64, 3, 384, 2)
        
        self.nonspatial_MLP = nn.Sequential(nn.Linear(64, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512))

        # concatenate the output of input2d and nonspatial
    
    def forward(self, minimap, screen, player, last_action):

        last_action = self.action_embedding(last_action)

        if debug:
            print("player shape", player.shape)
            print("last_action shape", last_action.shape)

        nonspatial = torch.cat([player, last_action], dim=-1)

        mini_conv = self.mini_conv(minimap)
        screen_conv = self.screen_conv(screen)
        input3d = torch.cat([mini_conv, screen_conv], dim=1)
        # (B, 64, 8, 8)

        output3d = input3d
        shape = output3d.shape

        # (B, 8x8, 64)
        output3d = output3d.view(-1, shape[1], shape[2]*shape[3]).transpose(2, 1)

        # (B, 64)
        input2d = self.nonspatial(nonspatial)

        # (B, 8x8, 64)
        relational_out = self.relational(output3d)

        # relational spatial
        relational_spatial = relational_out.transpose(1, 2).view(shape[0], shape[1], shape[2], -1)

        # relational nonspatial
        # TODO : adding Feature-wise max pooling
        # relational_nonspatial = torch.max(relational_out, dim=2)[0]
        relational_nonspatial = torch.max(relational_out, dim=2)[0]

        relational_nonspatial = self.nonspatial_MLP(relational_nonspatial)
        
        #(batch_size, 64) + (batch_size, 512)
        nonspatial_latent = torch.cat([input2d, relational_nonspatial], dim=-1)
        
        return relational_spatial, nonspatial_latent
