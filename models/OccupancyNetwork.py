import torch
import torch.nn as nn
from torch import distributions as dist
import torch.nn.functional as F
from torchvision import models
from utils.common import normalize_imagenet
from copy import deepcopy
import math
import sys
from typing import Tuple
sys.path.insert(0, './vit/models')
from vit_model import ViT
from vit_large_model import ViTModel

class ResNetFeatures(nn.Module):
    ''' ResNetFeatures class for extracting features using ResNet18
    '''
    def __init__(self, model=models.resnet18(pretrained=True)):
        super(ResNetFeatures, self).__init__()

        self.sequential = nn.Sequential(model.conv1,
                                  model.bn1,
                                  model.relu,
                                  model.maxpool,
                                  model.layer1,
                                  model.layer2,
                                  model.layer3,
                                  model.layer4)  

    def forward(self, x):
        return self.sequential(x)
    

def frozen_batch_norm(model):
    # Freeze the batch normalization layers of the model
    for name, module in model.named_modules():
        if name == 'bn1' :
            module.requires_grad_ = False
        if name == 'layer1' or name == 'layer2' or name == 'layer3' or name == 'layer4':
            for child_name, child_module in module.named_modules():
                if (len(child_name) > 1) and child_name[2:4] == 'bn':
                    child_module.requires_grad_ = False
    return model


class OccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        p0_z (dist): prior distribution for latent code z
        device (device): torch device
    '''

    def __init__(self, config , device="cuda"):
        super().__init__()

        self.decoder = Decoder().to(device)
        self.encoder = ViT((config['input_image_channels'],config['img_size'],config['img_size']),
                                                          n_patches=config['n_patches'],
                                                          n_blocks=config['n_blocks'],
                                                          hidden_d=config['hidden_d'],
                                                          n_heads=config['n_heads'],
                                                          out_d=config['out_d'],
                                                          mlp_ratio=config['mlp_ratio']
                                                          ).to(device)
        self._device = device
        p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))
        self.p0_z = p0_z


    def forward(self, p, inputs, sample=False, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        batch_size = p.size(0)
        c = self.encoder(inputs)
        p_r = self.decoder(p, c, **kwargs)
        return p_r

    def compute_elbo(self, p, occ, inputs, **kwargs):
        ''' Computes the expectation lower bound.

        Args:
            p (tensor): sampled points
            occ (tensor): occupancy values for p
            inputs (tensor): conditioning input
        '''
        c = self.encoder(inputs)
        p_r = self.decoder(p, c, **kwargs)
        rec_error = -p_r.log_prob(occ).sum(dim=-1)
        elbo = -rec_error
        return elbo, rec_error

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
    def infer_z(self, p, occ, c, **kwargs):
        ''' Infers z.

        Args:
            p (tensor): points tensor
            occ (tensor): occupancy values for occ
            c (tensor): latent conditioned code c
        '''

        batch_size = p.size(0)  # =64 for traiing ,10 for validation do
        mean_z = torch.empty(batch_size, 0).to(self._device)
        logstd_z = torch.empty(batch_size, 0).to(self._device)

        q_z = dist.Normal(mean_z, torch.exp(logstd_z))
        return q_z


    def get_z_from_prior(self, size=torch.Size([]), sample=True):
        ''' Returns z from prior distribution.

        Args:
            size (Size): size of z
            sample (bool): whether to sample
        '''
        if sample:
            z = self.p0_z.sample(size).to(self._device)
        else:
            z = self.p0_z.mean.to(self._device)
            z = z.expand(*size, *z.size())

        return z


class DecoderCBatchNorm(nn.Module):
    ''' Decoder with conditional batch normalization (CBN) class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''

    def __init__(self, dim=3, z_dim=0, c_dim=256,
                 hidden_size=256, leaky=False, legacy=False):
        super().__init__()
        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

        self.bn = CBatchNorm1d(c_dim, hidden_size)
        self.fc_out = nn.Conv1d(hidden_size, 1, 1)
        self.actvn = F.relu

    def forward(self, p, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)
        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)
        p_r = dist.Bernoulli(logits=out)

        return p_r


class Resnet18(nn.Module):
    r''' ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim=256, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = frozen_batch_norm(ResNetFeatures()).to("cuda")
        self.fc = nn.Conv1d(512, 1,1)
        self.num_predict = 256
        self.transformer = Transformer()
        self.positional_embeddings = PositionEmbeddingSine()
        self.query_embed = nn.Embedding(256, 512) # (num_predict, hidden_dim)
        self.bn = nn.BatchNorm1d(512)
    def forward(self, x):
        x = normalize_imagenet(x) # (N, 3, 224, 224)
        batch_size = x.shape[0]
        feature = self.features(x) # (N, 512, 7, 7)
        pos_embed = self.positional_embeddings(feature)
        net = self.transformer(feature, self.query_embed.weight, pos_embed)[0]  # (256, 512, bs)
        net = self.bn(net)
        out = self.fc(net) # (N, 256)

        out = out.squeeze(1).transpose(0,1) # (bs, 256)
        return out




class CResnetBlockConv1d(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.

    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks 
    '''

    def __init__(self, c_dim=256, size_in=256, size_h=None, size_out=None,
                 norm_method='batch_norm', legacy=False):
        super().__init__()
        size_h = size_in
        size_out = size_in
        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        self.bn_0 = CBatchNorm1d(c_dim, size_in, norm_method=norm_method)
        self.bn_1 = CBatchNorm1d(c_dim, size_h, norm_method=norm_method)
        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        self.shortcut = None
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))
        x_s = x
        return x_s + dx


class CBatchNorm1d(nn.Module):
    ''' Conditional batch normalization layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim=256, f_dim=256, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        # self.bn = nn.BatchNorm1d(f_dim, affine=False)
        self.gn = nn.GroupNorm(8, f_dim, affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        assert(x.size(0) == c.size(0))
        assert(c.size(1) == self.c_dim)

        # c is assumed to be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Batchnorm
        # net = self.bn(x)
        net=self.gn(x)
        out = gamma * net + beta
        return out


class PositionEmbeddingSine(nn.Module): 
    ''' Position embedding class, very similar to the one used by the Attention 
    is all you need paper, generalized to work on images.
    '''

    def __init__(self, n_position_features=256, temperature=10000, normalize=True, scale=None): 
        super(PositionEmbeddingSine, self).__init__()
        self.n_position_features = n_position_features
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        N, _, H, W = tensor.shape

        mask = torch.zeros(N, H, W, dtype=torch.bool, device=tensor.device)
        not_mask = ~mask

        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)

        if self.normalize:
            epsilon = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + epsilon) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + epsilon) * self.scale

        dim_t = torch.arange(self.n_position_features, dtype=torch.float32, device=tensor.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.n_position_features)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos # (bs, n_position_features*2, )


    

class Transformer(nn.Module):
    ''' DETR Transformer class.

        Copy-paste from torch.nn.Transformer with modifications:
            * positional encodings are passed in MHattention
            * extra LN at the end of encoder and decoder is removed
            * decoder returns a stack of activations from all decoding layers
    ''' 

    def __init__(self, n_dims=512, n_head=8, n_encoder_layers=6, n_decoder_layers=6, dim_feed_forward=512, dropout=0.1):
        super().__init__()
        self.n_dims = n_dims

        encoder_layer = TransformerEncoderLayer(n_dims, n_head, dim_feed_forward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, n_encoder_layers)

        decoder_layer = TransformerDecoderLayer(n_dims, n_head, dim_feed_forward, dropout)
        decoder_norm = nn.LayerNorm(n_dims)
        self.decoder = TransformerDecoder(decoder_layer, n_decoder_layers, decoder_norm)

    def forward(self, src, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape

        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, pos=pos_embed)
        hs = self.decoder(tgt, memory, pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)

    
class TransformerEncoder(nn.Module):
    ''' Encoder part of the Transformer architecture
    '''
    def __init__(self, encoder_layer, n_layer, norm=None):
        super().__init__()
        self.layers = clone_layer(encoder_layer, n_layer)
        self.norm = norm

    def forward(self, src, pos=None):
        out = src

        for layer in self.layers:
            out = layer(out, pos)

        if self.norm:
            out = self.norm(out)

        return out


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, n_layer, norm=None):
        super().__init__()
        self.layers = clone_layer(decoder_layer, n_layer)
        self.norm = norm

    def forward(self, tgt, memory, pos=None, query_pos=None):
        out = tgt

        for layer in self.layers:
            out = layer(out, memory, pos, query_pos)

        if self.norm:
            out = self.norm(out)

        return out


class TransformerEncoderLayer(nn.Module):
    ''' A single layer in the encoder part of the Transformer architecture
    '''
    def __init__(self, n_dims=512, n_head=8, dim_feed_forward=512, dropout=0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(n_dims, n_head, dropout=dropout)
        self.linear1 = nn.Linear(n_dims, dim_feed_forward)
        self.linear2 = nn.Linear(dim_feed_forward, n_dims)

        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(n_dims)
        self.norm2 = nn.LayerNorm(n_dims)
        
    def forward(self, src, pos=None):
        q = k = pos_add(src, pos)
        self.self_attention = self.self_attention.to("cuda")

        src2 = self.self_attention(q, k, value=src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):
    ''' A single layer in the decoder part of the Transformer architecture
    '''
    def __init__(self, n_dims=512, n_head=8, dim_feed_forward=512, dropout=0.1):
        super().__init__()
        self.self_attention1 = nn.MultiheadAttention(n_dims, n_head, dropout=dropout)
        self.self_attention2 = nn.MultiheadAttention(n_dims, n_head, dropout=dropout)

        self.linear1 = nn.Linear(n_dims, dim_feed_forward)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feed_forward, n_dims)

        self.norm1 = nn.LayerNorm(n_dims)
        self.norm2 = nn.LayerNorm(n_dims)
        self.norm3 = nn.LayerNorm(n_dims)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, pos=None, query_pos=None):
        q = k = pos_add(tgt, query_pos)
        tgt2 = self.self_attention1(q, k, value=tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.self_attention2(query=pos_add(tgt, query_pos), key=pos_add(memory, pos), value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(3, 256, 1)
        self.cbn = CBatchNorm1d()

        # self.conv0 = nn.Conv1d(512, 256, kernel_size=1)
        # self.bn0 = nn.BatchNorm1d(256)

        self.conv1 = nn.Conv1d(256, 128, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(128)
        #self.dropout1 = nn.Dropout(p=0.5)
        self.relu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv1d(128, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(64)
        #self.dropout2 = nn.Dropout(p=0.5)
        self.relu2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv1d(64, 32, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(32)
        #self.dropout3 = nn.Dropout(p=0.5)
        self.relu3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv1d(32, 16, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(16)
        #self.dropout4 = nn.Dropout(p=0.5)
        self.relu4 = nn.LeakyReLU(0.2)

        self.conv5 = nn.Conv1d(16, 4, kernel_size=1)
        self.bn5 = nn.BatchNorm1d(4)
        #self.dropout5 = nn.Dropout(p=0.5)
        self.relu5 = nn.ReLU()   ## only ReLu here

        self.conv6 = nn.Conv1d(4, 1, kernel_size=1)

    def forward(self,p,c):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        p = self.conv(p)
        p= self.cbn(p, c)  # N,256,num_points

        # net = self.conv0(p)
        # net = self.bn0(net)
        ## to convert N,256,num_points to N,1,256
        net = self.conv1(p)
        net = self.bn1(net)
        #net = self.dropout1(net)
        net = self.relu1(net)

        net = self.conv2(net)
        net = self.bn2(net)
        #net = self.dropout2(net)
        net = self.relu2(net)

        net = self.conv3(net)
        net = self.bn3(net)
        #net = self.dropout3(net)
        net = self.relu3(net)

        net = self.conv4(net)
        net = self.bn4(net)
        #net = self.dropout4(net)
        net = self.relu4(net)

        net = self.conv5(net)
        net = self.bn5(net)
        #net = self.dropout5(net)
        net = self.relu5(net)

        net = self.conv6(net) # N,1,num_points

        net = net.squeeze(1)
        p_r = dist.Bernoulli(logits=net)
        return p_r
    
def pos_add(tensor, pos):
    return tensor + pos if pos is not None else tensor

def clone_layer(layer, n_layer):
    return nn.ModuleList([deepcopy(layer) for _ in range(n_layer)])
    