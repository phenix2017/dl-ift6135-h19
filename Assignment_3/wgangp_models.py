import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))


def snlinear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))


def sn_embedding(num_embeddings, embedding_dim):
    return spectral_norm(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim))


class ConditionalBatchNorm2d(nn.Module):
    # https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        # self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.bn = nn.GroupNorm(4, num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        # self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, :num_features].fill_(1.)  # Initialize scale to 1
        self.embed.weight.data[:, num_features:].zero_()  # Initialize bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class GenResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, sn):
        super(GenResidualBlock, self).__init__()

        if sn:
            conv2d = snconv2d
        else:
            conv2d = nn.Conv2d

        self.cond_bn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2d1 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.cond_bn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.conv2d2 = conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d0 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, labels, upsample=True):
        x0 = x
        if upsample:
            x0 = F.interpolate(x0, scale_factor=2, mode='nearest') # upsample

        x0 = self.conv2d0(x0)

        x = self.cond_bn1(x, labels)
        x = self.relu(x)
        if upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest') # upsample

        x = self.conv2d1(x)

        x = self.cond_bn2(x, labels)
        x = self.relu(x)
        x = self.conv2d2(x)

        out = x + x0
        return out


class Generator(nn.Module):
    """Generator."""

    def __init__(self, g_conv_dim, noise_dim, num_classes,
                 use_qa=False, q_skipthought_dim=4800, a_skipthought_dim=4800, q_emb_dim=128, a_emb_dim=64,
                 sn=False):
        super(Generator, self).__init__()

        print("Building G")

        self.noise_dim = noise_dim
        self.g_conv_dim = g_conv_dim
        self.num_classes = num_classes
        self.use_qa = use_qa
        self.q_skipthought_dim = q_skipthought_dim
        self.a_skipthought_dim = a_skipthought_dim
        self.q_emb_dim = q_emb_dim
        self.a_emb_dim = a_emb_dim
        self.sn = sn

        if self.sn:
            linear = snlinear
            conv2d = snconv2d
        else:
            linear = nn.Linear
            conv2d = nn.Conv2d

        # QA embeddings
        if self.use_qa:
            self.linear_q = linear(in_features=self.q_skipthought_dim, out_features=self.q_emb_dim)
            self.linear_a = linear(in_features=self.a_skipthought_dim, out_features=self.a_emb_dim)
        # Concatenate G_inputs: noise, q_emb, a_emb
        if self.use_qa:
            self.linear_G_input = linear(in_features=(self.noise_dim + self.q_emb_dim + self.a_emb_dim),
                                             out_features=self.g_conv_dim*4*4)
        else:
            self.linear_G_input = linear(in_features=self.noise_dim, out_features=self.g_conv_dim*4*4)
        # Gen blocks
        self.block1 = GenResidualBlock(self.g_conv_dim, self.g_conv_dim, self.num_classes, self.sn)
        self.block2 = GenResidualBlock(self.g_conv_dim, self.g_conv_dim, self.num_classes, self.sn)
        self.block3 = GenResidualBlock(self.g_conv_dim, self.g_conv_dim, self.num_classes, self.sn)
        # self.block4 = GenResidualBlock(self.g_conv_dim, self.g_conv_dim, self.num_classes, self.sn)
        self.bn = nn.BatchNorm2d(self.g_conv_dim)
        # self.bn = nn.GroupNorm(self.g_conv_dim, self.g_conv_dim)
        self.relu = nn.ReLU(inplace=False)
        self.conv2d1 = conv2d(in_channels=self.g_conv_dim, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        # Weight init
        # print("Applying weight init to G")
        # self.apply(init_weights)

        print("yoyo")

    def forward(self, noise, labels, q_skipthought=None, a_skipthought=None):
        # n x noise_dim
        if self.use_qa:
            # n x q_skipthought_dim
            act_q = self.linear_q(q_skipthought)            # n x q_emb_dim
            # n x a_skipthought_dim
            act_a = self.linear_a(a_skipthought)            # n x a_emb_dim
        # Concatenate G inputs
        if self.use_qa:
            G_input = torch.cat((noise, act_q, act_a), 1)
        else:
            G_input = noise
        # Linear Tx G_input
        act0 = self.linear_G_input(G_input)   # n x g_conv_dim*4*4
        act0 = act0.view(-1, self.g_conv_dim, 4, 4)     # n x g_conv_dim x 4 x 4
        act1 = self.block1(act0, labels)    # n x g_conv_dim x  8 x  8
        act2 = self.block2(act1, labels)    # n x g_conv_dim x 16 x 16
        act3 = self.block3(act2, labels)    # n x g_conv_dim x 32 x 32
        # act4 = self.block4(act3, labels)    # n x g_conv_dim x 64 x 64
        act4 = self.bn(act3)                # n x g_conv_dim x 32 x 32
        act4 = self.relu(act4)              # n x g_conv_dim x 32 x 32
        act5 = self.conv2d1(act4)           # n x 3 x 32 x 32
        act5 = self.tanh(act5)              # n x 3 x 32 x 32
        return act5


class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sn):
        super(DiscOptBlock, self).__init__()

        if sn:
            conv2d = nn.Conv2d
        else:
            conv2d = snconv2d

        self.conv2d1 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.conv2d2 = conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.conv2d0 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, downsample=True):
        x0 = x
        if downsample:
            x0 = self.downsample(x0)

        x0 = self.conv2d0(x0)

        x = self.conv2d1(x)
        x = self.relu(x)
        x = self.conv2d2(x)
        if downsample:
            x = self.downsample(x)

        out = x + x0
        return out


class DiscResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sn):
        super(DiscResidualBlock, self).__init__()

        self.relu = nn.ReLU(inplace=False)
        self.downsample = nn.AvgPool2d(2)

        if sn:
            conv2d = nn.Conv2d
        else:
            conv2d = snconv2d

        self.conv2d1 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d0 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, downsample=True):
        x0 = x
        x0 = self.conv2d0(x0)
        if downsample:
            x0 = self.downsample(x0)

        x = self.relu(x)
        x = self.conv2d1(x)

        x = self.relu(x)
        x = self.conv2d2(x)
        if downsample:
            x = self.downsample(x)

        out = x + x0
        return out


class Discriminator(nn.Module):
    """Discriminator."""

    def __init__(self, d_conv_dim, num_classes, sn=False):
        super(Discriminator, self).__init__()
        print("Building D")
        self.d_conv_dim = d_conv_dim
        self.num_classes = num_classes
        self.sn = sn
        self.block1 = DiscOptBlock(3, d_conv_dim, sn)
        self.block2 = DiscResidualBlock(d_conv_dim, d_conv_dim, sn)
        # self.block3 = DiscResidualBlock(d_conv_dim, d_conv_dim, sn)
        self.block4 = DiscResidualBlock(d_conv_dim, d_conv_dim, sn)
        self.block5 = DiscResidualBlock(d_conv_dim, d_conv_dim, sn)
        self.relu = nn.ReLU(inplace=False)
        if sn:
            self.linear1 = snlinear(in_features=d_conv_dim, out_features=1)
            self.linear2 = snlinear(in_features=d_conv_dim, out_features=num_classes)
        else:
            self.linear1 = nn.Linear(in_features=d_conv_dim, out_features=1)
            self.linear2 = nn.Linear(in_features=d_conv_dim, out_features=num_classes)

        # Weight init
        # print("Applying weight init to D")
        # self.apply(init_weights)

        print("yoyo")

    def forward(self, x, labels):
        # n x 3 x 32 x 32
        h1 = self.block1(x)     # n x d_conv_dim x 16 x 16
        h2 = self.block2(h1)    # n x d_conv_dim x  8 x  8
        # h3 = self.block3(h2)    # n x d_conv_dim x  8 x  8
        h4 = self.block4(h2, downsample=False)  # n x d_conv_dim x 8 x 8
        h5 = self.block5(h4, downsample=False)  # n x d_conv_dim x 8 x 8
        h5 = self.relu(h5)              # n x d_conv_dim x 8 x 8
        h6 = torch.mean(h5, dim=[2,3])  # n x d_conv_dim
        output_wgan = torch.squeeze(self.linear1(h6)) # n

        # ACGAN
        output_acgan = self.linear2(h6) # n x 10

        return output_wgan, output_acgan