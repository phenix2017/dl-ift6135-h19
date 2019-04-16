import torch
from torch import nn
import numpy as np
from utils import *



class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.nn = nn.Sequential(nn.Linear(args.in_dim, 5),
                                nn.ReLU(),
                                nn.Linear(5, 10),
                                nn.ReLU(),
                                nn.Linear(10, 1))

        self.sigm = nn.Sigmoid()

    def forward(self, x):
        out  = self.nn(x)

        if self.args.distance == 'js':
            out = self.sigm(out)

        return out

class VAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 0),
            nn.Tanh(),
            nn.Conv2d(16, 32, 3, 2, 0),
            nn.Tanh(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.Tanh(),
            nn.Conv2d(64, 64, 3, 2, 1),
            Flatten()
        )

        self.mean = nn.Linear(256, args.latent_dim)
        self.logvar = nn.Linear(256, args.latent_dim)

        self.generator = Generator(args)

    def forward(self, input = None, prior = False):

        if prior :
            latent = torch.randn(self.args.batch_size, self.args.latent_dim)
            meand, logvar =0, 0

        else :
            hidden = self.encoder(input)
            mean, logvar = self.mean(hidden), self.logvar(hidden)

            latent = sample_gauss(mean, logvar)


        output = self.generator(latent)

        return output, mean, logvar

    def generate(self, device):
        latent = torch.randn(self.args.batch_size, self.args.latent_dim, device = device)
        output = self.generator(latent)
        return output


class GAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.generator = Generator(args)
        self.discriminator = Discriminator_big(args)

    def generate(self,device, latent = None):
        if latent == None :
            latent = torch.randn(self.args.batch_size, self.args.latent_dim, device = device)

        output = self.generator(latent)
        return output

    def discriminate(self, input):

        out = self.discriminator(input)

        return out


class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args


        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 32 * 8, 4, 1, 0, ),
            nn.Tanh(),
            nn.ConvTranspose2d(32* 8, 32 * 4, 4, 2, 1, ),
            nn.Tanh(),
            nn.ConvTranspose2d(32 * 4, 32 * 2, 4, 2, 1, ),
            nn.Tanh(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1,),
            nn.Tanh()

        )

    def forward(self, z):
        img = self.model(z.unsqueeze(-1).unsqueeze(-1))

        return img


class Discriminator_big(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 0),
            nn.Tanh(),
            nn.Conv2d(16, 32, 3,2,  0),
            nn.Tanh(),
            nn.Conv2d(32, 64, 3,2,  1),
            nn.Tanh(),
            nn.Conv2d(64, 64, 3, 2, 1),
            Flatten()
        )

        self.linear = nn.Sequential(nn.Linear(256, 100),
                                    nn.Tanh(),
                                    nn.Linear(100, 1))

    def forward(self, img):

        hidden = self.model(img)
        final  = self.linear(hidden)
        return final

