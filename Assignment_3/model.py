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



class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            # nn.ReLU(True),
            nn.ELU(),

            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),       
            nn.BatchNorm2d(num_features=1024),
            # nn.ReLU(True),
            nn.ELU(),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            # nn.ReLU(True),
            nn.ELU(),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            # nn.ReLU(True),
            nn.ELU(),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=4, stride=2, padding=1))
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x.unsqueeze(-1).unsqueeze(-1))
        return self.output(x)



class Discriminator_big(nn.Module):
    def __init__(self, args):
        super().__init__()

        ndf = 64
        nc=3
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ELU(),
            # nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.ELU(),
            # nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.ELU(),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.ELU(),
            # nn.LeakyReLU(0.2, inplace=True),

            Flatten(),
            nn.Linear(512,100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100,1),
            nn.Sigmoid()
            # # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, input):
        output = self.model(input)

        return output.view(-1, 1).squeeze(1)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 1, 1)