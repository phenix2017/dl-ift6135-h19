import torchvision.datasets
from torch.utils.data import dataset
import torch.autograd as autograd
import torch
from torch import nn
import numpy as np
import torchvision.transforms as transforms
image_transform = transforms.Compose([
    transforms.ToTensor()

])

def get_data_loader(dataset_location, batch_size):
    trainvalid = torchvision.datasets.SVHN(
        dataset_location, split='train',
        download=False,
        transform=image_transform
    )

    trainset_size = int(len(trainvalid) * 0.9)
    trainset, validset = dataset.random_split(
        trainvalid,
        [trainset_size, len(trainvalid) - trainset_size]
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
    )

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(
            dataset_location, split='test',
            download=False,
            transform=image_transform
        ),
        batch_size=batch_size,
    )

    return trainloader, validloader, testloader

def sample_gauss(mu, logvar):
    eps = torch.randn(mu.size(), device = mu.device)

    return mu + torch.exp(0.5*logvar)*eps

def get_kl( q_loc, q_logvar) :
    p_loc, p_logvar = torch.zeros_like(q_loc), torch.zeros_like(q_loc)

    kl = 0.5*(q_logvar - p_logvar) + 0.5*(torch.exp(p_logvar) + (p_loc - q_loc)**2)/torch.exp(q_logvar) - 0.5

    return kl

def calc_gradient_penalty(net, real_data, fake_data, lam = 10):
    alpha = torch.rand(real_data.size()[0], 1, device=real_data.device).unsqueeze(-1).unsqueeze(-1)

    alpha = alpha.expand(real_data.size()).to(real_data.device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data[:real_data.size()[0]])

    interpolates = autograd.Variable(interpolates, requires_grad=True).to(real_data.device)

    disc_interpolates = net(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(
                                  disc_interpolates.size(), device=real_data.device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lam

    return gradient_penalty

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)