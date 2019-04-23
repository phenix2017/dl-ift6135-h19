import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import scipy.io as spio
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.distributions as tdist

import numpy as np
import pickle
import struct
import os
import scipy.io

bs = 100
img_size = 28

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 1, 1)


class VAE(nn.Module):
    def __init__(self, h_dim=256, z_dim=100):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ELU(),
            nn.AvgPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ELU(),
            nn.AvgPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(64, 256, kernel_size=5),
            nn.ELU(),
            Flatten(),
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)


        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ELU(),
            nn.Conv2d(256, 64, kernel_size=(5, 5), padding=(4, 4)),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(2, 2)) ,
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=(3, 3), padding=(2, 2)),
            nn.ELU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=(2, 2)),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z1 = self.fc3(z)
        return self.decoder(z1), z, mu, logvar

    def log_normal(x, mu, var, eps=0.0):
        if eps > 0.0:
            var = var + eps
        result = np.log(2 * np.pi) + torch.log(var) + (x - mu).pow(2) / var
        res_sum_axis = result.shape[-1]
        return -0.5 * result.sum(res_sum_axis) 

    def iw_sample(self, x, k=200):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        ## looping over batch members
        logp_batch = []
        for idx, zvec in enumerate(z):       
            mv_normal = MultivariateNormal(mu[idx], torch.diag(torch.exp(logvar[idx])))
            sample_list = []
            for i in range(k):
                sample_list.append(mv_normal.sample().cuda())
            samples = torch.stack(sample_list)
            x_samples = self.decoder(self.fc3(samples))
            x_tiled = x[idx].unsqueeze(0).repeat((k,1,1,1))
            batch_iw_result = self.eval_likelihood(x_samples.cpu(), x_tiled.cpu(), samples.cpu(), mu[idx].cpu(), logvar[idx].cpu())
            logp_batch.append(batch_iw_result)
        logp = torch.stack(logp_batch)  #log p for each el in batch
        return logp

    def eval_likelihood(self, recon_x, x, z, mu, log_var):
        samples = []
        ## enumerating over k
        m_p = MultivariateNormal(torch.zeros(z.shape[-1]), torch.eye(z.shape[-1]))
        for idx, zvec in enumerate(z):
            BCE = F.binary_cross_entropy(recon_x[idx:idx+1], x[idx:idx+1].view(-1, 784), reduction='sum')
            m_q = MultivariateNormal(mu, torch.diag(torch.exp(log_var)))
            qz = m_q.log_prob(zvec)
            pz = m_p.log_prob(zvec)
            #BCE = torch.sum(inputs*torch.log(p) + (1-inputs)*torch.log(1-p), -1)
            result = pz - qz - BCE
            samples.append(result)
        result_vals = torch.stack(samples).unsqueeze(0)
        logp = self.logsumexp(result_vals) -torch.log(torch.tensor(float(200)))

        return logp

    def logsumexp(self, inputs, dim=1):
        max_score, _ = inputs.max(dim)
        stable_vec = inputs - max_score.unsqueeze(dim)
        return max_score + (stable_vec.exp().sum(dim)).log() 


# build model
vae = VAE(h_dim=256, z_dim=100)
if torch.cuda.is_available():
    vae.cuda()


optimizer = optim.Adam(vae.parameters())
# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train(epoch, loader):
    vae.train()
    train_loss = 0
    loader.start_epoch()
    batches = loader.batcher()
    for batch_idx, data in enumerate(batches):
        data = data.cuda()
        optimizer.zero_grad()
        
        recon_batch, z, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item() 
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), loader.dataset_size,
                100. * batch_idx / loader.num_batches, loss.item() / (data.shape[0]*1.0)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / loader.dataset_size))

def val(loader, epoch):
    vae.eval()
    loader.val_mode()
    val_loss= 0.0
    total_logp = 0.0
    with torch.no_grad():
        batches = loader.batcher()
        for batch_idx, data in enumerate(batches):
            data = data.cuda()
            recon, z, mu, log_var = vae(data)
            pickle.dump(recon.cpu(), open('./pkls/val'+str(epoch)+'.pkl','wb'))            
            # sum up batch loss
            val_loss += loss_function(recon, data, mu, log_var).item()
            if(epoch>19):
                logp = vae.iw_sample(data)
                total_logp += logp.unsqueeze(0).sum().item()
    total_logp /= loader.dataset_size
    val_loss /= loader.dataset_size
    print('====> Val set loss: {:.4f}'.format(val_loss))
    print('====> Val set log_p: {:.4f}'.format(total_logp))

def test(loader):
    vae.eval()
    test_loss= 0.0
    total_logp = 0.0
    loader.test_mode()
    with torch.no_grad():
        batches = loader.batcher()
        for batch_idx, data in enumerate(batches):
            data = data.cuda()
            recon, z, mu, log_var = vae(data)
            
            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()
            logp = vae.iw_sample(data)
            total_logp += logp.unsqueeze(0).sum().item()

    total_logp /= loader.dataset_size

    test_loss /= loader.dataset_size
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('====> Test set loss: {:.4f}'.format(total_logp))

class DataLoader():

    def __init__(self):
        
        self.ptr = 0
        self.train_data_size = 0
        self.val_data_size = 0
        self.test_data_size = 0
        self.binarized_mnist_fixed_binarization()
        self.batch_size = 128
        self.dataset_size = 0

    def binarized_mnist_fixed_binarization(self):
        DATASETS_DIR = './'
        def lines_to_np_array(lines):
            return np.array([[int(i) for i in line.split()] for line in lines])
        with open(os.path.join(DATASETS_DIR, 'BinaryMNIST', 'binarized_mnist_train.amat')) as f:
            lines = f.readlines()

        self.train_data = torch.tensor(lines_to_np_array(lines).astype('float32'))
        self.train_data = self.train_data.reshape([self.train_data.shape[0],img_size,img_size])

        with open(os.path.join(DATASETS_DIR, 'BinaryMNIST', 'binarized_mnist_valid.amat')) as f:
            lines = f.readlines()

        self.validation_data = torch.tensor(lines_to_np_array(lines).astype('float32'))
        self.validation_data = self.validation_data.reshape([self.validation_data.shape[0],img_size,img_size])

        with open(os.path.join(DATASETS_DIR, 'BinaryMNIST', 'binarized_mnist_test.amat')) as f:
            lines = f.readlines()

        self.test_data = torch.tensor(lines_to_np_array(lines).astype('float32'))
        self.test_data = self.test_data.reshape([self.test_data.shape[0],img_size,img_size])

        self.train_data_size = self.train_data.shape[0]
        self.val_data_size = self.validation_data.shape[0]
        self.test_data_size = self.test_data.shape[0]

        print(self.train_data.shape)        
        print(self.validation_data.shape)
        print(self.test_data.shape)

    def start_epoch(self):
        self.shuffle()
        self.ptr = 0
        self.data_in_use = self.train_data.clone()
        self.dataset_size = self.train_data_size

    def val_mode(self):
        self.ptr = 0
        self.data_in_use = self.validation_data.clone()
        self.dataset_size = self.val_data_size

    def test_mode(self):
        self.ptr = 0
        self.data_in_use = self.test_data.clone()
        self.dataset_size = self.test_data_size

    def shuffle(self):
        idx = np.arange(0 , self.train_data_size)
        np.random.shuffle(idx)
        self.train_data = self.train_data[idx]

    def batcher(self):
        # if(self.ptr + self.batch_size >= self.dataset_size):
        self.num_batches = int(loader.dataset_size/loader.batch_size)
        print('num batches '+str(self.num_batches))
        for batch_num in range(self.num_batches):
            self.ptr += self.batch_size
            yield torch.unsqueeze(self.data_in_use[ self.ptr-self.batch_size : self.ptr ], 1)


loader = DataLoader()

for epoch in range(1, 21):
    train(epoch, loader)
    val(loader, epoch)
test(loader)