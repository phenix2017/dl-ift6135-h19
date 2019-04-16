import torch
import numpy as np
from model import *
from utils import *
from tensorboardX import SummaryWriter
import argparse


class trainer():
    def __init__(self, args):
        self.args = args

        self.device = torch.device('cpu')
        if self.args.use_cuda :
            self.device = torch.device('cuda')


        if args.mode =='vae':
            self.model = VAE(args).to(self.device)
        if args.mode =='gan':
            self.model = GAN(args).to(self.device)

    def train_vae(self, num_epochs):
        train, valid, test = get_data_loader("svhn", 32)
        writer = SummaryWriter(self.args.log_path)
        step = 0

        optim = torch.optim.Adam(self.model.parameters(),  lr=3e-4)


        for epoch in range(num_epochs):
            for i, (x, y) in enumerate(train):
                x = x.to(self.device)
                loss, kl , out= self.get_loss_vae(x)

                optim.zero_grad()
                loss.backward()
                optim.step()

            step = self.store_logs_vae(writer, step, loss, kl, out, x)


    def train_gan(self, num_epochs):
        train, valid, test = get_data_loader("svhn", 32)
        writer = SummaryWriter(self.args.log_path)
        step =0

        optim_dis = torch.optim.Adam(self.model.discriminator.parameters(),betas=(0.,0.9),lr = 1e-4)
        optim_gen = torch.optim.Adam(self.model.generator.parameters(),betas=(0.,0.9),lr = 1e-4)

        for epoch in range(num_epochs):

            for i, (x, y) in enumerate(train):
                real = x.to(self.device)
                for j in range(10):
                    loss_dis = self.get_loss_dis(real)
                    optim_dis.zero_grad()
                    loss_dis.backward()
                    optim_dis.step()

                loss_gen = self.get_loss_gen()
                optim_gen.zero_grad()
                loss_gen.backward()
                optim_gen.step()


                step = self.store_logs_gan(writer, step, loss_dis, loss_gen)



    def get_loss_vae(self, real ):
        output, mean, logvar = self.model(real)

        kl = get_kl(mean, logvar)

        loss = (output-real).pow(2).mean() + kl.mean()

        return loss, kl.mean(), output

    def get_loss_dis(self, real):
        fake = self.model.generate(self.device)

        out_real, out_fake = self.model.discriminate(real), self.model.discriminate(fake)

        gradient_penalty = calc_gradient_penalty(self.model.discriminator, real,fake)

        loss = out_fake.mean() - out_real.mean() + gradient_penalty

        return loss

    def get_loss_gen(self):
        fake = self.model.generate(self.device)
        out_fake = self.model.discriminate(fake)

        loss = -out_fake.mean()

        return loss

    def store_logs_gan(self, writer, step, loss_dis, loss_gen):

        writer.add_scalars('losses', {'dis':loss_dis,
                                      'gen':loss_gen}, step)

        sample = self.model.generate(self.device)[0]


        writer.add_image('generated', sample, global_step = step)

        step +=1
        return step

    def store_logs_vae(self, writer, step, loss, kl , out, x):
        writer.add_scalars('losses', {'tot':loss,
                                      'kl':kl}, step)

        sample = self.model.generate(out.device)[0]

        writer.add_image('generated', sample, global_step=step)

        writer.add_image('original', x[0], step)
        writer.add_image('rec',  out[0], step)
        step +=1
        return step




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.latent_dim = 100
    args.batch_size = 32
    args.use_cuda = True
    args.log_path = '/network/home/assouelr/DL_assignment/logs/wgan/'
    #args.log_path = '/Users/rimassouel/PycharmProjects/DL_assignment/logs/wgan/'

    args.mode = 'gan'
    runner = trainer(args)
    runner.train_gan(100)












