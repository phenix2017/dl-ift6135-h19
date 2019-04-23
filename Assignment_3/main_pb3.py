import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import tqdm

import torchvision.utils as vutils

# from tensorboardX import SummaryWriter

from model import *
from utils import *
import scipy.misc as misc
from torch.autograd import Variable
from torchvision import utils

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

        if args.mode =='vae':
            self.args.image_log_dir = os.path.join(args.log_path_img, 'images/vae/')
            self.args.image_log_dir_parent = os.path.join(args.log_path_img, 'images/fid/vae/')

        elif args.mode =='gan':
            self.args.image_log_dir = os.path.join(args.log_path_img, 'images/gan/')
            self.args.image_log_dir_parent = os.path.join(args.log_path_img, 'images/fid/gan/')

        if not os.path.exists(self.args.image_log_dir):
            os.makedirs(self.args.image_log_dir)

        self.epochs = []
        self.losses = []

        self.kl_losses = []
        self.recon_losses = []

        self.real_losses = []
        self.fake_losses = []
        self.D_x = []
        self.D_G_z = []

        self.disc_label = torch.full((args.batch_size,), 1., device=self.device)

    def train_vae(self, num_epochs):
        train, valid, test = get_data_loader(self.args.dataset_location, 32)
        # writer = SummaryWriter(self.args.log_path)
        step = 0

        optim = torch.optim.Adam(self.model.parameters(),  lr=1e-4)

        iter = 0
        for epoch in range(num_epochs):
            for i, (x, y) in enumerate(train):
                iter+=1
                x = x.to(self.device)
                if( iter%self.args.save_every==0):
                    loss, kl , out= self.get_loss_vae(x, True, iter)
                else:
                    loss, kl , out= self.get_loss_vae(x)
                # print('vae loss: '+str(loss.cpu().item()))
                # print('vae kl: '+str(kl.cpu().item()))
                optim.zero_grad()
                loss.backward()
                optim.step()

            # step = self.store_logs_vae(writer, step, loss, kl, out, x)
        self.save(epoch)


    def train_gan(self, num_epochs):
        train, valid, test = get_data_loader(self.args.dataset_location, 64)
        self.dataloader = train
        # writer = SummaryWriter(self.args.log_path)
        step = 0

        optim_dis = torch.optim.Adam(self.model.discriminator.parameters(),betas=(0.5, 0.999), lr=1e-4)
        optim_gen = torch.optim.Adam(self.model.generator.parameters(),betas=(0.5, 0.999), lr=1e-4)
        iter = 0 
        for epoch in range(num_epochs):
            print('epoch '+str(epoch))

            for j in range(4):
                iter += 1
                real, _ = self.get_real_samples()
                if(iter%self.args.save_every==0):
                    loss_dis = self.get_loss_dis(real, True, iter)
                else:
                    loss_dis = self.get_loss_dis(real)
                optim_dis.zero_grad()
                loss_dis.backward()
                optim_dis.step()

            print('disc loss: '+str(loss_dis.cpu().item()))

            loss_gen = self.get_loss_gen()
            optim_gen.zero_grad()
            loss_gen.backward()
            optim_gen.step()

            print('gen loss: '+str(loss_gen.cpu().item()))

            # step = self.store_logs_gan(writer, step, loss_dis, loss_gen)
            step += 1

            if(epoch%self.args.save_every==0 and epoch>0):
                self.save(epoch)

    def load_vae_interpolate(self):
        train, valid, test = get_data_loader(self.args.dataset_location, 32)
        for i, (x, y) in enumerate(train):
            self.load(19)
            self.save_images()
            self.generate_latent_walk(0)
            self.generate_latent_walk_disentangled(0)
            self.generate_data_walk(x, 0)
            break

    def load_gan_interpolate(self):
        self.load(step)
        self.save_images()
        self.generate_latent_walk(0)
        self.generate_latent_walk_disentangled(0)

    def save_images(self):
        for i in range(16):
            output = self.model.generate(self.device)
            output_imgs_to_save = output.detach().cpu().add(1).mul(.5)
            # for img_i,_ in enumerate(output_imgs_to_save):
            #     vutils.save_image(output_imgs_to_save[img_i], os.path.join(self.args.image_log_dir_parent, 'output_{}_{}.png'.format(i,str(img_i))))
            vutils.save_image(output_imgs_to_save, os.path.join(self.args.image_log_dir, 'grid_output_{}.png'.format(i)))

    def get_real_samples(self):
        try:
            yield_values = next(self.dataloader)
        except:
            self.data_iter = iter(self.dataloader)
            yield_values = next(self.data_iter)

        real_images, real_labels = yield_values
        real_images, real_labels = real_images.to(self.device), real_labels.to(self.device)
        return real_images, real_labels


    def get_loss_vae( self, real, save=False, epoch=0, finalsave= False):
        output, mean, logvar = self.model(real)
        kl = get_kl2(mean, logvar)
        loss = (output-real).pow(2).sum() + kl
        print(loss.shape)
        print(kl.shape)
        print('epoch : '+str(epoch))
        print('vae loss: '+str(loss.mean().cpu().item()))
        print('vae kl: '+str(kl.mean().cpu().item()))
        print('vae output: '+str((output-real).pow(2).mean().cpu().item()))

        if save:
            real_imgs_to_save = real.detach().cpu().add(1).mul(.5)
            output_imgs_to_save = output.detach().cpu().add(1).mul(.5)

            vutils.save_image(real_imgs_to_save, os.path.join(self.args.image_log_dir, 'inputs.png'.format(epoch)))

            vutils.save_image(output_imgs_to_save, os.path.join(self.args.image_log_dir, 'output_{:05d}.png'.format(epoch)))

            ## save loss
            self.epochs.append(epoch)

            self.losses.append(loss.item())
            self.kl_losses.append(kl.mean().item())
            self.recon_losses.append((output-real).pow(2).mean().item())

            plt.plot(self.epochs, self.losses, color='C0', alpha=0.7, label='loss')
            plt.plot(self.epochs, self.kl_losses, color='C1', alpha=0.7, label='kl loss')
            plt.plot(self.epochs, self.recon_losses, color='C2', alpha=0.7, label='recon loss')
            plt.legend()
            save_path = os.path.join(self.args.image_log_dir, 'losses.png')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.5)

            plt.clf()
            plt.close()

        return loss, kl.mean(), output

    def get_loss_dis(self, real, save=False, epoch=0):
        fake = self.model.generate(self.device)

        out_real, out_fake = self.model.discriminate(real), self.model.discriminate(fake)

        # loss = out_fake.mean() - out_real.mean()

        gradient_penalty = calc_gradient_penalty(self.model.discriminator, real, fake)
        loss = out_fake.mean() - out_real.mean() + gradient_penalty

        if save:
            ## save images
            real_imgs_to_save = real.detach().cpu().add(1).mul(.5)
            fake_imgs_to_save = fake.detach().cpu().add(1).mul(.5)

            vutils.save_image(real_imgs_to_save, os.path.join(self.args.image_log_dir, 'real.png'.format(epoch)), nrow=8)
            vutils.save_image(fake_imgs_to_save, os.path.join(self.args.image_log_dir, 'fake_{:05d}.png'.format(epoch)), nrow=8)
            ## save loss
            self.epochs.append(epoch)
            self.losses.append(loss.item())
            self.real_losses.append(-out_real.mean().item())
            self.fake_losses.append(out_fake.mean().item())
            self.D_x.append(out_real.mean().item())
            self.D_G_z.append(out_fake.mean().item())
            plt.plot(self.epochs, self.losses, color='C0', alpha=0.7, label='loss')
            plt.plot(self.epochs, self.real_losses, color='C1', alpha=0.7, label='real loss')
            plt.plot(self.epochs, self.fake_losses, color='C2', alpha=0.7, label='fake loss')
            plt.legend()
            save_path = os.path.join(self.args.image_log_dir, 'D_losses_.png')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.5)
            plt.clf()
            plt.close()
            plt.plot(self.epochs, np.zeros((len(self.D_x,))), color='k', alpha=0.7)
            plt.plot(self.epochs, np.ones((len(self.D_x,))), color='k', alpha=0.7)
            plt.plot(self.epochs, self.D_x, color='C1', alpha=0.7, label='D(x)')
            plt.plot(self.epochs, self.D_G_z, color='C2', alpha=0.7, label='D(G(z))')
            plt.legend()
            # plt.ylim([-0.1, 1.1])
            save_path = os.path.join(self.args.image_log_dir, 'D_out.png')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.5)
            plt.clf()
            plt.close()

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

    def save(self, step):
        torch.save(self.model.cpu().state_dict(), os.path.join(self.args.save_path, self.args.saving_file + '_' + str(step) + '.pt'))
        self.model.to(self.device)

    def load(self, step):
        self.model.load_state_dict(torch.load(self.args.saved_save_path + self.args.saving_file + '_' + str(step) + '.pt'))
        self.model.to(self.device)

    def generate_latent_walk(self, number):
        self.model.eval()
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        print('interpolating')

        # Interpolate between the noise(z1, z2) with number_int steps between
        number_int = 10
        z_intp = torch.FloatTensor(1, 100)
        z1 = torch.randn(1, 100)
        z2 = torch.randn(1, 100)
        z_intp = z_intp.cuda()
        z1 = z1.cuda()
        z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha_step = 1.0 / float(number_int + 1)
        print(alpha_step)
        alpha = 0
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha_step 
            fake_im = self.model.generator(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            images.append(fake_im.view(3,32,32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int )
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved latent interpolated images to interpolated_images/interpolated_{}.".format(str(number).zfill(3)))

    def generate_latent_walk_disentangled(self, number):
        self.model.eval()
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')
        z = torch.randn(1, 100)

        for i in range(10,90):
            z1 = torch.clone(z).cuda()
            images = []
            lower_lim = -1.0
            step_size = 0.2
            for j in range(10):

                z1[0,i] = lower_lim + step_size
                lower_lim += step_size

                fake_im = self.model.generator(z1)
                fake_im = fake_im.mul(0.5).add(0.5) #denormalize
                images.append(fake_im.view(3,32,32).data.cpu())

            grid = utils.make_grid(images, nrow=10 )
            utils.save_image(grid, 'interpolated_images/varying_dim-{}_{}_epoch{}.png'.format(str(i),str(j),str(number).zfill(3)))           
            print("Saved disentangled interpolated images to interpolated_images/interpolated_{}.".format(str(number).zfill(3)))

    def generate_data_walk(self, real, number):
        self.model.eval()
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        print('interpolating')

        # Interpolate between the noise(z1, z2) with number_int steps between
        number_int = 10
        img1 = real[0]
        img2 = real[1]

        images = []
        alpha_step = 1.0 / float(number_int + 1)
        print(alpha_step)
        alpha = 0

        for i in range(1, number_int + 1):
            img = img1*alpha + img2*(1.0 - alpha)
            alpha += alpha_step 
            im = img.mul(0.5).add(0.5) #denormalize
            images.append(im.view(3,32,32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int )
        utils.save_image(grid, 'interpolated_images/data_interpolated_{}.png'.format(str(number).zfill(3)), nrow=8)
        print("Saved data interpolated images to interpolated_images/interpolated_{}.".format(str(number).zfill(3)))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.latent_dim = 100
    args.batch_size = 64
    args.use_cuda = True
    args.basepath = '/network/home/guptagun/code/dl/'
    args.log_path_img = args.basepath

    args.mode = 'gan'
    args.task = 'train' #'sample'

    if(args.mode=='gan'):
        args.log_path = args.basepath + 'saved/{0:%Y%m%d_%H%M%S}_wgan'.format(datetime.datetime.now())
    else:
        args.log_path = args.basepath + 'saved/{0:%Y%m%d_%H%M%S}_vae'.format(datetime.datetime.now())
    args.save_path = os.path.join(args.log_path, 'weights/')

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    args.saved_log_path = args.basepath + 'saved/20190423_063358_vae'
    args.saved_save_path = os.path.join(args.saved_log_path, 'weights/')


    args.saving_file = 'ckpt'
    args.dataset_location = args.basepath
    args.save_every = 100
    runner = trainer(args)
    if args.task == 'train':
        if(args.mode=='gan'):
            runner.train_gan(100000)
        else:
            runner.train_vae(40)
    else:
        if(args.mode=='gan'):
            runner.load_gan_interpolate()
        else:
            runner.load_vae_interpolate()



