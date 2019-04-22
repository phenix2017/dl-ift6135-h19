import argparse
import datetime
import numpy as np
import os
import torch
import tqdm

from tensorboardX import SummaryWriter

from model import *
from utils import *
import scipy.misc as misc

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
            self.args.image_log_dir = '/network/home/guptagun/code/dl/images/vae/'
        if args.mode =='gan':
            self.args.image_log_dir = '/network/home/guptagun/code/dl/images/gan/'

    def train_vae(self, num_epochs):
        train, valid, test = get_data_loader(self.args.dataset_location, 32)
        writer = SummaryWriter(self.args.log_path)
        step = 0

        optim = torch.optim.Adam(self.model.parameters(),  lr=3e-4)

        for epoch in range(num_epochs):
            for i, (x, y) in enumerate(train):
                x = x.to(self.device)
                loss, kl , out= self.get_loss_vae(x)
                print('vae loss: '+str(loss.cpu().item()))
                print('vae kl: '+str(kl.cpu().item()))
                optim.zero_grad()
                loss.backward()
                optim.step()

            step = self.store_logs_vae(writer, step, loss, kl, out, x)
            self.save(epoch)


    def train_gan(self, num_epochs):
        train, valid, test = get_data_loader(self.args.dataset_location, 32)
        self.dataloader = train
        writer = SummaryWriter(self.args.log_path)
        step = 0

        optim_dis = torch.optim.Adam(self.model.discriminator.parameters(), betas=(0.5,0.9),lr=1e-4)
        optim_gen = torch.optim.Adam(self.model.generator.parameters(), betas=(0.5,0.9),lr=1e-4)

        for epoch in range(num_epochs):
            print('epoch '+str(epoch))

            for j in range(5):
                real, _ = self.get_real_samples()
                if(j==0 and epoch%self.args.save_every==0):
                    loss_dis = self.get_loss_dis(real, True, epoch)
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

            step = self.store_logs_gan(writer, step, loss_dis, loss_gen)

            if(epoch%self.args.save_every==0 and epoch>0):
                self.save(epoch)

        generate_latent_walk(epoch)

    def get_real_samples(self):
        try:
            yield_values = next(self.dataloader)
        except:
            self.data_iter = iter(self.dataloader)
            yield_values = next(self.data_iter)

        real_images, real_labels = yield_values
        real_images, real_labels = real_images.to(self.device), real_labels.to(self.device)
        return real_images, real_labels


    def get_loss_vae(self, real ):
        output, mean, logvar = self.model(real)

        kl = get_kl(mean, logvar)

        loss = (output-real).pow(2).mean() + kl.mean()

        return loss, kl.mean(), output

    def get_loss_dis(self, real, save=False, epoch=0):
        fake = self.model.generate(self.device)

        if save:
            ## save images
            imgs_to_save = fake.detach().cpu().numpy()
            imgs_to_save = np.transpose(imgs_to_save , (0,2,3,1))
            for i,img in enumerate(imgs_to_save):
                misc.imsave(self.args.image_log_dir + str(epoch)+ '_' +str(i)+'.png', img)

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

    def save(self, step):
        torch.save(self.model.cpu().state_dict(), os.path.join(self.args.save_path, self.args.saving_file + '_' + str(step) + '.pt'))
        self.model.to(self.device)

    def load(self, step):
        self.model.load_state_dict(torch.load(self.args.save_path, self.args.saving_file + '_' + str(step) + '.pt'))
        self.model.to(self.device)

    # def save_model(self):
    #     torch.save(self.model.generator.state_dict(), './generator.pkl')
    #     torch.save(self.model.discriminator.state_dict(), './discriminator.pkl')
    #     print('Models save to ./generator.pkl & ./discriminator.pkl ')

    # def load_model(self, D_model_path, G_model_path):
    #     self.model.discriminator.load_state_dict(torch.load(D_model_path))
    #     self.model.generator.load_state_dict(torch.load(G_model_path))
    #     print('Generator model loaded from {}.'.format(G_model_path))
    #     print('Discriminator model loaded from {}-'.format(D_model_path))

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        # Interpolate between twe noise(z1, z2) with number_int steps between
        number_int = 10
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.model.generator(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            images.append(fake_im.view(3,32,32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int )
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images to interpolated_images/interpolated_{}.".format(str(number).zfill(3)))

    def generate_latent_walk_disentangled(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        sigma_mean = torch.ones((100))
        mu_mean = torch.zeros((100))

        # Interpolate between twe noise(z1, z2) with number_int steps between
        number_int = 10
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)          ### ? ?
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.model.generator(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            images.append(fake_im.view(3,32,32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int )
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images to interpolated_images/interpolated_{}.".format(str(number).zfill(3)))


    def stub():
            # Save generated variable images :
            var_z = betavae.encoder(var_x)
            mu_z, log_var_z = torch.chunk(var_z, 2, dim=1 )
            mu_mean /= batch_size
            
            sigma_mean /= batch_size
            gen_images = None

            for latent in range(z_dim) :
                #var_z0 = torch.stack( [mu_mean]*nbr_steps, dim=0)
                #var_z0 = torch.zeros(nbr_steps, z_dim)
                var_z0 = mu_z.cpu().data
                val = mu_mean[latent]-sigma_mean[latent]
                step = 2.0*sigma_mean[latent]/nbr_steps
                print(latent,mu_mean[latent],step)
                for i in range(nbr_steps) :
                    var_z0[i] = mu_mean
                    var_z0[i][latent] = val
                    val += step

                var_z0 = Variable(var_z0)
                if use_cuda :
                    var_z0 = var_z0.cuda()


                gen_images_latent = betavae.decoder(var_z0)
                gen_images_latent = gen_images_latent.view(-1, img_depth, img_dim, img_dim).cpu().data
                if gen_images is None :
                    gen_images = gen_images_latent
                else :
                    gen_images = torch.cat( [gen_images,gen_images_latent], dim=0)

            #torchvision.utils.save_image(gen_images.data.cpu(),'./beta-data/{}/gen_images/dim{}/{}.png'.format(path,latent,(epoch+1)) )
            torchvision.utils.save_image(255.0*gen_images,'./beta-data/{}/gen_images/{}.png'.format(path,(epoch)) )



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.latent_dim = 200
    args.batch_size = 32
    args.use_cuda = True
    args.log_path = '/network/home/guptagun/code/dl/wgan/{0:%Y%m%d_%H%M%S}_wgan'.format(datetime.datetime.now())
    args.save_path = os.path.join(args.log_path, 'weights')
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    args.saving_file = 'ckpt'
    args.dataset_location = '/network/home/guptagun/code/dl'
    args.save_every = 50
    args.mode = 'gan'
    runner = trainer(args)
    runner.train_gan(100)

