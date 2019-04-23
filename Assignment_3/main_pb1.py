import torch
from torch import nn
from model import Discriminator
from samplers import *

import argparse
import matplotlib.pyplot as plt
from utils import *


class trainer():
    def __init__(self, args):
        self.args = args
        self.net = Discriminator(args)

    def train(self, d0, d1, iterations):
        optim = torch.optim.SGD(self.net.parameters(), lr=1e-3)
        losses = []

        for it in range(iterations):
            samples_0, samples_1 = next(d0), next(d1)

            samples_0, samples_1 = torch.from_numpy(samples_0).float(), torch.from_numpy(samples_1).float()

            loss = self.get_loss(samples_0, samples_1)
            losses.append(loss)

            optim.zero_grad()
            loss.backward()
            optim.step()

        #torch.save(self.net.state_dict(), self.args.save_path + self.args.save_file)
        return losses

    def get_loss(self, samples_0, samples_1):

        out_0, out_1 = self.net(samples_0), self.net(samples_1)

        if self.args.distance == 'js':
            neg_js = (- 0.5*torch.log(out_0 + 1e-8) - 0.5*torch.log(1- out_1 + 1e-8)).mean()
            return neg_js

        elif self.args.distance =='wd':
            wd_loss = (out_1 - out_0).mean() + calc_gradient_penalty(self.net, samples_0, samples_1)
            return wd_loss

    def eval_q3(self, iterations):
        thetas = np.arange(-1, 1,  0.1)

        dist_0 = distribution1(0)
        js_sq = []

        for theta in thetas :
            dist_1 = distribution1(theta)
            losses= self.train(dist_0, dist_1, iterations)
            js = self.get_distance(dist_0,dist_1,self.net)
            js_sq.append(js)


        return js_sq

    def eval_q4(self, iterations):
        dist_0 = distribution4()
        dist_1 = distribution3()


        losses = self.train(dist_0, dist_1,iterations)

        plt.plot(losses)
        plt.show()

        xx = torch.randn((10000,1))
        N = lambda x: np.exp(-x ** 2 / 2.) / ((2 * np.pi) ** 0.5)
        f0 = torch.from_numpy(N(xx.numpy()))
        out = self.net(xx)
        f1 = f0*torch.div(out, 1-out)
        self.plot(f1,xx, out)

        return f1, xx, out



    def get_distance(self,dist_0, dist_1, net):
        samples_0, samples_1 = next(dist_0), next(dist_1)
        samples_0, samples_1 = torch.from_numpy(samples_0).float(), torch.from_numpy(samples_1).float()

        out_0, out_1 = net(samples_0), net(samples_1)
        if self.args.distance =='js':
            d = np.log(2) +0.5*(torch.log(out_0) ).mean().detach().numpy() +\
                 0.5*(torch.log(1-out_1 )).mean().detach().numpy()

        else :
            d = out_0.mean().detach().numpy() - out_1.mean().detach().numpy()

        return d

    def plot(self, f1, xx, out):
        r = out.detach().numpy()
        xx = xx.numpy()

        f = lambda x: torch.tanh(x * 2 + 1) + x * 0.75
        d = lambda x: (1 - torch.tanh(x * 2 + 1) ** 2) * 2 + 0.75
        N = lambda x: np.exp(-x ** 2 / 2.) / ((2 * np.pi) ** 0.5)

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(xx, r)
        plt.title(r'$D(x)$')

        estimate = f1.detach().numpy()  # estimate the density of distribution4 (on xx) using the discriminator;
        # replace "np.ones_like(xx)*0." with your estimate
        plt.subplot(1, 2, 2)
        plt.plot(xx, estimate)
        plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy() ** (-1) * N(xx))
        plt.legend(['Estimated', 'True'])
        plt.title('Estimated vs True')
        plt.show()

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.save_path = '/Users/rimassouel/PycharmProjects/DL_assignment/weights/'
    iterations = 15000
    args.in_dim = 1
    args.distance = 'js'

    runner = trainer(args)

    f1, xx, out = runner.eval_q4(iterations)

    br
    iterations = 200
    args.in_dim = 2
    args.distance = 'js'

    dist_0 = distribution1(0)
    dist_1 = distribution1(1)

    runner = trainer(args)

    for i in range(10):
        js = runner.eval_q3(iterations)
        thetas = np.arange(-1, 1, 0.1)
        plt.title('Estimated JS after '+str((i+1)*500)+' training steps')
        plt.plot(thetas ,js)
        plt.show()

    args.distance = 'wd'

    dist_0 = distribution1(0)
    dist_1 = distribution1(1)

    runner = trainer(args)

    for i in range(10):
        js = runner.eval_q3(iterations)
        thetas = np.arange(-1, 1, 0.1)
        plt.title('Estimated WD after ' + str((i + 1) * 500) + ' training steps')
        plt.plot(thetas, js)
        plt.show()



