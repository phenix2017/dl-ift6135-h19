import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from torchvision import transforms


def make_plots(losses, accuracy, log_interval, dataset_length, save_path, init_epoch=0):
    iters = np.arange(len(losses))*log_interval/dataset_length + init_epoch
    fig = plt.figure(figsize=(10, 20))
    plt.subplot(211)
    plt.plot(iters, np.zeros(iters.shape), 'k--', alpha=0.5)
    plt.plot(iters, losses, label='Loss')
    # plt.legend()
    plt.yscale("symlog")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.subplot(212)
    plt.plot(iters, accuracy, label='Accuracy')
    # plt.legend()
    plt.ylim([0, 1])
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join(save_path, "plots.png"), bbox_inches='tight', pad_inches=0.5)
    plt.clf()
    plt.close()


def make_transform(resize=False, imsize=64, centercrop=False, centercrop_size=128,
                   totensor=True, tanh_scale=True, normalize=False):
        options = []
        if resize:
            options.append(transforms.Resize((imsize, imsize)))
        if centercrop:
            options.append(transforms.CenterCrop(centercrop_size))
        if totensor:
            options.append(transforms.ToTensor())
        if tanh_scale:
            f = lambda x: x*2 - 1
            options.append(transforms.Lambda(f))
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform


def write_config_to_file(config, save_path):
    with open(os.path.join(save_path, 'config.txt'), 'w') as file:
        for arg in vars(config):
            file.write(str(arg) + ': ' + str(getattr(config, arg)) + '\n')

