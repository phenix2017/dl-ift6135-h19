import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import torch

from torchvision import datasets as dset
from torchvision import transforms


def check_for_CUDA(args):
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.use_cuda else "cpu")
    args.kwargs = {'num_workers': 1, 'pin_memory': True} if args.use_cuda else {}


def copy_scripts(dst):
    for file in glob.glob('*.py'):
        shutil.copy(file, dst)


def make_transform(resize=False, imsize=64, centercrop=False, centercrop_size=128,
                   tanh_scale=True, normalize=False, norm_mean=(0.5, 0.5, 0.5), norm_std=(0.5, 0.5, 0.5)):
        options = []
        if resize:
            options.append(transforms.Resize((imsize, imsize)))
        if centercrop:
            options.append(transforms.CenterCrop(centercrop_size))
        options.append(transforms.ToTensor())
        if tanh_scale:
            f = lambda x: x*2 - 1
            options.append(transforms.Lambda(f))
        if normalize:
            options.append(transforms.Normalize(norm_mean, norm_std))
        transform = transforms.Compose(options)
        return transform


def write_config_to_file(config, save_path):
    with open(os.path.join(save_path, 'config.txt'), 'w') as file:
        for arg in vars(config):
            file.write(str(arg) + ': ' + str(getattr(config, arg)) + '\n')


def make_dataloader(args):
    transform = make_transform(args.resize, args.imsize, args.centercrop, args.centercrop_size, args.tanh_scale, args.normalize)
    assert os.path.exists(args.data_path), "data_path does not exist! Given: " + args.data_path
    dataset = dset.ImageFolder(root=args.data_path, transform=transform)
    args.num_of_classes = sum([1 if os.path.isdir(os.path.join(args.data_path, i)) else 0 for i in os.listdir(args.data_path)])
    print("Data found! # of classes =", args.num_of_classes, ", # of images =", len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=args.drop_last, **args.kwargs)
    return dataloader


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
