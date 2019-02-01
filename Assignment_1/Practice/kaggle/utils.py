import datetime
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import shutil
import torch

from torchvision import datasets as dset
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def mem_check():
    import os
    import psutil
    process = psutil.Process(os.getpid())
    print("Mem:", process.memory_info().rss/1024/1024/1024, "GB")


def check_for_CUDA(args):
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Use CUDA:", args.use_cuda)
    args.device = torch.device("cuda" if args.use_cuda else "cpu")
    args.kwargs = {'num_workers':4, 'pin_memory':True} if args.use_cuda else {}


def copy_scripts(dst):
    for file in glob.glob('*.py'):
        shutil.copy(file, dst)


def get_time_elapsed_str(time_diff):
    delta = datetime.timedelta(seconds=time_diff)
    return str(delta - datetime.timedelta(microseconds=delta.microseconds))


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
    # Make transforms
    transform = make_transform(args.resize, args.imsize, args.centercrop, args.centercrop_size, args.tanh_scale, args.normalize)
    # Make dataset
    assert os.path.exists(args.data_path), "data_path does not exist! Given: " + args.data_path
    # If no split
    if args.valid_split == 0:
        dataset = dset.ImageFolder(root=args.data_path, transform=transform)
        args.num_of_classes = sum([1 if os.path.isdir(os.path.join(args.data_path, i)) else 0 for i in os.listdir(args.data_path)])
        print("Data found! # of classes =", args.num_of_classes, ", # of images =", len(dataset))
        print("Classes:", dataset.classes)
        torch.manual_seed(args.seed)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=args.drop_last, **args.kwargs)
        return dataloader
    # If data needs to be split into Train and Val
    else:
        # https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
        train_dataset = dset.ImageFolder(root=args.data_path, transform=transform)
        valid_dataset = dset.ImageFolder(root=args.data_path, transform=transform)
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(args.valid_split * num_train))
        # Shuffle
        if args.shuffle:
            np.random.seed(args.seed)
            np.random.shuffle(indices)
        # Train & Val indices
        train_idx, valid_idx = indices[split:], indices[:split]
        print("Train images #:", len(train_idx), "; Valid images #:", len(valid_idx))
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        # Train & Val dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, **args.kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler, **args.kwargs)
        return train_loader, valid_loader


def make_plots(train_losses, train_accuracy, log_interval, train_iters_per_epoch, save_path,
               valid_losses, valid_accuracy, init_epoch=0):
    train_iters = np.arange(len(train_losses))*log_interval/train_iters_per_epoch + init_epoch
    valid_iters = np.arange(len(valid_losses)) + init_epoch
    fig = plt.figure(figsize=(10, 20))
    plt.subplot(211)
    plt.plot(train_iters, np.zeros(train_iters.shape), 'k--', alpha=0.5)
    plt.plot(train_iters, train_losses, label='Train Loss')
    plt.plot(valid_iters, valid_losses, label='Valid Loss')
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.subplot(212)
    plt.plot(train_iters, train_accuracy, label='Train Acc')
    plt.plot(valid_iters, valid_accuracy, label='Valid Acc')
    plt.legend()
    plt.ylim([0, 1])
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join(save_path, "plots.png"), bbox_inches='tight', pad_inches=0.5)
    plt.clf()
    plt.close()
