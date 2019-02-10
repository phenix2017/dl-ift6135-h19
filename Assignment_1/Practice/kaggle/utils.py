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


def copy_scripts(dst, src='.'):
    for file in glob.glob(os.path.join(src, '*.py')):
        shutil.copy(file, dst)


def get_time_str(start_time, curr_time):
    curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
    delta = datetime.timedelta(seconds=(curr_time - start_time))
    delta_str = str(delta - datetime.timedelta(microseconds=delta.microseconds))
    return curr_time_str, delta_str


def make_transform(eval=False, imsize=64):
    f = lambda x: x*2 - 1
    options = []
    if 'eval':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(imsize, scale=(0.08, 1.0), ratio=(1, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.RandomAffine(10, translate=(.2, .2), scale=(.8, 1.2), shear=1),
            transforms.ToTensor(),
            transforms.Lambda(f)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(f)
        ])
    return transform


# def make_transform(resize=False, imsize=64, centercrop=False, centercrop_size=128,
#                    tanh_scale=True, normalize=False, norm_mean=(0.5, 0.5, 0.5), norm_std=(0.5, 0.5, 0.5)):
#         options = []
#         if resize:
#             options.append(transforms.Resize((imsize, imsize)))
#         if centercrop:
#             options.append(transforms.CenterCrop(centercrop_size))
#         options.append(transforms.ToTensor())
#         if tanh_scale:
#             f = lambda x: x*2 - 1
#             options.append(transforms.Lambda(f))
#         if normalize:
#             options.append(transforms.Normalize(norm_mean, norm_std))
#         transform = transforms.Compose(options)
#         return transform


def write_config_to_file(config, save_path):
    with open(os.path.join(save_path, 'config.txt'), 'w') as file:
        for arg in vars(config):
            file.write(str(arg) + ': ' + str(getattr(config, arg)) + '\n')


def make_dataloader(args):
    # Make transforms
    # transform = make_transform(args.resize, args.imsize, args.centercrop, args.centercrop_size, args.tanh_scale, args.normalize)
    # Make dataset
    assert os.path.exists(args.data_path), "data_path does not exist! Given: " + args.data_path

    # If no split
    if args.valid_split == 0 or args.val_data_path != '':
        transform = make_transform(args.eval, args.imsize)
        dataset = dset.ImageFolder(root=args.data_path, transform=transform)
        args.num_of_classes = len(dataset.classes)
        print("Data found! # of classes =", args.num_of_classes, ", # of images =", len(dataset))
        print("Classes:", dataset.classes)
        torch.manual_seed(args.seed)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=args.drop_last, **args.kwargs)
        if args.val_data_path == '':
            return dataloader
        else:
            val_dataset = dset.ImageFolder(root=args.val_data_path, transform=make_transform(eval=True, imsize=args.imsize))
            print("Val Data found! # of classes =", len(val_dataset.classes), ", # of images =", len(val_dataset))
            print("Classes:", val_dataset.classes)
            torch.manual_seed(args.seed)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=args.drop_last, **args.kwargs)
            return dataloader, val_dataloader

    # If data needs to be split into Train and Val
    else:
        # https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
        train_transform = make_transform(args.eval, args.imsize)
        train_dataset = dset.ImageFolder(root=args.data_path, transform=train_transform)
        val_transform = make_transform(True, args.imsize)
        valid_dataset = dset.ImageFolder(root=args.data_path, transform=val_transform)
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


def make_plots(save_path, train_iters, train_losses, train_accuracy,
               valid_iters, valid_losses, valid_accuracy, lr_change=[], init_epoch=0):
    fig = plt.figure(figsize=(10, 20))
    plt.subplot(211)
    plt.plot(train_iters, np.zeros((len(train_iters))), 'k--', alpha=0.5)
    plt.plot(train_iters, train_losses, label='Train Loss')
    plt.plot(valid_iters, valid_losses, label='Valid Loss')
    for x in lr_change:
        plt.axvline(x=x, linestyle='--', color='k')
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.grid()
    plt.subplot(212)
    plt.plot(train_iters, 0.5*np.ones((len(train_iters))), 'k--', alpha=0.5)
    plt.plot(train_iters, train_accuracy, label='Train Acc')
    plt.plot(valid_iters, valid_accuracy, label='Valid Acc')
    for x in lr_change:
        plt.axvline(x=x, linestyle='--', color='k')
    plt.legend()
    plt.ylim([0, 1])
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.grid()
    plt.savefig(os.path.join(save_path, "plots.png"), bbox_inches='tight', pad_inches=0.5)
    plt.clf()
    plt.close()


def imshow(inp, title=None):
    # del sys.modules['matplotlib']
    # import matplotlib
    # matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0)  # pause a bit so that plots are updated
