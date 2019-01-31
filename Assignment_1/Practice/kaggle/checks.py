import os
import imageio

top_dir = '/home/voletivi/scratch/catsndogs/data'

# Kaggle dir
kaggle_dir = os.path.join(top_dir, 'kaggle')
# This contains:
# - 'trainset'
#     - 'Cat': 0.Cat.jpg, ...
#     - 'Dog': 10.Dog.jpg, ...
# - 'testset'
#     - 'test': 0.jpg, ...

# PetImages dir
pet_images_dir = os.path.join(top_dir, 'PetImages')
# This contains:
# - 'Cat': 0.jpg, ...
# - 'Dog': 0.jpg, ...

# Check images size
kaggle_image_file = os.path.join(kaggle_dir, 'trainset', 'Cat', '1.Cat.jpg')
kaggle_image = imageio.imread(kaggle_image_file)
kaggle_image.shape
# (64, 64, 3)
pet_image_file = os.path.join(pet_images_dir, 'Cat', '1.Cat.jpg')
pet_image = imageio.imread(kaggle_image_file)
pet_image.shape
# (64, 64, 3)

# Check network
import torch
import torch.nn as nn
x = torch.randn(1, 3, 64, 64)
nn.MaxPool2d(2, 2)(nn.Conv2d(3, 64, 3, 1, 1)(x)).shape

import torch.utils.data
from torchvision import datasets, transforms
train_loader = torch.utils.data.DataLoader(
         datasets.MNIST('/home/user1/Datasets', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
         batch_size=3, shuffle=True)

############################

kaggle_dir = '/home/user1/Datasets/CatsAndDogs'
# This contains:
# - 'trainset'
#     - 'Cat': 0.Cat.jpg, ...
#     - 'Dog': 10.Dog.jpg, ...
# - 'testset'
#     - 'test': 0.jpg, ...

transform = utils.make_transform()
dataset = dset.ImageFolder(root=kaggle_dir, transform=transform)
num_of_classes = sum([1 if os.path.isdir(os.path.join(kaggle_dir, i)) else 0 for i in os.listdir(kaggle_dir)])

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

