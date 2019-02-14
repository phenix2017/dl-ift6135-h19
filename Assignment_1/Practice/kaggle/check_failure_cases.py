import csv
import imageio
import numpy as np
import os
import torch
import torchvision.utils as vutils

train_data_dir = '/home/voletiv/Datasets/CatsAndDogs/trainset'

train_result_dir = '/home/voletiv/EXPERIMENTS/CnD_experiments/experiments/20190211_172747_cnd_kaggle_ResNet18_skip_expDecay/train_result'
cat_result_file = os.path.join(train_result_dir, 'Cat.csv')
dog_result_file = os.path.join(train_result_dir, 'Dog.csv')

cat_lines = []
with open(cat_result_file, mode='r') as csv_file:
    for line in csv_file:
        cat_lines.append(line.strip())

cat_lines = cat_lines[1:]
wrong_cats = [os.path.join(train_data_dir, 'Cat', line.split(',')[0] + '.Cat.jpg') for line in cat_lines if line.split(',')[-1] != 'Cat']

dog_lines = []
with open(dog_result_file, mode='r') as csv_file:
    for line in csv_file:
        dog_lines.append(line.strip())

dog_lines = dog_lines[1:]
wrong_dogs = [os.path.join(train_data_dir, 'Dog', line.split(',')[0] + '.Dog.jpg') for line in dog_lines if line.split(',')[-1] != 'Dog']

wrong_ims = wrong_cats + wrong_dogs

choice_idx = np.random.choice(len(wrong_ims), 100, replace=False)
ims = []
for idx in choice_idx:
    ims.append(imageio.imread(wrong_ims[idx]))

ims_np = np.array(ims)
ims_np = np.transpose(ims, (0, 3, 1, 2))
ims_t = torch.from_numpy(ims_np).float().div(255)

vutils.save_image(ims_t, os.path.join(train_result_dir, 'wrongs.png'), nrow=10)
sum(p.numel() for p in model.parameters())

