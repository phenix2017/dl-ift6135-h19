import gc
import numpy as np
import os
import torch
import torchvision.utils as vutils
import tqdm

import utils


class MyArgs():
    def __init__(self, data_path):
        self.data_path = data_path
        self.valid_split = 0
        self.eval = True
        self.imsize = 64
        self.seed = 29
        self.batch_size = 128
        self.shuffle = False
        self.drop_last = False
        self.val_data_path = ''
        self.kwargs = {}


# special_data_path = "/home/voletiv/Datasets/CatsAndDogs/special"
special_data_path = "/home/voletivi/scratch/catsndogs/data/special"

# Data
args = MyArgs(special_data_path)
dl = utils.make_dataloader(args)
data, classes = next(iter(dl))

# Model
# model_pth = '/home/voletiv/EXPERIMENTS/CnD_experiments/experiments/20190211_172747_cnd_kaggle_ResNet18_skip_expDecay/model.pth'
# model_state_dict = '/home/voletiv/EXPERIMENTS/CnD_experiments/experiments/20190211_172747_cnd_kaggle_ResNet18_skip_expDecay/model_epoch_0082_batch_00076_of_00282.pth'
model_pth = '/home/voletivi/scratch/catsndogs/experiments/20190211_172747_cnd_kaggle_ResNet18_skip_expDecay/model.pth'
model_state_dict = '/home/voletivi/scratch/catsndogs/experiments/20190211_172747_cnd_kaggle_ResNet18_skip_expDecay/model_epoch_0082_batch_00076_of_00282.pth'
model = torch.load(model_pth)
model.load_state_dict(torch.load(model_state_dict))
# model.to('cpu')

occ_size = 7

# Data occ sample
data_occ = data.clone()
i, j = 20, 20
left = max(i-occ_size//2, 0)
right = min(i+occ_size//2, data.shape[2]-1)
up = max(j-occ_size//2, 0)
down = min(j+occ_size//2, data.shape[3]-1)
data_occ[:, :, left:right, up:down] = 0
vutils.save_image(data_occ.add(1.).div(2.), 'data_occ.png', nrow=4)

# Model output with data occlusion
data_shape = list(data.shape)
stride = 1
occluded_data = torch.empty(0, data_shape[1], data_shape[2], data_shape[3])
for i in tqdm.tqdm(range(data.shape[2]//stride)):
    for j in tqdm.tqdm(range(data.shape[3]//stride)):
        # gc.collect();
        data_occ = data.clone()
        left = max(i*stride-occ_size//stride, 0)
        right = min(i*stride+occ_size//stride, data.shape[2]-1)
        up = max(j*stride-occ_size//stride, 0)
        down = min(j*stride+occ_size//stride, data.shape[3]-1)
        data_occ[:, :, left:right, up:down] = 0
        occluded_data = torch.cat((occluded_data, data_occ), dim=0)

batch_size = 12
8model_outs = np.empty((0, 2))
for i in tqdm.tqdm(range(len(occluded_data)//batch_size)):
    data_occ = occluded_data[i*batch_size:(i+1)*batch_size]
    data_occ = data_occ.to('cuda')
    model_outs = np.vstack((model_outs, np.exp(model(data_occ).detach().cpu().numpy())))

# Extract respective probs
mask = np.array([0, 0, 1, 1] * (len(model_outs)//4)).reshape(-1, 1)
mask = np.hstack((mask, 1-mask))
model_probs = np.sum(model_outs*mask, axis=1)

# Reshape to nx4
batch_probs = model_probs.reshape(-1, 4)

# Make heatmaps
im = np.zeros((data_shape[2]//stride, data_shape[3]//stride, 4))
for i in range(data_shape[2]//stride):
    for j in range(data_shape[3]//stride):
        im[i, j] = batch_probs[i*(data_shape[2]//stride) + j]

im_t = torch.from_numpy(np.expand_dims(im.transpose(2, 0, 1), 1))
vutils.save_image(im_t, 'heatmap.png', nrow=4)

im_t_log = im_t.log()
for i in range(len(im_t_log)):
    im_t_log[i] = (im_t_log[i] - im_t_log[i].min())/(im_t_log[i].max() - im_t_log[i].min())

vutils.save_image(im_t_log, 'heatmap_log.png', nrow=4)

im_t_sqrt = im_t.sqrt()
for i in range(len(im_t_sqrt)):
    im_t_sqrt[i] = (im_t_sqrt[i] - im_t_sqrt[i].min())/(im_t_sqrt[i].max() - im_t_sqrt[i].min())

vutils.save_image(im_t_sqrt, 'heatmap_sqrt.png', nrow=4)

im_data = data.add(1.).div(2.)
vutils.save_image(im_data, 'data.png', nrow=4)

masked_data = im_t_log.float()*im_data.float()
vutils.save_image(.1*im_data + .9*masked_data, 'heatmap_mask_log.png', nrow=4)

masked_data = im_t_sqrt.float()*im_data.float()
vutils.save_image(.1*im_data + .9*masked_data, 'heatmap_mask_sqrt.png', nrow=4)
