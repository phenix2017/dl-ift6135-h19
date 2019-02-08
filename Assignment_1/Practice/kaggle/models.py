# https://github.com/pytorch/examples/blob/master/mnist/main.py
import os
import torch
import torch.nn as nn

from torch.nn.init import xavier_uniform_


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


class CnDClassifier(nn.Module):

    def __init__(self, state_dict_path=''):
        super(CnDClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.x_shape = [0, 64, 8, 8]
        self.linear_dim = 16
        self.fc1 = nn.Linear(self.x_shape[1]*self.x_shape[2]*self.x_shape[3], self.linear_dim)
        self.fc2 = nn.Linear(self.linear_dim, 2)

        if state_dict_path != '':
            # Check
            if os.path.exists(state_dict_path):
                print("Loading", state_dict_path)
            # Load pretrained model
            self.load_state_dict(torch.load(state_dict_path))
        else:
            self.apply(init_weights)

    def forward(self, x):
        # bx3x64x64
        x = self.conv1(x)   # bx16x64x64
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)   # bx16x32x32
        x = self.conv2(x)   # bx32x32x32
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)   # bx32x16x16
        x = self.conv3(x)   # bx64x16x16
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)   # bx64x8x8
        x = x.view(-1, self.x_shape[1]*self.x_shape[2]*self.x_shape[3])     # bx128*8*8
        x = self.fc1(x)     # bx16
        x = nn.ReLU()(x)
        x = self.fc2(x)     # bx2
        return nn.LogSoftmax(dim=1)(x)


# https://guillaumebrg.wordpress.com/2016/02/06/dogs-vs-cats-project-first-results-reaching-87-accuracy/
class CnDBigClassifier(nn.Module):

    def __init__(self, state_dict_path=''):
        super(CnDBigClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 0)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 0)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 0)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 0)
        self.x_shape = [0, 256, 2, 2]
        self.linear_dim = 128
        self.fc1 = nn.Linear(self.x_shape[1]*self.x_shape[2]*self.x_shape[3], self.linear_dim)
        self.fc2 = nn.Linear(self.linear_dim, 2)

        if state_dict_path != '':
            # Check
            if os.path.exists(state_dict_path):
                print("Loading", state_dict_path)
            # Load pretrained model
            self.load_state_dict(torch.load(state_dict_path))
        else:
            self.apply(init_weights)

    def forward(self, x):
        # bx3x64x64
        x = self.conv1(x)   # bx16x62x62
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)   # bx16x31x31
        x = self.conv2(x)   # bx32x29x29
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)   # bx32x14x14
        x = self.conv3(x)   # bx64x12x12
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)   # bx64x6x6
        x = self.conv4(x)   # bx128x4x4
        x = nn.ReLU()(x)
        x = self.conv5(x)   # bx256x2x2
        x = nn.ReLU()(x)
        x = x.view(-1, self.x_shape[1]*self.x_shape[2]*self.x_shape[3])     # bx256*2*2
        x = self.fc1(x)     # bx64
        x = nn.ReLU()(x)
        x = self.fc2(x)     # bx2
        return nn.LogSoftmax(dim=1)(x)


class TinyImageNetClassifier(nn.Module):

    def __init__(self, state_dict_path=''):
        super(TinyImageNetClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 0)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 0)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 0)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 0)
        self.x_shape = [0, 256, 2, 2]
        self.linear_dim1 = 512
        self.linear_dim2 = 256
        self.fc1 = nn.Linear(self.x_shape[1]*self.x_shape[2]*self.x_shape[3], self.linear_dim1)
        self.fc2 = nn.Linear(self.linear_dim1, self.linear_dim2)
        self.fc3 = nn.Linear(self.linear_dim2, 200)

        if state_dict_path != '':
            # Check
            if os.path.exists(state_dict_path):
                print("Loading", state_dict_path)
            # Load pretrained model
            self.load_state_dict(torch.load(state_dict_path))
        else:
            self.apply(init_weights)

    def forward(self, x):
        # bx3x64x64
        x = self.conv1(x)   # bx16x62x62
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)   # bx16x31x31
        x = self.conv2(x)   # bx32x29x29
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)   # bx32x14x14
        x = self.conv3(x)   # bx64x12x12
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)   # bx64x6x6
        x = self.conv4(x)   # bx128x4x4
        x = nn.ReLU()(x)
        x = self.conv5(x)   # bx256x2x2
        x = nn.ReLU()(x)
        x = x.view(-1, self.x_shape[1]*self.x_shape[2]*self.x_shape[3])     # bx256*2*2
        x = self.fc1(x)     # bx512
        x = nn.ReLU()(x)
        x = self.fc2(x)     # bx256
        x = nn.ReLU()(x)
        x = self.fc3(x)     # bx200
        return nn.LogSoftmax(dim=1)(x)


class TransferModel(nn.Module):

    def __init__(self, model_pth, model_weights_pth='', freeze_pt=False, device='cpu'):
        super(TransferModel, self).__init__()

        # Load model
        assert os.path.exists(model_pth)
        self.pt_model = torch.load(model_pth)
        if model_weights_pth != '':
            assert os.path.exists(model_weights_pth)
            self.pt_model.load_state_dict(torch.load(model_weights_pth))

        self.pt_model.to(device)

        # Freeze
        print("Freezing pretrained model params:", freeze_pt)
        for param in self.pt_model.parameters():
                param.requires_grad = False if freeze_pt else True

        # Cut off last layer
        self.pt_modulelist = list(self.pt_model.modules())[1:-1]
        self.pt_out_features = self.pt_modulelist[-1].out_features

        # Add a layer with 2 neurons (for Cat and Dog)
        self.fc1 = nn.Linear(self.pt_out_features, 2)

    def forward(self, x):
        # Run through pretrained model
        # till penultimate layer
        for module in self.pt_modulelist:
            x = module(x)

        # Run through FC layer
        x = self.fc1(x)

        # Return output
        return nn.LogSoftmax(dim=1)(x)
