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

    def __init__(self, state_dict_path='', freeze_pt=False, keep_last=True):
        super(TinyImageNetClassifier, self).__init__()

        self.keep_last = keep_last

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
            self.to('cpu')
        else:
            self.apply(init_weights)

        # Freeze
        print("Freezing pretrained model params:", freeze_pt)
        for param in self.parameters():
            param.requires_grad = False if freeze_pt else True

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

        if not self.keep_last:
            return x

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

        self.x_shape = [0, 256, 2, 2]

    def forward(self, x):
        # Run through pretrained model
        # till penultimate layer
        # import pdb; pdb.set_trace()
        # bx3x64x64
        x = self.pt_modulelist[0](x)   # bx16x62x62
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)   # bx16x31x31
        x = self.pt_modulelist[1](x)   # bx32x29x29
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)   # bx32x14x14
        x = self.pt_modulelist[2](x)   # bx64x12x12
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)   # bx64x6x6
        x = self.pt_modulelist[3](x)   # bx128x4x4
        x = nn.ReLU()(x)
        x = self.pt_modulelist[4](x)   # bx256x2x2
        x = nn.ReLU()(x)
        x = x.view(-1, self.x_shape[1]*self.x_shape[2]*self.x_shape[3])     # bx256*2*2
        x = self.pt_modulelist[5](x)     # bx512
        x = nn.ReLU()(x)
        x = self.pt_modulelist[6](x)     # bx256
        x = nn.ReLU()(x)

        # Run through FC layer
        x = self.fc1(x)

        # Return output
        return nn.LogSoftmax(dim=1)(x)


class CnDSkipClassifier(nn.Module):

    def __init__(self, state_dict_path=''):
        super(CnDSkipClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv1d1 = nn.Conv2d(64, 128, 1, 1, 0)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv6 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv1d2 = nn.Conv2d(128, 128, 1, 2, 0)
        self.conv7 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv1d3 = nn.Conv2d(128, 256, 1, 1, 0)
        self.conv9 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv10 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv1d4 = nn.Conv2d(256, 256, 1, 2, 0)
        self.conv11 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv1d5 = nn.Conv2d(256, 512, 1, 1, 0)
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv14 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv1d6 = nn.Conv2d(512, 512, 1, 2, 0)
        self.x_shape = [0, 512, 8, 8]
        self.linear_dim1 = 512
        self.linear_dim2 = 256
        self.fc1 = nn.Linear(self.x_shape[1]*self.x_shape[2]*self.x_shape[3], self.linear_dim1)
        self.fc2 = nn.Linear(self.linear_dim1, self.linear_dim2)
        self.fc3 = nn.Linear(self.linear_dim2, 2)

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
        # 1
        x = self.conv1(x)   # bx64x64x64
        x = nn.ReLU()(x)
        x = self.conv2(x)   # bx64x64x64
        x = nn.ReLU()(x)
        # 2
        x1 = x
        x = self.conv3(x)   # bx128x64x64
        x = nn.ReLU()(x)
        x = self.conv4(x)   # bx128x64x64
        x = nn.ReLU()(x)
        x += self.conv1d1(x1)
        # 3
        x1 = x
        x = self.conv5(x)   # bx128x64x64
        x = nn.ReLU()(x)
        x = self.conv6(x)   # bx128x64x64
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)   # bx128x32x32
        x += self.conv1d2(x1)
        # 4
        x1 = x
        x = self.conv7(x)   # bx256x32x32
        x = nn.ReLU()(x)
        x = self.conv8(x)   # bx256x32x32
        x = nn.ReLU()(x)
        x += self.conv1d3(x1)
        # 5
        x1 = x
        x = self.conv9(x)   # bx256x32x32
        x = nn.ReLU()(x)
        x = self.conv10(x)   # bx256x32x32
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)   # bx256x16x16
        x += self.conv1d4(x1)
        # 6
        x1 = x
        x = self.conv11(x)   # bx512x16x16
        x = nn.ReLU()(x)
        x = self.conv12(x)   # bx512x16x16
        x = nn.ReLU()(x)
        x += self.conv1d5(x1)
        # 7
        x1 = x
        x = self.conv13(x)   # bx512x16x16
        x = nn.ReLU()(x)
        x = self.conv14(x)   # bx512x16x16
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)   # bx512x8x8
        x += self.conv1d6(x1)
        # Reshape
        x = x.view(-1, self.x_shape[1]*self.x_shape[2]*self.x_shape[3])     # bx512*8*8
        # Fc1
        x = self.fc1(x)     # bx512
        x = nn.ReLU()(x)
        # Fc2
        x = self.fc2(x)     # bx256
        x = nn.ReLU()(x)
        # Fc3
        x = self.fc3(x)     # bx2
        return nn.LogSoftmax(dim=1)(x)
