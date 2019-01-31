# https://github.com/pytorch/examples/blob/master/mnist/main.py
import argparse
import datetime
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets as dset

import utils


class PerfectClassifier(nn.Module):

    def __init__(self, state_dict_path=None):
        super(PerfectClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.fc1 = nn.Linear(128*8*8, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        # bx3x64x64
        x = self.conv1(x)   # bx32x64x64
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)   # bx32x32x32
        x = self.conv2(x)   # bx64x32x32
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)   # bx64x16x16
        x = self.conv3(x)   # bx128x16x16
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)   # bx128x8x8
        x = x.view(-1, 128*8*8)     # bx128*8*8
        x = self.fc1(x)     # bx512
        x = nn.ReLU()(x)
        x = self.fc2(x)     # bx2
        return nn.LogSoftmax(dim=1)(x)


def train(args, model, train_loader, optimizer, epoch,
          accuracy, accuracy_in_interval, losses, losses_in_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Get data
        data, target = data.to(args.device), target.to(args.device)
        # Get model output
        optimizer.zero_grad()
        output = model(data)
        # Check accuracy
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        accuracy_in_interval.append(pred.eq(target.view_as(pred)).sum().item()/args.batch_size)
        # Calc loss
        loss = nn.NLLLoss()(output, target)
        losses_in_interval.append(loss.item())
        # Backprop
        loss.backward()
        optimizer.step()
        # Log, Plot
        if batch_idx % args.log_interval == 0:
            accuracy.append(np.mean(accuracy_in_interval))
            accuracy_in_interval = []
            losses.append(np.mean(losses_in_interval))
            losses_in_interval = []
            print('Train Epoch: {} [{}/{} ({}/{}) ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.4f}'.format(
                epoch, batch_idx, len(train_loader), batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), losses[-1], accuracy[-1]))
            utils.make_plots(losses, accuracy, args.log_interval, len(train_loader), args.out_path, init_epoch=0)
        # Save models
        if batch_idx % args.model_save_interval == 0:
            model_name = os.path.join(args.out_path, 'model_epoch_{:04d}_batch_{:05d}_of_{:05d}.pth'.format(epoch, batch_idx, len(train_loader)))
            print("Saving model", model_name)
            torch.save(model.state_dict(), model_name)



def test(args, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            test_loss += nn.NLLLoss()(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':

    # ARGS
    parser = argparse.ArgumentParser(description='PyTorch Cats & Dogs')
    parser.add_argument('--data_path', type=str,
                        help='Path to data : Images Folder')
    parser.add_argument('--out_path', type=str,
                        help='Path to data : Directory out: a/b/exp1 (-> a/b/<TIME>_exp1)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=29, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model-save-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before saving model')

    args = parser.parse_args()

    args.out_path = os.path.join(os.path.dirname(args.out_path),
                                 '{0:%Y%m%d_%H%M%S}_{1}'.format(datetime.datetime.now(), os.path.basename(args.out_path)))

    args.use_cuda = not args.no_cuda and torch.cuda.is_available()

    args.device = torch.device("cuda" if args.use_cuda else "cpu")

    args.kwargs = {'num_workers': 1, 'pin_memory': True} if args.use_cuda else {}

    torch.manual_seed(args.seed)

    # IMAGES DATALOADER

    transform = utils.make_transform()

    assert os.path.exists(args.data_path), "data_path does not exist! Given: " + args.data_path
    dataset = dset.ImageFolder(root=args.data_path, transform=transform)
    args.num_of_classes = sum([1 if os.path.isdir(os.path.join(args.data_path, i)) else 0 for i in os.listdir(args.data_path)])

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **args.kwargs)

    # OUT PATH
    if not os.path.exists(args.out_path):
        print("Making", args.out_path)
        os.makedirs(args.out_path)

    # Save all args
    utils.write_config_to_file(args, args.out_path)

    # MODEL

    model = PerfectClassifier().to(args.device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # TRAIN

    # Save full model
    print("Saving full model:", os.path.join(args.out_path, "model.pth"))
    torch.save(model, os.path.join(args.out_path, "model.pth"))

    print("Starting training...")

    try:
        accuracy, accuracy_in_interval, losses, losses_in_interval = [], [], [], []
        for epoch in range(1, args.epochs + 1):
            train(args, model, train_loader, optimizer, epoch,
                  accuracy, accuracy_in_interval, losses, losses_in_interval)
            # test(args, model, args.device, test_loader)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt!\n")

    # SAVE FINAL MODEL
    print("Saving final model")
    torch.save(model.state_dict(), os.path.join(args.out_path, "final.pth"))


# python3 perfect_classifier.py --data_path '/home/user1/Datasets/CatsAndDogs/trainset' --out_path '/home/user1/test/cnd1'
