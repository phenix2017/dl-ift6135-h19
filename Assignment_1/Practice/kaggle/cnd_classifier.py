# https://github.com/pytorch/examples/blob/master/mnist/main.py
import argparse
import datetime
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time

import utils

from parameters import get_params


class CnDClassifier(nn.Module):

    def __init__(self, state_dict_path=None):
        super(CnDClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.x_shape = [0, 128, 8, 8]
        self.linear_dim = 512
        self.fc1 = nn.Linear(self.x_shape[1]*self.x_shape[2]*self.x_shape[3], self.linear_dim)
        self.fc2 = nn.Linear(self.linear_dim, 2)

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
        x = x.view(-1, self.x_shape[1]*self.x_shape[2]*self.x_shape[3])     # bx128*8*8
        x = self.fc1(x)     # bx512
        x = nn.ReLU()(x)
        x = self.fc2(x)     # bx2
        return nn.LogSoftmax(dim=1)(x)


def train(args, model, train_loader, optimizer, epoch, start_time,
          train_accuracy, train_losses, valid_accuracy, valid_losses):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Get data
        data, target = data.to(args.device), target.to(args.device)
        # Get model output
        optimizer.zero_grad()
        output = model(data)
        # Calc loss
        loss = nn.NLLLoss()(output, target)
        # Backprop
        loss.backward()
        optimizer.step()
        # Log, Plot
        if batch_idx % args.log_interval == 0:
            # Check loss, accuracy
            train_losses.append(loss.item())
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            train_accuracy.append(pred.eq(target.view_as(pred)).sum().item()/len(pred))
            # Get time elapsed
            curr_time = time.time()
            curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
            elapsed = utils.get_time_elapsed_str(curr_time - start_time)
            # Log
            print('[{}] : Elapsed [{}]: Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.4f}'.format(
                curr_time_str, elapsed, epoch, batch_idx, len(train_loader), 100.*batch_idx/len(train_loader),
                train_losses[-1], train_accuracy[-1]))
            utils.mem_check()
            utils.make_plots(train_losses, train_accuracy, args.log_interval, len(train_loader), args.out_path,
                             valid_losses, valid_accuracy)
        # Save models
        if batch_idx % args.model_save_interval == 0:
            model_name = os.path.join(args.out_path, 'model_epoch_{:04d}_batch_{:05d}_of_{:05d}.pth'.format(epoch, batch_idx, len(train_loader)))
            print("Saving model", model_name)
            torch.save(model.state_dict(), model_name)


def test(args, model, test_loader,
         train_losses, train_accuracy, valid_losses, valid_accuracy):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            test_loss += nn.NLLLoss(reduction='sum')(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)*args.valid_split
    test_accuracy = correct/int(len(test_loader.dataset)*args.valid_split)

    valid_losses.append(test_loss)
    valid_accuracy.append(test_accuracy)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.4f} ({}/{})\n'.format(
          test_loss, test_accuracy, correct, int(len(test_loader.dataset)*args.valid_split)))

    utils.make_plots(train_losses, train_accuracy, args.log_interval, len(train_loader), args.out_path,
                     valid_losses, valid_accuracy)

if __name__ == '__main__':

    # Get all parameters
    args = get_params()

    # CUDA
    utils.check_for_CUDA(args)

    # Seed
    torch.manual_seed(args.seed)

    # IMAGES DATALOADER
    train_loader, valid_loader = utils.make_dataloader(args)

    print(args)

    # OUT PATH
    if not os.path.exists(args.out_path):
        print("Making", args.out_path)
        os.makedirs(args.out_path)

    # Copy all scripts
    utils.copy_scripts(args.out_path)

    # Save all args
    utils.write_config_to_file(args, args.out_path)

    # MODEL

    torch.manual_seed(args.seed)
    model = CnDClassifier().to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # TRAIN

    # Save full model
    print("Saving full model:", os.path.join(args.out_path, "model.pth"))
    torch.save(model, os.path.join(args.out_path, "model.pth"))

    print("Starting training...")
    start_time = time.time()

    try:
        train_losses, train_accuracy = [], []
        valid_losses, valid_accuracy = [], []
        test(args, model, valid_loader, train_losses, train_accuracy, valid_losses, valid_accuracy)
        for epoch in range(1, args.epochs + 1):
            train(args, model, train_loader, optimizer, epoch, start_time,
                  train_losses, train_accuracy, valid_losses, valid_accuracy)
            test(args, model, valid_loader, train_losses, train_accuracy, valid_losses, valid_accuracy)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt!\n")

    # SAVE FINAL MODEL
    print("Saving final model:", os.path.join(args.out_path, "final.pth"))
    torch.save(model.state_dict(), os.path.join(args.out_path, "final.pth"))


# python3 cnd_classifier.py --data_path '/home/user1/Datasets/CatsAndDogs/trainset' --out_path '/home/user1/test/cnd2'
# python cnd_classifier.py --data_path '/home/voletivi/scratch/catsndogs/data/kaggle/trainset' --out_path '/home/voletivi/scratch/catsndogs/experiments/cnd_kaggle_1'
