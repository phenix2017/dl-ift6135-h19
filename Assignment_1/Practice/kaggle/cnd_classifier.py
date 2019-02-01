# https://github.com/pytorch/examples/blob/master/mnist/main.py
import argparse
import csv
import datetime
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import time

import utils

from parameters import get_params


class CnDClassifier(nn.Module):

    def __init__(self, state_dict_path=''):
        super(CnDClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        # self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.x_shape = [0, 16, 16, 16]
        self.linear_dim = 16
        self.fc1 = nn.Linear(self.x_shape[1]*self.x_shape[2]*self.x_shape[3], self.linear_dim)
        self.fc2 = nn.Linear(self.linear_dim, 2)

        if state_dict_path != '':
            # Check
            if os.path.exists(state_dict_path):
                print("Loading", state_dict_path)
            # Load pretrained model
            self.load_state_dict(torch.load(state_dict_path))

    def forward(self, x):
        # bx3x64x64
        x = self.conv1(x)   # bx8x64x64
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)   # bx8x32x32
        x = self.conv2(x)   # bx16x32x32
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)   # bx16x16x16
        # x = self.conv3(x)   # bx128x16x16
        # x = nn.ReLU()(x)
        # x = nn.MaxPool2d(2, 2)(x)   # bx128x8x8
        x = x.view(-1, self.x_shape[1]*self.x_shape[2]*self.x_shape[3])     # bx128*8*8
        x = self.fc1(x)     # bx512
        x = nn.ReLU()(x)
        x = self.fc2(x)     # bx2
        return nn.LogSoftmax(dim=1)(x)


def train(args, model, train_loader, optimizer, epoch, start_time, log_file,
          train_losses, train_accuracy, valid_losses, valid_accuracy):

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
            log = '[{}] : Elapsed [{}]: Epoch: {} [{}/{} ({:.0f}%)]\tTRAIN Loss: {:.6f}\tAccuracy: {:.4f}\n'.format(
                curr_time_str, elapsed, epoch, batch_idx, len(train_loader), 100.*batch_idx/len(train_loader),
                train_losses[-1], train_accuracy[-1])
            print(log)
            log_file.write(log)
            log_file.flush()
            utils.mem_check()
            utils.make_plots(train_losses, train_accuracy, args.log_interval, len(train_loader), args.out_path,
                             valid_losses, valid_accuracy)

        # Save models
        if batch_idx % args.model_save_interval == 0:
            model_name = os.path.join(args.out_path, 'model_epoch_{:04d}_batch_{:05d}_of_{:05d}.pth'.format(epoch, batch_idx, len(train_loader)))
            print("Saving model", model_name)
            torch.save(model.state_dict(), model_name)


def test(args, model, test_loader, epoch, start_time, log_file,
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

    # Get time elapsed
    curr_time = time.time()
    curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
    elapsed = utils.get_time_elapsed_str(curr_time - start_time)

    log = '\n[{}] : Elapsed [{}] : Epoch {}:\tVALIDATION Loss: {:.4f}, Accuracy: {:.4f} ({}/{})\n'.format(
          curr_time_str, elapsed, epoch,
          test_loss, test_accuracy, correct, int(len(test_loader.dataset)*args.valid_split))
    print(log)
    log_file.write(log)
    log_file.flush()

    utils.make_plots(train_losses, train_accuracy, args.log_interval, len(train_loader), args.out_path,
                     valid_losses, valid_accuracy)


def eval(args, model, eval_loader):
    model.eval()
    preds = []

    # Predict
    with torch.no_grad():
        for data, target in tqdm.tqdm(eval_loader):
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            preds += output.argmax(dim=1).tolist() # get the index of the max log-probability

    # Read image names
    ids = [int(os.path.splitext(i)[0]) for i in sorted(os.listdir(os.path.join(args.data_path, 'test')))]

    # Sort ids
    sort_order = np.argsort(ids)
    ids = [ids[i] for i in sort_order]

    # Sort preds and make labels
    labels = eval_loader.dataset.classes
    pred_labels = [labels[preds[i]] for i in sort_order]

    # Write csv
    csv_file_name = os.path.join(os.path.dirname(args.pth), 'submission_' + os.path.basename(os.path.dirname(args.pth)) + '_' + os.path.splitext(os.path.basename(args.pth))[0] + '.csv')
    print("Writing", csv_file_name)
    with open(csv_file_name, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['id', 'label'])
        for i, l in zip(ids, pred_labels):
            csv_writer.writerow([str(i), l])


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


if __name__ == '__main__':

    # Get all parameters
    args = get_params()

    # CUDA
    utils.check_for_CUDA(args)

    # Eval
    if args.eval:
        pth_dir_name = os.path.dirname(args.pth)
        model = torch.load(os.path.join(pth_dir_name, 'model.pth'))
        model.load_state_dict(torch.load(args.pth))
        args.valid_split = 0
        args.centercrop = False
        args.shuffle = False
        args.drop_last = False
        eval_loader = utils.make_dataloader(args)
        # # Visualize
        # inputs, classes = next(iter(eval_loader))
        # import torchvision.utils
        # out = torchvision.utils.make_grid(inputs)
        # imshow(out)
        # # Visualize end
        eval(args, model, eval_loader)
        sys.exit()

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

    # Log file
    log_file_name = os.path.join(args.out_path, 'log.txt')
    log_file = open(log_file_name, "wt")

    try:

        train_losses, train_accuracy = [], []
        valid_losses, valid_accuracy = [], []

        # Test before training
        test(args, model, valid_loader, 0, start_time, log_file,
            train_losses, train_accuracy, valid_losses, valid_accuracy)

        for epoch in range(1, args.epochs + 1):

            # Train
            train(args, model, train_loader, optimizer, epoch, start_time, log_file,
                  train_losses, train_accuracy, valid_losses, valid_accuracy)

            # Validate
            test(args, model, valid_loader, epoch, start_time, log_file,
                 train_losses, train_accuracy, valid_losses, valid_accuracy)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt!\n")

    # SAVE FINAL MODEL
    print("Saving final model:", os.path.join(args.out_path, "final.pth"))
    torch.save(model.state_dict(), os.path.join(args.out_path, "final.pth"))


# TRAIN
# python3 cnd_classifier.py --data_path '/home/user1/Datasets/CatsAndDogs/trainset' --out_path '/home/user1/CnD_experiments/cnd2'
# python cnd_classifier.py --data_path '/home/voletivi/scratch/catsndogs/data/kaggle/trainset' --out_path '/home/voletivi/scratch/catsndogs/experiments/cnd_kaggle_1'

# EVAL
# python3 cnd_classifier.py --eval --pth '/home/user1/CnD_experiments/20190201_014519_cnd_kaggle_smllerFC_1/model_epoch_0016_batch_00100_of_00141.pth' --data_path '/home/user1/Datasets/CatsAndDogs/testset' --out_path '/home/user1/CnD_experiments/20190201_014519_cnd_kaggle_smllerFC_1'
