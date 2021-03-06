# https://github.com/pytorch/examples/blob/master/mnist/main.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import time

import utils

from models import *
from model_functions import *
from parameters import get_params


if __name__ == '__main__':

    # Get all parameters
    args = get_params()
    args.command = 'python ' + ' '.join(sys.argv)

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
    if args.pth != '':
        pth_dir_name = os.path.dirname(args.pth)
        full_model_pth = os.path.join(pth_dir_name, 'model.pth')
        if os.path.exists(full_model_pth):
            print("Loading", full_model_pth)
            model = torch.load(full_model_pth)
            print("Loading pretrained state_dict", args.pth)
            model.load_state_dict(torch.load(args.pth))
            model = model.to(args.device)
        else:
            if args.model == 'baseline':
                model = CnDClassifier()
            elif args.model == 'big':
                model = CnDBigClassifier()
            elif args.model == 'TinyImageNet':
                model = TinyImageNetClassifier()
            elif args.model == 'transfer':
                pth_dir_name = os.path.dirname(args.transfer)
                full_model_pth = os.path.join(pth_dir_name, 'model.pth')
                model = TransferModel(full_model_pth, args.transfer, args.freeze)
            elif args.model == 'skip':
                model = CnDSkipClassifier().to(args.device)
            elif args.model == 'bn_skip':
                model = CnDBNSkipClassifier(args.bn, args.skip, args.device)
            print("Loading pretrained state_dict", args.pth)
            model.load_state_dict(torch.load(args.pth))
            model = model.to(args.device)
    else:
        if args.model == 'baseline':
            model = CnDClassifier().to(args.device)
        elif args.model == 'big':
            model = CnDBigClassifier().to(args.device)
        elif args.model == 'TinyImageNet':
            model = TinyImageNetClassifier().to(args.device)
        elif args.model == 'transfer':
            pth_dir_name = os.path.dirname(args.transfer)
            full_model_pth = os.path.join(pth_dir_name, 'model.pth')
            model = TransferModel(full_model_pth, args.transfer, args.freeze).to(args.device)
        elif args.model == 'skip':
            model = CnDSkipClassifier().to(args.device)
        elif args.model == 'bn_skip':
            model = CnDBNSkipClassifier(args.bn, args.skip, args.device).to(args.device)

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

        train_epochs, train_losses, train_accuracy = [], [], []
        valid_epochs, valid_losses, valid_accuracy = [], [], []
        lr_change = []

        # Test before training
        test(args, model, valid_loader, 0, start_time, log_file,
             train_epochs, train_losses, train_accuracy, valid_epochs, valid_losses, valid_accuracy, lr_change)

        # Early stopping
        if args.early_stopping:
            early_stopping_counter = 0
            best_val_loss = valid_losses[-1]

        # For early stop and resume
        epoch_start = 0
        epoch = epoch_start
        while epoch < args.epochs:

            for epoch in range(epoch_start, args.epochs + 1):

                # Train
                train(args, model, train_loader, optimizer, epoch, start_time, log_file,
                      train_epochs, train_losses, train_accuracy, valid_epochs, valid_losses, valid_accuracy, lr_change)

                # Validate
                test(args, model, valid_loader, epoch, start_time, log_file,
                     train_epochs, train_losses, train_accuracy, valid_epochs, valid_losses, valid_accuracy, lr_change)

                # Exponential decay
                if args.exp_decay:
                    if epoch % args.exp_decay_epochs == 0:
                        args.lr *= args.exp_decay_rate
                        lr_change.append(epoch)

                # Early stopping
                if args.early_stopping:
                    if valid_losses[-1] > best_val_loss:
                        early_stopping_counter += 1
                        if early_stopping_counter == args.patience:
                            print("Early stopping! Resuming with half LR...")
                            lr_change.append(epoch)
                            best_val_loss = np.inf
                            early_stopping_counter = 0
                            epoch_start = epoch + 1
                            args.lr /= 2
                            print("New LR:", args.lr)
                            optimizer = optim.SGD(model.parameters(), lr=args.lr)
                            break
                    else:
                        best_val_loss = valid_losses[-1]
                        early_stopping_counter = 0

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt!\n")

    # SAVE FINAL MODEL
    print("Saving final model:", os.path.join(args.out_path, "final.pth"))
    torch.save(model.state_dict(), os.path.join(args.out_path, "final.pth"))


# TRAIN
# # Mila
# python3 train.py --data_path '/home/user1/Datasets/CatsAndDogs/trainset' --out_path '/home/user1/CnD_experiments/cnd2'
# # CC
# python train.py --data_path '/home/voletivi/scratch/catsndogs/data/kaggle/trainset' --out_path '/home/voletivi/scratch/catsndogs/experiments/cnd_kaggle_BIG' --model 'bn_skip' --no_bn --no_skip --early_stopping
# # Dell
# python train.py --data_path '/home/voletiv/Datasets/CatsAndDogs/trainset' --out_path '/home/voletiv/EXPERIMENTS/CnD_experiments/cnd_kaggle_C16C32C64Fc16_XInit_DataAugLesser_LR0.5_cont1' --pth ''

# Tiny ImageNet
# python train.py --data_path '/home/voletivi/scratch/catsndogs/data/TinyImageNet/tiny-imagenet-200/train' --val_data_path '/home/voletivi/scratch/catsndogs/data/TinyImageNet/tiny-imagenet-200/val' --out_path '/home/voletivi/scratch/catsndogs/experiments/pretrain_TinyImageNet' --model 'big'
