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
            model = CnDBigClassifier()
            print("Loading pretrained state_dict", args.pth)
            model.load_state_dict(torch.load(args.pth))
            model = model.to(args.device)
    else:
        model = CnDBigClassifier().to(args.device)

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
                      train_losses, train_accuracy, valid_losses, valid_accuracy)

                # Validate
                test(args, model, valid_loader, epoch, start_time, log_file,
                     train_losses, train_accuracy, valid_losses, valid_accuracy)

                # Early stopping
                if args.early_stopping:
                    if valid_losses[-1] > best_val_loss:
                        early_stopping_counter += 1
                        if early_stopping_counter == args.patience:
                            early_stopping_counter = 0
                            epoch_start = epoch + 1
                            args.lr /= 2
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
# python3 cnd_classifier.py --data_path '/home/user1/Datasets/CatsAndDogs/trainset' --out_path '/home/user1/CnD_experiments/cnd2'
# # CC
# python cnd_classifier.py --data_path '/home/voletivi/scratch/catsndogs/data/kaggle/trainset' --out_path '/home/voletivi/scratch/catsndogs/experiments/cnd_kaggle_1'
# # Dell
# python cnd_classifier.py --data_path '/home/voletiv/Datasets/CatsAndDogs/trainset' --out_path '/home/voletiv/EXPERIMENTS/CnD_experiments/cnd_kaggle_C16C32C64Fc16_XInit_DataAugLesser_LR0.5_cont1' --pth ''

# EVAL
# python3 cnd_classifier.py --eval --pth '/home/user1/CnD_experiments/20190201_014519_cnd_kaggle_smllerFC_1/model_epoch_0016_batch_00100_of_00141.pth' --data_path '/home/user1/Datasets/CatsAndDogs/testset' --out_path '/home/user1/CnD_experiments/20190201_014519_cnd_kaggle_smllerFC_1'
