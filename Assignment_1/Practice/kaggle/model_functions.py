import csv
import datetime
import numpy as np
import os
import torch
import torch.nn as nn
import tqdm
import time

import utils


def train(args, model, train_loader, optimizer, epoch, start_time, log_file,
          train_epochs, train_losses, train_accuracy, valid_epochs, valid_losses, valid_accuracy):

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
            train_epochs.append(epoch + batch_idx/len(train_loader))
            train_losses.append(loss.item())
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            train_accuracy.append(pred.eq(target.view_as(pred)).sum().item()/len(pred))

            # Get time elapsed
            curr_time = time.time()
            curr_time_str, elapsed_str = utils.get_time_str(start_time, curr_time)

            # Log
            log = '[{}] : Elapsed [{}]: Epoch: {} [{}/{} ({:.0f}%)]\tTRAIN Loss: {:.6f}\tAccuracy: {:.4f}\n'.format(
                curr_time_str, elapsed_str, epoch, batch_idx, len(train_loader), 100.*batch_idx/len(train_loader),
                train_losses[-1], train_accuracy[-1])
            print(log)
            log_file.write(log)
            log_file.flush()
            utils.mem_check()
            utils.make_plots(args.out_path, train_epochs, train_losses, train_accuracy, valid_epochs, valid_losses, valid_accuracy)

        # Save models
        if batch_idx % args.model_save_interval == 0:
            model_name = os.path.join(args.out_path, 'model_epoch_{:04d}_batch_{:05d}_of_{:05d}.pth'.format(epoch, batch_idx, len(train_loader)))
            print("Saving model", model_name)
            torch.save(model.state_dict(), model_name)


def test(args, model, test_loader, epoch, start_time, log_file,
         train_epochs, train_losses, train_accuracy, valid_epochs, valid_losses, valid_accuracy):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        counter = 0
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            test_loss += nn.NLLLoss(reduction='sum')(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            counter += len(pred)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= counter
    test_accuracy = correct/counter

    valid_epochs.append(epoch)
    valid_losses.append(test_loss)
    valid_accuracy.append(test_accuracy)

    # Get time elapsed
    curr_time = time.time()
    curr_time_str, elapsed_str = utils.get_time_str(start_time, curr_time)

    log = '\n[{}] : Elapsed [{}] : Epoch {}:\tVALIDATION Loss: {:.4f}, Accuracy: {:.4f} ({}/{})\n'.format(
          curr_time_str, elapsed_str, epoch,
          test_loss, test_accuracy, correct, counter)
    print(log)
    log_file.write(log)
    log_file.flush()

    utils.make_plots(args.out_path, train_epochs, train_losses, train_accuracy, valid_epochs, valid_losses, valid_accuracy)


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
    labels = ['Cat', 'Dog']
    pred_labels = [labels[preds[i]] for i in sort_order]

    # Write csv
    csv_file_name = os.path.join(os.path.dirname(args.pth), 'submission_' + os.path.basename(os.path.dirname(args.pth)) + '_' + os.path.splitext(os.path.basename(args.pth))[0] + '.csv')
    print("Writing", csv_file_name)
    with open(csv_file_name, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['id', 'label'])
        for i, l in zip(ids, pred_labels):
            csv_writer.writerow([str(i), l])
