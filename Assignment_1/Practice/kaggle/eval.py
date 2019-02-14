import os
import sys
import torch

import utils

from model_functions import eval
from parameters import get_params


if __name__ == '__main__':

    # Get all parameters
    args = get_params(eval_mode=True)
    args.command = 'python ' + ' '.join(sys.argv)
    assert args.pth
    assert os.path.exists(args.pth)
    assert args.data_path
    assert os.path.exists(args.data_path)
    assert args.out_path
    assert os.path.exists(args.out_path)

    # CUDA
    utils.check_for_CUDA(args)

    # Load pth
    pth_dir_name = os.path.dirname(args.pth)
    print("Loading model", os.path.join(pth_dir_name, 'model.pth'))
    model = torch.load(os.path.join(pth_dir_name, 'model.pth'))
    print("Loading model state dict", args.pth)
    model.load_state_dict(torch.load(args.pth))
    model = model.to(args.device)

    # Make dataloader with Eval parameters
    print("Making dataloader")
    args.valid_split = 0
    args.centercrop = False
    args.shuffle = False
    args.drop_last = False
    eval_loader = utils.make_dataloader(args)

    # # Visualize
    # inputs, classes = next(iter(eval_loader))
    # import torchvision.utils
    # out = torchvision.utils.make_grid(inputs)
    # utils.imshow(out)
    # # Visualize end

    # Evaluate
    eval(args, model, eval_loader)

# EVAL
# python eval.py --eval --pth '/home/voletiv/EXPERIMENTS/CnD_experiments/20190208_223808_cnd_kaggle_pt_UNfreeze_ES/model_epoch_0136_batch_00000_of_00141.pth' --data_path '/home/voletiv/Datasets/CatsAndDogs/testset' --out_path '/home/voletiv/EXPERIMENTS/CnD_experiments/20190208_223808_cnd_kaggle_pt_UNfreeze_ES/' --no-cuda
