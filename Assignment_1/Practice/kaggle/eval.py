import os
import sys
import torch

import utils

from model_functions import eval
from parameters import get_params


if __name__ == '__main__':

    # Get all parameters
    args = get_params()
    args.command = 'python ' + ' '.join(sys.argv)

    # CUDA
    utils.check_for_CUDA(args)

    # Load pth
    pth_dir_name = os.path.dirname(args.pth)
    model = torch.load(os.path.join(pth_dir_name, 'model.pth'))
    model.load_state_dict(torch.load(args.pth))
    model = model.to(args.device)

    # Make dataloader with Eval parameters
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
# python eval.py --pth '/home/voletiv/EXPERIMENTS/CnD_experiments/20190208_223808_cnd_kaggle_pt_UNfreeze_ES/model_epoch_0136_batch_00000_of_00141.pth' --data_path '/home/voletiv/Datasets/CatsAndDogs/testset' --out_path '/home/voletiv/EXPERIMENTS/CnD_experiments/20190208_223808_cnd_kaggle_pt_UNfreeze_ES/' --no-cuda
