import argparse
import logging
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import time
import train
import nni
import nets
from main import get_fold, xlfold, get_datatransformer, get_params, initialize,get_model
from nni.utils import merge_parameter
from torch.utils.data import ConcatDataset

TRIAL_NAME = ""
TRIAL_TIME = ""
TL_BASE = ""
DATA_DIR = ""
FOLD_NUM = ""
initialize(False)

args = vars(get_params())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transforms = get_datatransformer(args)
fold_list = get_fold(0)
image_datasets = xlfold(data_transforms, fold_list)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args['batch_size'], shuffle=True,
                                              num_workers=args['workers'], pin_memory=True)
               for x in ['train', 'val']}
model = get_model(args)
pArgs = train.primaryArgs(
    TRIAL_TIME + '_' + TRIAL_NAME, 'logger', args, device, "" , args['epochs'], FOLD_NUM)
train_runners = train.trainRunner(dataloaders, model, pArgs, 0)
train_runners.load_state("state_dict/"+TL_BASE)
train_runners.test(TL_BASE,print = True)
