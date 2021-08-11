import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch_lr_finder import LRFinder

import arg
import lib
from main import get_model
torch.backends.cudnn.benchmark = True
args = vars(arg.get_params())
model = get_model(args)
DATA_DIR = "/mnt/ramdisk/408/ap"
dataloaders = lib.get_dataloaders(args, 1, 10, 2, DATA_DIR)
logger = logging.getLogger()

classes_weights=lib.weights_calc(dataloaders["train"], logger)
pos_weight = classes_weights[0]/classes_weights[1]
criterion = nn.BCEWithLogitsLoss(pos_weight.to("cuda"))

#criterion = nn.CrossEntropyLoss(weight=lib.weights_calc(dataloaders["train"], logger).to("cuda"))
optimizer = optim.SGD(model.parameters(), lr=1e-7, weight_decay=1e-4)
lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(
    dataloaders["train"],
    dataloaders["val"],
    start_lr=1e-7,
    end_lr=1,
    num_iter=100,
    step_mode="exp",
)
torch.save(lr_finder, "408.lr")
