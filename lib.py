import logging
import os

import torch

import torchvision
from sklearn.metrics import confusion_matrix
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms


def output_calc(output, label, alpha=0.0):
    outputn = output[:, 1] - output[:, 0]
    outputn = outputn / max([max(outputn), -min(outputn)])
    outputn = outputn / 2 + 0.5
    pred = torch.round(outputn - alpha)
    acc = torch.sum(pred == label.data).item() / len(label)
    print(acc)
    print(confusion_matrix(pred.cpu(), label.cpu()))
    # TP TN
    return outputn


def cal_i(output, preds, label):
    imax = 0
    max_by_loss = 0.0
    max_by_pred = 0.0
    for i in range(100, 1000):
        i = i / 100
        final_loss = (output["fp"] + output["hp"] + output["ap"] * i) / (2 + i)
        _, preds_by_loss = torch.max(final_loss, 1)
        acc_loss = torch.sum(preds_by_loss == label.data).item() / len(label)
        if acc_loss > max_by_loss:
            imax = i
            max_by_loss = acc_loss
    print("by loss i:{} acc:{}".format(imax, max_by_loss))
    for i in range(100, 1000):
        i = i / 100
        pred_by_preds = (preds["fp"] + preds["hp"] + preds["ap"] * i) / (2 + i)
        pred_by_preds = torch.round(pred_by_preds)
        acc_preds = torch.sum(pred_by_preds == label.data).item() / len(label)
        if acc_preds > max_by_pred:
            imax = i
            max_by_pred = acc_preds
    print("by pred i:{} acc:{}".format(imax, max_by_pred))


def set_logger(trial_time, trial_name):
    output = logging.getLogger("log")
    hander = logging.FileHandler(
        "train_log/" + trial_time + "(" + trial_name + ").log",
        encoding="UTF-8",
        mode="w+",
    )
    formatter = logging.Formatter(
        "[%(levelname)1.1s %(asctime)s %(module)s.%(funcName)s] %(message)s ",
        datefmt="%Y%m%d %H:%M:%S",
    )
    hander.setFormatter(formatter)
    output.addHandler(hander)
    output.setLevel("INFO")
    return output


def get_foldlist(fold, fold_num):
    """
    get fold list
    fold : fold number, starting from 1
    global variable : fold_num
    """
    output = dict()
    fold_range = []
    _ = [[fold_range.append(i) for i in range(1, fold_num + 1)]
         for j in range(2)]
    output["train"] = fold_range[fold - 1: fold + fold_num - 2]
    # output["test"] = fold_range[fold + fold_num - 3: fold + fold_num - 2]
    output["val"] = fold_range[fold + fold_num - 2: fold + fold_num - 1]
    return output


def get_dataloaders(args, fold, fold_num, plane, data_dir):
    fold_list = get_foldlist(fold, fold_num)
    data_transforms = get_datatransformer(args, plane)
    dataset = dict()
    for phase in fold_list.keys():
        temp = []
        for fold in fold_list[phase]:
            temp.append(
                datasets.ImageFolder(
                    os.path.join(data_dir, str(plane),str(fold)), data_transforms[phase]
                )
            )
            dataset[phase] = ConcatDataset(temp)
    dataloaders = {
        x: torch.utils.data.DataLoader(
            dataset[x],
            batch_size=args["batch_size"],
            shuffle=(x != "val"),
            num_workers=args["workers"],
            pin_memory=True,
        )
        for x in fold_list.keys()
    }
    return dataloaders


def get_dataloaders_all_shuffle(args, fold, fold_num, plane, data_dir):
    data_transforms = get_datatransformer(args, plane)
    train_tuple = []
    test_tuple = []
    new_datasets = dict()
    for fold in range(1, fold_num + 1):
        new_datasets = dict()
        temp_dataset = datasets.ImageFolder(
            os.path.join(data_dir, str(plane),str(fold)), data_transforms["train"]
        )
        trainsetA_size = round(len(temp_dataset) * 0.1*(fold-1))
        testset_size = round(len(temp_dataset) * 0.1)
        trainsetB_size = round(len(temp_dataset)-trainsetA_size-testset_size)
        [new_datasets["trainA"], new_datasets["val"], new_datasets["trainB"]] = torch.utils.data.random_split(
            temp_dataset, [trainsetA_size, testset_size,
                           trainsetB_size], generator=torch.Generator().manual_seed(1024)
        )
        train_tuple.append(new_datasets["trainA"])
        train_tuple.append(new_datasets["trainB"])
        test_tuple.append(new_datasets["val"])
    new_datasets["train"] = ConcatDataset(train_tuple)
    new_datasets["val"] = ConcatDataset(test_tuple)
    for i in range(fold_num):
        new_datasets["val"].datasets[i].dataset.transforms = data_transforms["val"]
    dataloaders = {
        x: torch.utils.data.DataLoader(
            new_datasets[x],
            batch_size=args["batch_size"],
            shuffle=(x == "train"),
            num_workers=args["workers"],
            pin_memory=True,
        )
        for x in ["train", "val"]
    }
    return dataloaders


def __FourCrop(img, quadrant):
    img_width, img_height = img.size
    return img.crop(
        (
            img_width / 2 * quadrant[0],
            img_height / 2 * quadrant[1],
            img_width * (quadrant[0] + 1),
            img_height * (quadrant[1] + 1),
        )
        * (quadrant[0] + 1)
    )


def __TwoCrop(img, plane):
    img_width, img_height = img.size
    if not plane == 2:
        # img = img.crop((0, img_height / 2 * plane, img_width,img_height / 2 * (plane + 1)))  # up down
        img = img.crop((img_width / 2 * plane, 0, img_width /2 * (plane + 1), img_height)) #LR
    return img.resize((224, 224))


def get_datatransformer(args, plane):
    output = {
        "train": transforms.Compose(
            [
                # transforms.Lambda(lambda img: __TwoCrop(img, plane)),
                transforms.Resize(224),
                transforms.RandomAffine(
                    degrees=args["degrees"], translate=[args["xtras"], 0]
                ),
                # transforms.ColorJitter([args['CJ_brightness'], 1], [1, 1], [
                #                       1, 1], [-args['CJ_hue'], args['CJ_hue']]),
                transforms.ToTensor(), ]
        ),
        "val": transforms.Compose(
            [
                # transforms.Lambda(lambda img: __TwoCrop(img, plane)),
                transforms.Resize(224),
                # transforms.Lambda(lambda img:__FourCrop(img, 1)),
                transforms.ToTensor(), ]
        ),
        # "test": transforms.Compose(
        #     [
        #         transforms.Lambda(lambda img: __TwoCrop(img, plane)),
        #         # transforms.Resize(224),
        #         # transforms.Lambda(lambda img:__FourCrop(img, 1)),
        #         transforms.ToTensor(), ]
        # ),
    }
    if args["normalization"]:
        for phase in ["train", "val"]:
            output[phase] = transforms.Compose(
                [output[phase], transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
    return output


def weights_calc(dataloaders, logger):
    targets = []
    sample_dataset = None
    if type(dataloaders.dataset.datasets[0]) == torchvision.datasets.folder.ImageFolder:
        for i in range(len(dataloaders.dataset.datasets)):
            sample_dataset = dataloaders.dataset.datasets[i]
            targets += sample_dataset.targets
        classes = sample_dataset.classes
    else:  # All shuffle case
        for i in range(len(dataloaders.dataset.datasets)):
            sample_dataset = dataloaders.dataset.datasets[i]
            targets += sample_dataset.dataset.targets
        classes = sample_dataset.dataset.classes
    targets = torch.IntTensor(targets)
    classes_size = torch.sum(targets == 0)
    logger.info("{:5}:{}".format(classes[0], classes_size))
    for i in range(1, len(classes)):
        classes_size = torch.stack([classes_size, torch.sum(targets == i)])
        logger.info("{:5}:{}".format(classes[i], classes_size[i]))
    classes_weights = 2-(classes_size / torch.sum(classes_size) * len(classes))
    logger.info("weights:{}".format(classes_weights.tolist()))
    return classes_weights
