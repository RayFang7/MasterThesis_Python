# coding=utf-8
import os
import time

import numpy as np
import progressbar
import torch
import torch.nn as nn
import torch.optim as optim
import json
from sklearn import metrics
from torch.cuda.amp import autocast
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_pil_image

import main
from lib import weights_calc


class TrainRunner:
    def __del__(self):
        torch.torch.cuda.empty_cache()

    def __init__(self, args, logger, dataloaders, fold):
        self.args = args
        self.trial_name = args["TRIAL_TIME"] + "#" + args["TRIAL_NAME"]

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info("using device:: %s", str(self.device))
        model = main.get_model(args)

        self.logger = logger
        self.writer = {
            phase: SummaryWriter("runs/"+self.trial_name+"/" + phase) for phase in ["train", "val"]
        }

        self.writer["train"].add_text(
            "arguments", json.dumps(args, sort_keys=False, indent=4))

        self.result = {measure: np.array([0.0])
                       for measure in ["f1s", "tacc", "vacc"]}

        self.dataset_sizes = {
            phase: len(dataloaders[phase].dataset) for phase in dataloaders.keys()
        }

        self.dataloaders = dataloaders
        self.logger.info("trainset sizes:")
        classes_weights = weights_calc(dataloaders["train"], self.logger)
        self.val_label = torch.empty(0)
        self.criterion = nn.CrossEntropyLoss(classes_weights.to(self.device))
        # pos_weight = classes_weights[0]/classes_weights[1]
        # self.criterion = nn.BCEWithLogitsLoss(pos_weight.to(self.device))

        self.fold = fold
        self.epoch = 0
        self.zero = {"loss": 0.0, "acc": 0.0, "f1s": 0.0}
        self.shutdown = 0
        self.result = {"val": self.zero.copy(), "train": self.zero.copy()}
        self.confusion = [dict()]

        model = model.to(self.device, non_blocking=True)
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.args["lr"],
            weight_decay=self.args["weight_decay"],
            momentum=self.args["momentum"],
        )
        self.model = model
        self.optimizer = optimizer
        # optimizer.param_groups['lr']
        self.scaler = torch.cuda.amp.GradScaler()
        self.iter_per_epoch = round(
            self.dataset_sizes["train"] / self.args["batch_size"]
        )
        self.scheduler = lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=args["base_lr"],
            max_lr=args["max_lr"],
            mode="triangular2",
            verbose=False,
            step_size_up=self.iter_per_epoch * args["step_size"],
        )
        self.logger.info(
            "fold&epoch| lr_rate | t_loss | t_acc  | t_f1s  | v_loss | v_acc  | v_f1s  || time")

    def load_state(self, dict_path):
        self.model.load_state_dict(torch.load(dict_path))

    def save_checkpoint(self, PATH):
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "result": self.result,
            },
            PATH,
        )

    def load_checkpoint(self, PATH):
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.result = checkpoint["result"]

    def train(self):

        sigmoid = nn.Sigmoid()
        device = self.device
        criterion = self.criterion
        since = time.time()
        cat = torch.cat
        flood_level = self.args["flood_level"]
        batch_size = self.args["batch_size"]

        sum_label = None
        time_elapsed = None
        pbar = None

        # dataloaders = self.dataloaders
        for phase in self.dataloaders.keys():
            inepoch_loss = 0.0
            inepoch_acc = 0.0
            running_loss = 0.0
            running_corrects = 0
            batch_number = 0
            sum_label = torch.empty(0)
            sum_output = torch.empty(0)
            epoch_result = self.zero.copy()

            if not self.args["nni"]:
                widgets = [
                    "  fold: {}/{} ".format(self.fold,
                                            self.args["fold_num"]),
                    progressbar.Variable("epoch", width=len(
                        str(self.args["epochs_num"]))),
                    "/{} | {:5} ".format(self.args["epochs_num"], phase),
                    progressbar.Variable("loss"),
                    " ",
                    progressbar.Variable("acc"),
                    " ",
                    progressbar.Percentage(),
                    " ",
                    progressbar.Bar(),
                    " ",
                    progressbar.ETA(),
                ]
                pbar = progressbar.ProgressBar(
                    maxval=round(
                        self.dataset_sizes[phase] / self.args["batch_size"] + 1
                    ),
                    widgets=widgets,
                ).start()
            for inputs, labels in self.dataloaders[phase]:
                if self.shutdown:
                    return 1
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()

                batch_number += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    with autocast():
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # outputs = torch.squeeze(outputs, dim=1)
                        # loss = criterion(outputs, labels.type_as(outputs))
                        # preds = torch.round(sigmoid(outputs))

                        sum_output = cat((sum_output, outputs.cpu()), 0)
                        sum_label = cat((sum_label, labels.cpu()), 0)
                    if phase == "train":
                        loss = (loss - flood_level).abs() + flood_level
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                inepoch_loss = running_loss / (batch_number * batch_size)
                inepoch_acc = running_corrects / (batch_number * batch_size)

                if not self.args["nni"]:
                    pbar.update(  # type:ignore
                        batch_number,
                        epoch=self.epoch + 1,
                        loss=inepoch_loss,
                        acc=inepoch_acc,
                    )
                if phase == "train":
                    self.tb_write(
                        "Fold " + str(self.fold)+"/Plane" +
                        str(self.args["plane"]),
                        phase,
                        loss.item(),
                        torch.sum(preds == labels.data).item()/inputs.size(0),
                        self.optimizer.param_groups[-1]["lr"],
                        batch_number + self.epoch * self.iter_per_epoch
                    )
            # sum_pred = torch.round(sigmoid(sum_output)).cpu()
            _, sum_pred = torch.max(sum_output, 1)
            if not self.args["nni"]:
                pbar.finish()
            time_elapsed = time.time() - since
            epoch_result["loss"] = running_loss / self.dataset_sizes[phase]
            epoch_result["acc"] = running_corrects / self.dataset_sizes[phase]
            epoch_result["f1s"] = metrics.f1_score(
                sum_label.numpy(), sum_pred.numpy(), pos_label=0
            )
            if phase == "val":
                self.tb_write(
                    "Fold " + str(self.fold)+"/Plane"+str(self.args["plane"]),
                    phase,
                    epoch_result["loss"],
                    epoch_result["acc"],
                    self.optimizer.param_groups[-1]["lr"],
                    (self.epoch + 1) * self.iter_per_epoch,
                )
            self.result[phase] = epoch_result.copy()
        
        if self.epoch == 0:
            self.val_label = sum_label
        log = " {:.5f}".format(self.optimizer.param_groups[-1]["lr"]) + " |"
        for p in ["train", "val"]:
            for result in self.result[p]:
                log += " {:.4f} |".format(self.result[p][result])
        self.logger.info(
            "{:2}/{},{:2}/{} |{}| {:.0f}m{:.0f}s".format(
                self.fold,
                self.args["fold_num"],
                self.epoch + 1,
                self.args["epochs_num"],
                log,
                time_elapsed // 60,
                time_elapsed % 60,
            )
        )
        self.confusion.append(metrics.confusion_matrix(
            sum_pred, sum_label))  # type:ignore
        print("confusion martix:\n{}".format(self.confusion[-1]))
        self.epoch += 1
        return 0

    def print_wrong(self, dictname, inputs, pred, labels, batch_number):
        wrong_idx = (pred != labels.view_as(pred)).nonzero()[:, 0]
        wrong_samples = inputs[wrong_idx]
        wrong_preds = pred[wrong_idx]
        actual_preds = labels.view_as(pred)[wrong_idx]
        for i in enumerate(wrong_idx):
            sample = wrong_samples[i]
            wrong_pred = wrong_preds[i]
            actual_pred = actual_preds[i]
            img = to_pil_image(sample)
            idx = wrong_idx[i] + batch_number * self.args["batch_size"]
            if not os.path.exists("wrong/" + dictname):
                os.makedirs("wrong/" + dictname)
            img.save(
                "wrong/"
                + dictname
                + "/pred{}_actual{}_idx{}_.png".format(
                    wrong_pred.item(), actual_pred.item(), idx
                )
            )

    def test(self, dictname="", print_w=False):
        tcat = torch.cat
        tmax = torch.max
        # sigmoid = nn.Sigmoid()
        device = self.device
        sum_output = torch.empty([0])
        sum_label = torch.empty([0])
        batch_number = 0
        for inputs, labels in self.dataloaders["val"]:
            batch_number += 1
            self.model.eval()
            inputs = inputs.to(device)
            labels = labels.to(device)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
            # outputs = torch.squeeze(outputs, dim=1)
            sum_output = tcat((sum_output, outputs.cpu()), 0)
            sum_label = tcat((sum_label, labels.cpu()), 0)
            if print_w:
                # pred = torch.round(sigmoid(outputs))
                _, pred = tmax(outputs.cpu(), 1)
                self.print_wrong(dictname, inputs, pred, labels, batch_number)
            if len(self.val_label) < self.dataset_sizes['val']:
                self.val_label = sum_label

        _, pred_sum = tmax(sum_output, 1)
        self.result["acc"] = sum(
            pred_sum == self.val_label) / len(self.val_label)
        self.logger.info(self.result["acc"])
        return sum_output.cpu()

    def stop(self):
        print("-" * 10 + "END" + "-" * 10)
        for p in ["train", "val"]:
            for result in self.result[p]:
                self.logger.info(
                    "{:5} {:4}:{}".format(p, result, self.result[p][result])
                )
        self.logger.info("confusion martix:")
        self.logger.info(self.confusion[-1])
        self.writer["val"].close()
        self.writer["train"].close()

    def tb_write(self, name, phase, loss, acc, LR, number):
        if self.logger.name == "log":
            self.writer[phase].add_scalar(name+"/Loss", loss, number)
            self.writer[phase].add_scalar(name+"/Accuracy", acc, number)
            if phase == "train":
                self.writer[phase].add_scalar(name+"/LearningRate", LR, number)
