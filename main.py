import gc
import numpy
import logging
import os
import time

import torch
from nni.utils import merge_parameter
from sklearn.metrics import confusion_matrix

import arg
import lib
import nets
import nni
import train
import json

TRIAL_NAME = ""
TRIAL_TIME = ""
TL_BASE = ""
DATA_DIR = ""
FOLD_NUM = ""
logger = logging.getLogger("silence")


def initialize(log):
    global TRIAL_NAME, TRIAL_TIME, TL_BASE, DATA_DIR, FOLD_NUM, logger
    TRIAL_NAME = "321_PURE_CI2"
    TRIAL_TIME = time.strftime("%y%m%d-%H%M", time.localtime())
    DATA_DIR = "/home/ray/TrainingData/321/plane"

    FOLD_NUM = len(os.listdir(DATA_DIR+"/2"))
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True  # use cudnn

    if log:
        logger = lib.set_logger(TRIAL_TIME, TRIAL_NAME)

def get_model(args):
    model = nets.vggnet(args)
    return model


def run(args, train_runner, save=False, sub_name=""):
    if TL_BASE != "":
        train_runner.load_state(TL_BASE)
    try:
        for _ in range(args["epochs_num"]):
            if train_runner.train():  # train will return status
                break
            nni.report_intermediate_result(train_runner.result['val']['f1s'])
            torch.save(train_runner.model.state_dict(), "temp.dict")
    except KeyboardInterrupt:
        logger.info("-" * 25 + "Interrupted" + "-" * 25)
    nni.report_final_result(train_runner.result['val']['f1s'])
    train_runner.stop()
    print(train_runner.result["val"]["acc"])
    if save:
        dictname = "state_dict/{}_{}.dict".format(TRIAL_TIME, sub_name)
        torch.save(train_runner.model.state_dict(), dictname)


def plane_training(args, fold_num, data_dir, logger_input):
    savename = "save/"+TRIAL_NAME+".save"
    if os.path.exists(savename):
        result = torch.load(savename)
    else:
        result = []
    cm = numpy.zeros([2, 2])
    label = None

    for fold in range(len(result)+1, fold_num+1):
        output = dict()
        for plane in [2,1,0]:
            gc.collect()
            logger.info('-'*25+"plane:{}".format(plane)+'-'*25)
            torch.cuda.empty_cache()
            # dataloaders = lib.get_dataloaders_all_shuffle(args,fold,fold_num,plane,data_dir)
            dataloaders = lib.get_dataloaders(args, fold, fold_num, plane, data_dir)
            tr = train.TrainRunner(args, logger_input, dataloaders, fold)
            args['plane'] = plane
            run(args, tr, False, str(plane))
            output[plane] = tr.test()
            label = tr.val_label.cpu()
            del tr
        if plane == 2:
            calc_output = output[2]
        else:
            calc_output = (output[0]*1+output[1]*1+output[2]*2) / 4
        _, pred = torch.max(calc_output, 1)
        logger_input.info("fold {} acc:{}".format(
            fold, sum(pred == label) / len(label)))
        cm = cm + confusion_matrix(pred, label)
        logger_input.info("confusion_matrix:{}".format(
            confusion_matrix(pred, label)))
        result.append([output, label, calc_output])
        torch.save(result,"save/"+TRIAL_NAME+".save")
    return result


def main(args):
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    global DATA_DIR
    # log information
    logger.info("DATA_DIR:" + DATA_DIR)
    logger.info("BASE ON:" + TL_BASE)
    logger.info('TRAIN NAME:"' + TRIAL_NAME + '"')

    # primary arguments
    args["TRIAL_TIME"] = TRIAL_TIME
    args["TRIAL_NAME"] = TRIAL_NAME
    args["fold_num"] = FOLD_NUM

    plane_training(args, FOLD_NUM, DATA_DIR, logger)


if __name__ == "__main__":
    try:
        log = 1
        save_args = 0
        initialize(log)
        tuner_params = nni.get_next_parameter()
        logger.info(tuner_params)
        if os.path.exists("arguments/"+TRIAL_NAME+".args") and log:
            params = torch.load("arguments/"+TRIAL_NAME+".args")
            logger.info("loaded arguments from save/"+TRIAL_NAME+".args")
        else:
            params = vars(merge_parameter(arg.get_params(), tuner_params))
            if save_args:
                torch.save(params, "arguments/"+TRIAL_NAME+".args")
                logger.info("arguments saved to save/"+TRIAL_NAME+".args")
        logger.info("args:\n"+json.dumps(params, sort_keys=False, indent=4))
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
