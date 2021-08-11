import argparse


def get_params():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--nni", type=bool, default=False, metavar="SG")
    parser.add_argument("--ekko", type=bool, default=False, metavar="EK")

    parser.add_argument("--sigmoid", type=bool, default=False, metavar="SG")
    parser.add_argument("--normalization", type=bool, default=False, metavar="nor")
    parser.add_argument("--weight_decay", type=float, default=0.0001, metavar="WD")#417 0.0001

    parser.add_argument("--degrees", type=int, default=0.0, metavar="dge")
    parser.add_argument("--xtras", type=float, default=0.2, metavar="xtra")
    parser.add_argument("--batch_size", type=int, default=32, metavar="N")

    parser.add_argument("--lr", type=float, default=1e-2, metavar="LR")
    parser.add_argument("--base_lr", type=float, default=1e-5, metavar="LR")
    parser.add_argument("--max_lr", type=float, default=1e-2, metavar="LR")

    parser.add_argument("--flood_level", type=float, default=0.07, metavar="fl") #
    # 0.09  317 
    # 0.025 417  
    # 0.06  418 
    # 0.025 316
    parser.add_argument("--momentum", type=float, default=0.9, metavar="MN")
    parser.add_argument("--epochs_num", type=int, default=6, metavar="EP") #8 417
    parser.add_argument("--step_size", type=int, default=1, metavar="SS")
    parser.add_argument("--workers", type=int, default=4, metavar="WK")
    '''
    parser.add_argument("--CJ_brightness", type=float, default=1.0, metavar="CJB")
    parser.add_argument("--CJ_hue", type=float, default=0.0, metavar="CJH")
    parser.add_argument("--RandomVerticalFlip", type=float, default=0.0, metavar="RVF")
    parser.add_argument("--RandomHorizontalFlip", type=float, default=0.0, metavar="RHF")
    parser.add_argument("--seed", type=int, default=1024, metavar="S")
    '''
    args, _ = parser.parse_known_args()
    return args

