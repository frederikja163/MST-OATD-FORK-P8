import numpy as np

from tdrive import main as tdrive
from porto import main as porto
from cd import main as cd
from utils import main as util, get_logger
from config import args
import random

if __name__ == '__main__':
    random.seed(1234)
    np.random.seed(1234)

    if args.find_boundary:
        util()
    else:
        match args.dataset:
            case "tdrive":
                tdrive()
            case "porto":
                porto()
            case "cd":
                cd()


