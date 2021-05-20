from util.option_hinge import args
from data import Data
from util import utility
from loss import Loss
from util.trainer_hinge import Trainer
from model_hinge import Model
from tensorboardX import SummaryWriter
from model_hinge.hinge_utility import calc_model_complexity, calc_model_complexity_running, plot_compression_ratio


checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = Data(args)

    my_model = Model(args, checkpoint)
    loss = Loss(args, checkpoint)

    t = Trainer(args, loader, my_model, loss, checkpoint, None, False, None)

    current_ratio, ratio_log = 1.0, []

    while current_ratio - args.ratio > args.stop_limit and not t.terminate():
        t.train()
        t.test()
        calc_model_complexity_running(my_model, False)
        current_ratio = my_model.get_model().flops_compress / my_model.get_model().flops
        ratio_log.append(current_ratio)