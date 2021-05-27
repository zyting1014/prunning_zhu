from util.option_hinge import args
from data import Data
from util import utility
from loss import Loss
from util.trainer_hinge import Trainer
from model_hinge import Model
from model_hinge.hinge_utility import calc_model_complexity, calc_model_complexity_running, plot_compression_ratio


checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = Data(args)

    my_model = Model(args, checkpoint)
    model_teacher = Model(args, checkpoint, teacher=True) if args.distillation else None

    loss = Loss(args, checkpoint)

    t = Trainer(args, loader, my_model, loss, checkpoint, None, False, model_teacher)

    current_ratio, ratio_log = 1.0, []

    while current_ratio > args.ratio and current_ratio - args.ratio > args.stop_limit and not t.terminate():
        t.train()
        t.test()
        calc_model_complexity_running(my_model, False)



    print("done")
