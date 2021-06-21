# Cluster Pruning: An Efficient Filter Pruning Method for Edge AI Vision Applications
# 专门测试这篇论文的类
from util.option_hinge import args
from data import Data
from util import utility
from loss import Loss
from util.trainer_hinge import Trainer
from model_hinge import Model
from model_hinge.hinge_utility import calc_model_complexity, calc_model_complexity_running, plot_compression_ratio
from model_hinge.hinge_utility import calc_model_complexity_running_new

args.distillation = True

checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = Data(args)

    my_model = Model(args, checkpoint)
    model_teacher = Model(args, checkpoint, teacher=True) if args.distillation else None

    loss = Loss(args, checkpoint)

    t = Trainer(args, loader, my_model, loss, checkpoint, None, False, model_teacher)

    current_ratio_list = []
    current_ratio, ratio_log = 1.0, []

    my_model.get_model().algorithm(t, 4, 0.99)

    # print("###################################################")
    # print("################进入fine-tune阶段！！###############")
    # print("###################################################")
    # for i in range(200):
    #     t.train()
    #     t.test()
    print("done")
