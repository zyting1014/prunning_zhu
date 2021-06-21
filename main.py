from util.option_hinge import args
from data import Data
from util import utility
from loss import Loss
from util.trainer_hinge import Trainer
from model_hinge import Model
from model_hinge.hinge_utility import calc_model_complexity, calc_model_complexity_running, plot_compression_ratio
from model_hinge.hinge_utility import calc_model_complexity_running_new, calc_model_complexity_running_sp_layers
from util.record import save

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
    # layer_num = 4
    # print("layer_num = %d" % layer_num)

    # calc_model_complexity_running_new(my_model, 3, 71)  # 72 cpu
    # calc_model_complexity_running_new(my_model, 3, 87)  # 88 gpu

    while current_ratio > args.ratio and current_ratio - args.ratio > args.stop_limit and not t.terminate():
        # my_model.get_model().layer_num = layer_num
        t.train()
        t.test()
        # 剪一层
        # calc_model_complexity_running_new(my_model, layer_num, 0)
        # 剪整个network
        # calc_model_complexity_running(my_model, False)
        # 剪随机某几个层
        calc_model_complexity_running_sp_layers(my_model, 4, 0)
        current_ratio = my_model.get_model().flops_compress / my_model.get_model().flops
        current_ratio_list.append("{:.4f}".format(current_ratio))
        print("current_ratio_list : ")
        print(current_ratio_list)

        my_model.get_model().parameter_ratio_list.append(
            "{:.4f}".format(my_model.get_model().params_compress / my_model.get_model().params))
        print("parameter ratio list : ")
        print(my_model.get_model().parameter_ratio_list)

    save(current_ratio_list, t.model.get_model().timer_test_list,
         t.model.get_model().sum_list, t.model.get_model().top1_err_list, my_model.get_model().parameter_ratio_list)

    print("done")
