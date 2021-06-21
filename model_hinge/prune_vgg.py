import torch
import torch.nn as nn
from model.vgg import VGG
from model.common import BasicBlock
from model_hinge.hinge_utility import init_weight_proj, get_nonzero_index, plot_figure, plot_per_layer_compression_ratio
from model_hinge.hinge_utility import get_nonzero_index_spec_layer_num, calc_model_complexity, get_nonzero_index_spec_threshold
from model.in_use.flops_counter import get_model_complexity_info
import math
import numpy as np
from util.record import save


def compress_module_param(module, percentage, threshold, index_pre=None, i=0):
    conv11 = module[0]
    batchnorm1 = module[1]

    ws1 = conv11.weight.shape
    weight1 = conv11.weight.data.view(ws1[0], -1).t()

    bias1 = conv11.bias.data if conv11.bias is not None else None

    bn_weight1 = batchnorm1.weight.data
    bn_bias1 = batchnorm1.bias.data
    bn_mean1 = batchnorm1.running_mean.data
    bn_var1 = batchnorm1.running_var.data

    pindex1 = get_nonzero_index(weight1, dim='output', percentage=percentage, threshold=threshold)[1]

    # conv11
    if index_pre is not None:
        index = torch.repeat_interleave(index_pre, ws1[2] * ws1[3]) * ws1[2] * ws1[3] \
                + torch.tensor(range(0, ws1[2] * ws1[3])).repeat(index_pre.shape[0]).cuda()
        weight1 = torch.index_select(weight1, dim=0, index=index)
    if i < 11:
        weight1 = torch.index_select(weight1, dim=1, index=pindex1)
        conv11.bias = nn.Parameter(torch.index_select(bias1, dim=0, index=pindex1))
        conv11.weight = nn.Parameter(weight1.t().view(pindex1.shape[0], -1, ws1[2], ws1[3]))

        batchnorm1.weight = nn.Parameter(torch.index_select(bn_weight1, dim=0, index=pindex1))
        batchnorm1.bias = nn.Parameter(torch.index_select(bn_bias1, dim=0, index=pindex1))
        batchnorm1.running_mean = torch.index_select(bn_mean1, dim=0, index=pindex1)
        batchnorm1.running_var = torch.index_select(bn_var1, dim=0, index=pindex1)
        batchnorm1.num_features = len(batchnorm1.weight)
    else:
        conv11.weight = nn.Parameter(weight1.t().view(ws1[0], -1, ws1[2], ws1[3]))

    conv11.out_channels, conv11.in_channels = conv11.weight.size()[:2]


def compress_module_param_specific_number(module, percentage, threshold, index_pre=None, i=0):
    #  此函数是从compress_module_param复制来的 为了调用get_nonzero_index_spec_threshold
    #  完全可以删除
    conv11 = module[0]
    batchnorm1 = module[1]

    ws1 = conv11.weight.shape
    weight1 = conv11.weight.data.view(ws1[0], -1).t()

    bias1 = conv11.bias.data if conv11.bias is not None else None

    bn_weight1 = batchnorm1.weight.data
    bn_bias1 = batchnorm1.bias.data
    bn_mean1 = batchnorm1.running_mean.data
    bn_var1 = batchnorm1.running_var.data

    pindex1 = get_nonzero_index_spec_threshold(weight1, dim='output', threshold=threshold)[1]


    # conv11
    if index_pre is not None:
        index = torch.repeat_interleave(index_pre, ws1[2] * ws1[3]) * ws1[2] * ws1[3] \
                + torch.tensor(range(0, ws1[2] * ws1[3])).repeat(index_pre.shape[0]).cuda()
        weight1 = torch.index_select(weight1, dim=0, index=index)
    if i < 11:
        weight1 = torch.index_select(weight1, dim=1, index=pindex1)
        conv11.bias = nn.Parameter(torch.index_select(bias1, dim=0, index=pindex1))
        conv11.weight = nn.Parameter(weight1.t().view(pindex1.shape[0], -1, ws1[2], ws1[3]))

        batchnorm1.weight = nn.Parameter(torch.index_select(bn_weight1, dim=0, index=pindex1))
        batchnorm1.bias = nn.Parameter(torch.index_select(bn_bias1, dim=0, index=pindex1))
        batchnorm1.running_mean = torch.index_select(bn_mean1, dim=0, index=pindex1)
        batchnorm1.running_var = torch.index_select(bn_var1, dim=0, index=pindex1)
        batchnorm1.num_features = len(batchnorm1.weight)
    else:
        conv11.weight = nn.Parameter(weight1.t().view(ws1[0], -1, ws1[2], ws1[3]))

    conv11.out_channels, conv11.in_channels = conv11.weight.size()[:2]



def make_model(args, ckp, converging):
    return Prune(args, ckp, converging)


def findMinGroup(G_MWG):
    key = min(G_MWG, key=lambda x: G_MWG[x])
    layer_num = int(key.split("#")[0])
    group_num = int(key.split("#")[1])
    return layer_num, group_num, key


class Prune(VGG):

    def __init__(self, args, ckp, converging):
        self.args = args
        self.ckp = ckp
        super(Prune, self).__init__(self.args)

        # traning or loading for searching
        if not self.args.test_only and not converging:
            self.load(self.args, strict=True)

        if self.args.data_train.find('CIFAR') >= 0:
            self.input_dim = (3, 32, 32)
        elif self.args.data_train.find('Tiny_ImageNet') >= 0:
            self.input_dim = (3, 64, 64)
        else:
            self.input_dim = (3, 224, 224)
        self.flops, self.params = get_model_complexity_info(self, self.input_dim, print_per_layer_stat=False)

    def find_modules(self):
        return [m for m in self.modules() if isinstance(m, BasicBlock)][1:]

    def sparse_param(self, module):
        param1 = module.state_dict(keep_vars=True)['0.weight']
        ws1 = param1.shape
        param1 = param1.view(ws1[0], -1).t()

        return param1

    def index_pre(self, percentage, threshold):
        index = []
        for module_cur in self.find_modules():
            conv11 = module_cur[0]
            # projection1 = conv11.weight.data.squeeze().t()

            ws1 = conv11.weight.shape
            projection1 = conv11.weight.data.view(ws1[0], -1).t()

            index.append(get_nonzero_index(projection1, dim='output', percentage=percentage, threshold=threshold)[1])
        return index

    def index_pre_sp(self, percentage, threshold):
        index = []
        for module_cur in self.find_modules():
            conv11 = module_cur[0]
            # projection1 = conv11.weight.data.squeeze().t()

            ws1 = conv11.weight.shape
            projection1 = conv11.weight.data.view(ws1[0], -1).t()

            index.append(get_nonzero_index_spec_threshold(projection1, dim='output', threshold=threshold)[1])
        return index

    def getThreshold(self, num=4):
        # filter全局排序 返回最不重要n个filter的二范数阈值
        modules = self.find_modules()
        conv = torch.empty(0)

        for i, module_cur in enumerate(modules):
            conv_each = module_cur[0]
            ws1 = conv_each.weight.shape
            weight1 = conv_each.weight.data.view(ws1[0], -1).t()
            n = torch.norm(weight1, p=2, dim=0)
            if i == 0:
                conv = n
            else:
                conv = torch.cat([conv, n], 0)
        print(conv)
        print("conv size : ")
        print(conv.shape)

        return conv.sort().values[num - 1]

    def compress_sp_threshold(self, **kwargs):
        threshold = self.getThreshold()
        print("threshold = ")
        print(threshold)
        index = [None] + self.index_pre_sp(self.args.remain_percentage, threshold)
        for i, module_cur in enumerate(self.find_modules()):
             compress_module_param_specific_number(module_cur, self.args.remain_percentage, threshold, index[i], i)


    def compress(self, **kwargs):
        index = [None] + self.index_pre(self.args.remain_percentage, self.args.threshold)
        for i, module_cur in enumerate(self.find_modules()):
            compress_module_param(module_cur, self.args.remain_percentage, self.args.threshold, index[i], i)

    def set_channels(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.out_channels, m.in_channels = m.weight.size()[:2]
            elif isinstance(m, nn.BatchNorm2d):
                m.num_features = m.weight.size()[0]

    def load_state_dict(self, state_dict, strict=True):
        if strict:
            # used to load the model parameters during training
            super(Prune, self).load_state_dict(state_dict, strict)
        else:
            # used to load the model parameters during test
            own_state = self.state_dict(keep_vars=True)
            for name, param in state_dict.items():
                if name in own_state:
                    if isinstance(param, nn.Parameter):
                        param = param.data
                    if param.size() != own_state[name].size():
                        own_state[name].data = param
                    else:
                        own_state[name].data.copy_(param)
            self.set_channels()

    def compress_one_layer(self, layer_num, prun_pum, **kwargs):
        print("要剪枝的BasicBlock卷积层编号 ：" + str(layer_num))
        print("剪枝的filter数 ： " + str(prun_pum + 1))

        modules = [m for m in self.modules() if isinstance(m, BasicBlock)][0:]
        cur_layer = modules[layer_num]
        next_layer = modules[layer_num + 1]

        # cur_layer = self.find_modules()[layer_num]
        # next_layer = self.find_modules()[layer_num + 1]

        conv11 = cur_layer[0]
        batchnorm1 = cur_layer[1]
        ws1 = conv11.weight.shape  # (output, input, kernel_size, kernel_size)
        weight1 = conv11.weight.data.view(ws1[0], -1).t()
        bias1 = conv11.bias.data if conv11.bias is not None else None
        bn_weight1 = batchnorm1.weight.data
        bn_bias1 = batchnorm1.bias.data
        bn_mean1 = batchnorm1.running_mean.data
        bn_var1 = batchnorm1.running_var.data

        if 'pindex1' not in kwargs:
            pindex1 = get_nonzero_index_spec_layer_num(weight1, prun_pum)[1]
        else:
            print("pindex1 shape: " + str(kwargs['pindex1'].shape[0]))
            pindex1 = kwargs['pindex1']

        # conv current layer
        weight1 = torch.index_select(weight1, dim=1, index=pindex1)
        conv11.bias = nn.Parameter(torch.index_select(bias1, dim=0, index=pindex1))
        conv11.weight = nn.Parameter(weight1.t().view(pindex1.shape[0], -1, ws1[2], ws1[3]))
        batchnorm1.weight = nn.Parameter(torch.index_select(bn_weight1, dim=0, index=pindex1))
        batchnorm1.bias = nn.Parameter(torch.index_select(bn_bias1, dim=0, index=pindex1))
        batchnorm1.running_mean = torch.index_select(bn_mean1, dim=0, index=pindex1)
        batchnorm1.running_var = torch.index_select(bn_var1, dim=0, index=pindex1)
        batchnorm1.num_features = len(batchnorm1.weight)
        conv11.out_channels, conv11.in_channels = conv11.weight.size()[:2]

        # conv next layer
        conv12 = next_layer[0]
        ws2 = conv12.weight.shape
        weight2 = conv12.weight.data.view(ws2[0], -1).t()

        pindex2 = torch.repeat_interleave(pindex1, ws2[2] * ws2[3]) * ws2[2] * ws2[3] \
                  + torch.tensor(range(0, ws2[2] * ws2[3])).repeat(pindex1.shape[0]).cuda()

        weight2 = torch.index_select(weight2, dim=0, index=pindex2)

        conv12.weight = nn.Parameter(weight2.t().view(ws2[0], -1, ws2[2], ws2[3]))
        conv12.in_channels = conv12.weight.size()[1]

    def algorithm(self, t, P, ratio):
        # 输入：预训练包含滤波器集合F的网络
        # 每组滤波器由几个滤波器组成 ：P
        # 输出：剪枝后的包含滤波器集合F'的网络
        # index = []
        current_ratio_list = []
        G_MWG = {}  # {key : layer_num#channel_num#P, value : importance}

        # 计算每组filter的重要性
        modules = [m for m in self.modules() if isinstance(m, BasicBlock)][0:11]

        for layer, module_cur in enumerate(modules):  # 不剪最后一层
            conv11 = module_cur[0]
            ws1 = conv11.weight.shape
            projection1 = conv11.weight.data.view(ws1[0], -1).t()  # reshape (input * k * k,output)
            Fl = torch.norm(projection1, p=2, dim=0)  # shape : output
            # FR eg: tensor([0, 2, 3, 1])
            FR = Fl.sort()[1]  # 每个filter在当前层的重要性排序 按二范数升序排列 越小越容易被剪枝
            filter_num = FR.shape[0]
            # print("FR : ")
            # print(FR)
            cluster_num = math.ceil(filter_num / P)
            factors = np.zeros(cluster_num)
            for i in range(filter_num):  # 当前层第i名
                filter_indice = FR[i]  # 对应的index索引
                factors[int(filter_indice / P)] = factors[int(filter_indice / P)] + (Fl[filter_indice] * i) / P
            for cluster in range(cluster_num):
                key = str(layer) + '#' + str(cluster) + '#' + str(P)
                G_MWG[key] = factors[cluster]
            # print("factors : ")
            # print(factors)

        current_ratio = 1.0
        self.flops_compress = self.flops

        while current_ratio > ratio:
            # t.train()
            t.test()
            # 以上是常规操作
            ######################################################
            (layer_num, group_num, key) = findMinGroup(G_MWG)
            del G_MWG[key]  # 删除该元素
            print("prun layer :%d, group : %d" % (layer_num, group_num))
            cur_layer = modules[layer_num]
            conv11 = cur_layer[0]
            ws1 = conv11.weight.shape
            weight1 = conv11.weight.data.view(ws1[0], -1).t()
            pindex1 = torch.ones(weight1.shape[1]).to(weight1.device)
            pindex1[group_num * P:group_num * P + P] = 0
            pindex1 = torch.nonzero(pindex1).squeeze(dim=1)
            self.compress_one_layer(layer_num, -1, pindex1=pindex1)

            # calc_model_complexity(self)

            self.flops_compress, self.params_compress = get_model_complexity_info(self, self.input_dim,
                                                                                  print_per_layer_stat=False)
            print('FLOPs ratio {:.2f} = {:.4f} [G] / {:.4f} [G]; Parameter ratio {:.2f} = {:.2f} [k] / {:.2f} [k].'
                  .format(self.flops_compress / self.flops * 100, self.flops_compress / 10. ** 9,
                          self.flops / 10. ** 9,
                          self.params_compress / self.params * 100, self.params_compress / 10. ** 3,
                          self.params / 10. ** 3))

            current_ratio = self.flops_compress / self.flops
            t.model.get_model().current_ratio_list.append("{:.4f}".format(current_ratio))
            print("current_ratio_list : ")
            print(t.model.get_model().current_ratio_list)

            t.model.get_model().parameter_ratio_list.append(
                "{:.4f}".format(self.params_compress / self.params))
            print("parameter ratio list : ")
            print(t.model.get_model().parameter_ratio_list)

        save(t.model.get_model().current_ratio_list, t.model.get_model().timer_test_list,
             t.model.get_model().sum_list, t.model.get_model().top1_err_list, t.model.get_model().parameter_ratio_list)

