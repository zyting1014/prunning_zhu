import torch
import torch.nn as nn
from model.vgg import VGG
from model.common import BasicBlock
from model_hinge.hinge_utility import init_weight_proj, get_nonzero_index, plot_figure, plot_per_layer_compression_ratio
from model.in_use.flops_counter import get_model_complexity_info


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


def make_model(args, ckp, converging):
    return Prune(args, ckp, converging)


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
