import torch
import matplotlib
import os

from IPython import embed
from tqdm import tqdm
from tensorboardX import SummaryWriter
from util import utility
from model_hinge.hinge_utility import reg_anneal
from loss import distillation
matplotlib.use('Agg')
#from IPython import embed


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp, writer=None, converging=False, model_teacher=None):
        self.args = args
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.model_teacher = model_teacher
        self.loss = my_loss
        self.converging = converging
        self.writer = writer
        self.lr_adjust_flag = self.args.model.lower().find('resnet') >= 0 #TODO: print rP

        self.optimizer = utility.make_optimizer_hinge(args, self.model, ckp, self.converging, self.lr_adjust_flag)
        self.scheduler = utility.make_scheduler_hinge(args, self.optimizer, self.converging, self.lr_adjust_flag)

        # if self.args.optimizer != 'SGD':
        #     self.optimizer = utility.make_optimizer_hinge(args, self.model, ckp, self.converging, self.lr_adjust_flag)
        #     self.scheduler = utility.make_scheduler_hinge(args, self.optimizer, self.converging, self.lr_adjust_flag)
        # else:
        #     # 单纯训练一个网络
        #     self.optimizer = utility.make_optimizer(args, self.model, ckp=ckp)
        #     self.scheduler = utility.make_scheduler(args, self.optimizer)

        self.device = torch.device('cpu' if args.cpu else 'cuda')

        if args.model.find('INQ') >= 0:
            self.inq_steps = args.inq_steps
        else:
            self.inq_steps = None

    def reset_after_optimization(self, epoch_continue):
        if not self.converging and not self.args.test_only:
            self.converging = True
            # In Phase 1 & 3, the optimizer and scheduler are reset.
            # In Phase 2, the optimizer and scheduler is not used.
            # In Phase 4, the optimizer and scheduler is already set during the initialization of the trainer.
            # during the converging stage, self.converging =True. Do not need to set lr_adjust_flag in make_optimizer_hinge
            #   and make_scheduler_hinge.
            self.optimizer = utility.make_optimizer_hinge(self.args, self.model, self.ckp, self.converging)
            self.scheduler = utility.make_scheduler_hinge(self.args, self.optimizer, self.converging)
        if not self.args.test_only and self.args.summary:
            self.writer = SummaryWriter(os.path.join(self.args.dir_save, self.args.save), comment='converging')
        self.epoch_continue = epoch_continue


    def train(self):
        epoch = self.start_epoch()
        self.model.begin(epoch, self.ckp)
        self.loss.start_log()
        # modules = self.model.get_model().find_modules() #TODO: merge this
        timer_data, timer_model = utility.timer(), utility.timer()
        n_samples = 0

        for batch, (img, label) in enumerate(self.loader_train):
            # if batch<=1:
            img, label = self.prepare(img, label)
            n_samples += img.size(0)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            # embed()
            prediction = self.model(img)
            loss, _ = self.loss(prediction, label)

            if self.args.distillation:
                with torch.no_grad():
                    prediction_teacher = self.model_teacher(img)
                loss_distill = distillation(prediction, prediction_teacher, T=4)
                loss = loss_distill * 0.4 + loss * 0.6

            loss.backward()
            self.optimizer.step()

            timer_model.hold()

            if self.args.summary:
                if (batch + 1) % 50 == 0:
                    for name, param in self.model.named_parameters():
                        if name.find('features') >= 0 and name.find('weight') >= 0:
                            self.writer.add_scalar('data/' + name, param.clone().cpu().data.abs().mean().numpy(),
                                                   1000 * (epoch - 1) + batch)
                            if param.grad is not None:
                                self.writer.add_scalar('data/' + name + '_grad',
                                                       param.grad.clone().cpu().data.abs().mean().numpy(),
                                                       1000 * (epoch - 1) + batch)
                if (batch + 1) == 500:
                    for name, param in self.model.named_parameters():
                        if name.find('features') >= 0 and name.find('weight') >= 0:
                            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), 1000 * (epoch - 1) + batch)
                            if param.grad is not None:
                                self.writer.add_histogram(name + '_grad', param.grad.clone().cpu().data.numpy(),
                                                          1000 * (epoch - 1) + batch)

            timer_data.tic()


        self.model.log(self.ckp)
        self.loss.end_log(len(self.loader_train.dataset))


    def test(self):
        self.model.get_model().total_time = [0] * len(self.model.get_model().body_list)
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.loss.start_log(train=False)
        self.model.eval()

        timer_test = utility.timer()
        i = 0
        with torch.no_grad():
            for img, label in tqdm(self.loader_test, ncols=80):
                i = i + 1
                # if i == 5:
                #     break
                img, label = self.prepare(img, label)
                timer_test.tic()
                prediction = self.model(img)
                timer_test.hold()
                self.loss(prediction, label, train=False)

        current_time = timer_test.acc

        self.loss.end_log(len(self.loader_test.dataset), train=False)

        # Lower is better
        best = self.loss.log_test.min(0)
        for i, measure in enumerate(('Loss', 'Top1 error', 'Top5 error')):
            self.ckp.write_log('{}: {:.3f} (Best: {:.3f} from epoch {})'.
                               format(measure, self.loss.log_test[-1, i], best[0][i], best[1][i] + 1))

        if hasattr(self, 'epoch_continue') and self.converging:
            best = self.loss.log_test[:self.epoch_continue, :].min(0)
            self.ckp.write_log('\nBest during searching')
            for i, measure in enumerate(('Loss', 'Top1 error', 'Top5 error')):
                self.ckp.write_log('{}: {:.3f} from epoch {}'.format(measure, best[0][i], best[1][i]))

        self.ckp.write_log('Time: {:.2f}s\n'.format(current_time), refresh=True)

        is_best = self.loss.log_test[-1, self.args.top] <= best[0][self.args.top]
        self.ckp.save(self, epoch, converging=self.converging, is_best=is_best)
        # This is used by clustering convolutional kernels
        # self.ckp.save_results(epoch, self.model)

        # scheduler.step is moved from training procedure to test procedure
        self.scheduler.step()

        # 下面是新加的统计内容
        self.model.get_model().timer_test_list.append("{:.3f}".format(current_time))
        print("whole network inference time : ")
        print(self.model.get_model().timer_test_list)

        print("each layer time: ")
        for i in range(len(self.model.get_model().total_time)):
            self.model.get_model().total_time[i] = float("{:.5f}".format(self.model.get_model().total_time[i]))
        print(self.model.get_model().total_time)
        print("sum : ")
        print("{:.5f}".format(sum(self.model.get_model().total_time)))
        if self.model.get_model().layer_num != -1:
            self.model.get_model().spec_list.append("{:.5f}".format(self.model.get_model().total_time[self.model.get_model().layer_num]))
            print("the %d 's layer inference time list : " % self.model.get_model().layer_num)
            print(self.model.get_model().spec_list)
        self.model.get_model().sum_list.append("{:.5f}".format(sum(self.model.get_model().total_time)))
        print("sum list : ")
        print(self.model.get_model().sum_list)
        self.model.get_model().top1_err_list.append("{:.3f}".format(self.loss.log_test[-1, 1]))
        print("top1 error list : ")
        print(self.model.get_model().top1_err_list)



    def prepare(self, *args):
        def _prepare(x):
            x = x.to(self.device)
            if self.args.precision == 'half': x = x.half()
            return x

        return [_prepare(a) for a in args]

    def start_epoch(self):
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()
        if len(lr) == 1:
            s = '[Epoch {}]\tLearning rate: {:.2}'.format(epoch, lr[0])
        else:
            s = '[Epoch {}]\tLearning rate:'.format(epoch)
            for i, l in enumerate(lr):
                s += ' Group {} - {:.2}'.format(i, l) if i + 1 == len(lr) else ' Group {} - {:.2} /'.format(i, l)

        if not self.converging:
            stage = 'Searching Stage'
        else:
            stage = 'Converging Stage (Searching Epoch {})'.format(self.epoch_continue)
        s += '\t{}'.format(stage)
        self.ckp.write_log(s)
        return epoch

    def terminate(self):
        if self.args.test_only:
            # if self.args.model.lower().find('hinge') >= 0:
            #     self.model.get_model().compress()
            #     if self.args.model.lower() in ['hinge_resnet56', 'hinge_wide_resnet', 'hinge_densenet']:
            #         self.model.get_model().merge_conv()
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            if not self.converging:
                return epoch > 200
            else:
                return epoch > self.args.epochs


def proximal_operator_l0(optimizer, regularization, lr):
    for i, param in enumerate(optimizer.param_groups[1]['params']):
        ps = param.data.shape
        p = param.data.squeeze().t()
        eps = 1e-6
        if i % 2 == 0:
            n = torch.norm(p, p=2, dim=0)
            scale = (n > regularization).to(torch.float32)
            scale = scale.repeat(ps[0], 1)
            if torch.isnan(n[0]):
                embed()
        else:
            n = torch.norm(p, p=2, dim=1)
            scale = (n > regularization).to(torch.float32)
            scale = scale.repeat(ps[0], 1).t()
        # p = param.data
        # scale = torch.ones(ps).to(param.device) * 0.9
        param.data = torch.mul(scale, p).t().view(ps)


def proximal_operator_l1(optimizer, regularization, lr):
    for i, param in enumerate(optimizer.param_groups[1]['params']):
        ps = param.data.shape
        p = param.data.squeeze().t()
        eps = 1e-6
        if i % 2 == 0:
            n = torch.norm(p, p=2, dim=0)
            scale = torch.max(1 - regularization * lr / (n + eps), torch.zeros_like(n, device=n.device))
            # scale = scale.repeat(ps[0], 1)
            scale = scale.repeat(ps[1], 1)
            if torch.isnan(n[0]):
                embed()
        else:
            n = torch.norm(p, p=2, dim=1)
            scale = torch.max(1 - regularization * lr / (n + eps), torch.zeros_like(n, device=n.device))
            scale = scale.repeat(ps[0], 1).t()
        # p = param.data
        # scale = torch.ones(ps).to(param.device) * 0.9
        param.data = torch.mul(scale, p).t().view(ps)
