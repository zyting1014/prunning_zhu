from util.option_hinge import args
from data import Data
from util import utility
from loss import Loss
from util.trainer_hinge import Trainer
from model_hinge import Model
import os

checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = Data(args)
    my_model = Model(args, checkpoint)
    loss = Loss(args, checkpoint)
    t = Trainer(args, loader, my_model, loss, checkpoint, None, False, None)
    for i in range(305):
        t.test()