import os
import sys
import copy
import torch
import numpy as np
from pprint import pprint

if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))

from oil.tuning.args import argupdated_config

from src.train.det_trainer import make_trainer
from src.models import CHNN
from src.systems.chain_pendulum import ChainPendulum
from src.viz.pendulum_uncertainty import viz_static, viz_dynamic

defaults = copy.deepcopy(make_trainer.__kwdefaults__)
defaults["save"] = False

cfg = argupdated_config(defaults)
cfg.pop('local_rank')
save = cfg.pop('save')

cfg["network"] = CHNN
cfg["device"] = None
cfg["body"] = ChainPendulum(3)
cfg["C"] = 10
cfg["num_epochs"] = 50

pprint(cfg)

trainer = make_trainer(**cfg)
trainer.train(cfg['num_epochs'])

cfg["C"] = 60
test_t = make_trainer(**cfg)

dataloader = test_t.dataloaders["test"]
for mb in test_t.dataloaders["test"]:
    (z0, T), true_zs = mb
    break

T = torch.linspace(0, T[0,-1] - T[0,0], 3*len(T[0]))

num_samples = 10
pred_zs = []
for _ in range(num_samples):
    trainer.model.sample()
    with torch.no_grad():
    	zt_pred = trainer.model.integrate_swag(z0, T, tol=1e-7, method='dopri5')
    pred_zs.append(zt_pred.numpy())
pred_zs = np.stack(pred_zs, axis=0)

pred_pos = pred_zs[:,:,:,0,:,:]
true_pos = true_zs[:,:,0,:,:]
viz_static(pred_pos, true_pos)
# viz_dynamic(pred_pos, true_pos)
