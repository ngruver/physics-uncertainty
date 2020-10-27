import os
import sys
import copy
import torch
from pprint import pprint

if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))

from oil.tuning.args import argupdated_config

from src.train.det_trainer import make_trainer
from src.models import HNN,NN
from src.systems.chain_pendulum import ChainPendulum

defaults = copy.deepcopy(make_trainer.__kwdefaults__)
defaults["save"] = False

cfg = argupdated_config(defaults)
cfg.pop('local_rank')
save = cfg.pop('save')

cfg["network"] = NN
cfg["device"] = None
cfg["body"] = ChainPendulum(1)
cfg["C"] = 10
cfg["num_epochs"] = 100

pprint(cfg)

trainer = make_trainer(**cfg)
trainer.train(cfg['num_epochs'])

trainer.model.swag_model.sample()

dataloader = trainer.dataloaders["test"]
for mb in trainer.dataloaders["test"]:
    z0, T = mb[0]
    break

T = T[0]
body = dataloader.dataset.body
long_T = body.dt * torch.arange(10*body.integration_time//body.dt).to(z0.device, z0.dtype)

num_samples = 10
sampled_trajs = []
for _ in range(num_samples):
    trainer.model.swag_model.sample()
    zt_pred = trainer.model.integrate_swag(z0, long_T, tol=1e-7, method='dopri5')
    sampled_trajs.append(zt_pred)

