import os
import sys
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
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
cfg["C"] = 40
cfg["num_epochs"] = 200

pprint(cfg)

trainer = make_trainer(**cfg)
trainer.train(cfg['num_epochs'])

trainer.model.swag_model.sample()

dataloader = trainer.dataloaders["test"]
for mb in trainer.dataloaders["test"]:
    (z0, T), true_zs = mb
    break

# T = T[0]
body = dataloader.dataset.body
long_T = body.dt * torch.arange(body.integration_time//body.dt).to(z0.device, z0.dtype)

num_samples = 5
sampled_trajs = []
for _ in range(num_samples):
    trainer.model.swag_model.sample()
    zt_pred = trainer.model.integrate_swag(z0, long_T, tol=1e-7, method='dopri5')
    sampled_trajs.append(zt_pred.detach().numpy())

xy_trajs = np.array([t[0,:,0,0,:] for t in sampled_trajs])
pred_z_mean = np.mean(xy_trajs, axis=0)
all_pts = np.reshape(xy_trajs, (-1, 2))

xy_z0 = z0[0,0,0,:]
xy_true_zs = true_zs[0,:,0,0,:]

sns.kdeplot(x=all_pts[:,0], y=all_pts[:,1], fill=True, color='pink')#, label='pred dist')
plt.scatter(xy_z0[0], xy_z0[1], c='black')
plt.plot(xy_true_zs[:,0], xy_true_zs[:,1], c='b', label='ground truth')
plt.plot(pred_z_mean[:,0], pred_z_mean[:,1], c='darkred', label='pred mean')

legend_objs = []
legend_objs.append(Line2D([0], [0], marker='o', markerfacecolor='black', 
									color='w', label='init cond'))
legend_objs.append(Line2D([0], [0], color='b', label='ground truth'))
legend_objs.append(Line2D([0], [0], color='darkred', label='pred mean'))
legend_objs.append(mpatches.Patch(color='pink', label='pred dist'))

plt.legend(handles=legend_objs)
plt.show()
