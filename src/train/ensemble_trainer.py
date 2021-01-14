import os
import re
import sys
import copy
import glob
import wandb
import subprocess
import numpy as np
import pathos.multiprocessing as mp

import torch
import torch.nn as nn

from oil.model_trainers import Trainer

from .det_trainer import make_trainer as make_det_trainer

class SWAGModel(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def forward(self, z, t, n_samples=10):
        pred_zt = []
        for _ in range(n_samples):
            self.model.sample()
            with torch.no_grad():
                zt_pred = self.model.integrate_swag(z, t, method='rk4')
            pred_zt.append(zt_pred)
        pred_zt = torch.stack(pred_zt, dim=0)
        return pred_zt

class SWAGTrainer():

    def __init__(self, swag_epochs=20, **kwargs):
        self._trainer = make_det_trainer(**kwargs)
        self.model = SWAGModel(self._trainer.model)
        self.swag_epochs = swag_epochs

    def train(self, num_epochs):
        self._trainer.train(num_epochs)
        self._trainer.collect = True
        self._trainer.train(self.swag_epochs)

class DeepEnsembleModel(nn.Module):
    
    def __init__(self, model, ensemble, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.ensemble = ensemble

    def forward(self, z, t, n_samples=10):
        pred_zt = []
        for state_dict in self.ensemble:
            self.model.load_state_dict(state_dict)
            self.model.eval()
            with torch.no_grad():
                zt_pred = self.model.integrate(z, t, method='rk4')
            pred_zt.append(zt_pred)
        pred_zt = torch.stack(pred_zt, dim=0)
        return pred_zt

class DeepEnsembleTrainer():

    def __init__(self, ensemble_size=5, num_bodies=2, **kwargs):
        config = {
            "name": "DeepEnsemble",
            "project": "physics-uncertainty-exps",
            "method": "grid",
            "parameters": {
                "net_seed": {
                    "values": list(range(ensemble_size))
                },
                "num_bodies": {
                    "value": num_bodies
                },
                "lr": {
                    "value": kwargs.get("lr")
                },
                "tau": {
                    "value": kwargs.get("tau")
                },
                "C": {
                    "value": kwargs.get("C")
                },
                "num_epochs": {
                    "value": kwargs.get("num_epochs")
                }
            },
            "program": "exps/train_chnn.py"
        }

        os.environ['WANDB_PROJECT'] = "physics-uncertainty-exps"
        self.ensemble_size = ensemble_size
        # self.sweep_id = wandb.sweep(config)

        self.ensemble = nn.ModuleList([])
        self._trainer = make_det_trainer(**kwargs)
        self.model = DeepEnsembleModel(self._trainer.model, self.ensemble)

        self._trainers = [make_det_trainer(**kwargs) for _ in range(self.ensemble_size)]
        # self.ensemble = [trainer.model.state_dict() for trainer in self._trainers]

    def train(self, num_epochs):
        # _submit = lambda x: subprocess.call(["sbatch", "--wait", "configs/sweep.sh"])

        # os.environ['WANDB_SWEEP_ID'] = self.sweep_id
        # with mp.Pool(5) as p:
        #     p.map(_submit, range(self.ensemble_size))

        # wandb_dir = os.path.join(os.environ["LOGDIR"], "wandb")
        #sweep_dir = os.path.join(wandb_dir, "sweep-{}".format(os.environ['WANDB_SWEEP_ID']))
        # wandb_dir = "/misc/vlgscratch4/WilsonGroup/ngruver/logs/wandb"
        # sweep_dir = "/misc/vlgscratch4/WilsonGroup/ngruver/logs/wandb/sweep-ovhpc8rp"
        # run_names = [re.match("config-(\w+).yaml",f).groups()[0] for f in os.listdir(sweep_dir)]
        # model_dirs = [glob.glob(os.path.join(wandb_dir,"*-{}".format(n)))[0] for n in run_names]
        # model_paths = [os.path.join(d, "files", "model.pt") for d in model_dirs]

        # for model_path in model_paths:
        #     model = torch.load(model_path)
        #     model = {k.partition('model.')[2]: model[k] for k in model}
        #     self.ensemble.append(model)

        for idx, trainer in enumerate(self._trainers):
            trainer.train(num_epochs)
            self.ensemble.append(copy.deepcopy(trainer.model))

class DeepEnsembleModel(nn.Module):
    
    def __init__(self, model, ensemble, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.ensemble = ensemble

    def forward(self, z, t, n_samples=10):
        pred_zt = []
        for model in self.ensemble:
            #self.model.load_state_dict(state_dict)
            #self.model.eval()
            with torch.no_grad():
                zt_pred = model.integrate(z, t, method='rk4')
            pred_zt.append(zt_pred)
        pred_zt = torch.stack(pred_zt, dim=0)
        return pred_zt

class AleotoricWrapper(nn.Module):

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def forward(self, z, t, n_samples=10):
        self.model.eval()
        with torch.no_grad():
            zt_pred = self.model.integrate(z, t, method='rk4')
            var = self.model.get_covariance(zt_pred, t).reshape(*zt_pred.shape)
        return zt_pred, var

class AleotoricTrainer():

    def __init__(self, **kwargs):
        self._trainer = make_det_trainer(**kwargs)
        self._trainer.prob_loss = True
        self.model = AleotoricWrapper(self._trainer.model)

    def train(self, num_epochs):
        self._trainer.train(num_epochs) 

class DeterministicWrapper(nn.Module):

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def forward(self, z, t, n_samples=10):
        self.model.eval()
        with torch.no_grad():
            zt_pred = self.model.integrate(z, t, method='rk4')
        return zt_pred

class DeterministicTrainer():

    def __init__(self, **kwargs):
        self._trainer = make_det_trainer(**kwargs)
        self.model = DeterministicWrapper(self._trainer.model)

    def train(self, num_epochs):
        self._trainer.train(num_epochs)

def make_trainer(uq_type=None, **kwargs):
    if uq_type == 'swag':
        kwargs.pop('num_bodies', None)
        return SWAGTrainer(**kwargs)
    elif uq_type == 'deep-ensemble':
        return DeepEnsembleTrainer(**kwargs)
    elif uq_type == 'output-uncertainty':
        kwargs.pop('num_bodies', None)
        return AleotoricTrainer(**kwargs)
    elif uq_type == 'cnf':
        kwargs.pop('num_bodies', None)
        return DeterministicTrainer(**kwargs)
    elif (uq_type == 'det') or (uq_type is None):
        kwargs.pop('num_bodies', None)
        return DeterministicTrainer(**kwargs)
    else:
        raise NotImplementedError