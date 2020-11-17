import os
import sys
import wandb
import subprocess
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
        pred_zt = torch.cat(pred_zt, dim=0)
        return pred_zt

class SWAGTrainer():

    def __init__(self, **kwargs):
        self._trainer = make_det_trainer(**kwargs)
        self.model = SWAGModel(self._trainer.model)

    def train(self, num_epochs):
        self._trainer.train(num_epochs)

class DeepEnsembleModel(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, z, t, n_samples=10):
        pass

class DeepEnsembleTrainer():

    def __init__(self, ensemble_size=2, **kwargs):
        config = {
            "name": "DeepEnsemble",
            "project": "physics-uncertainty-exps",
            "method": "grid",
            "parameters": {
                "net_seed": {
                    "values": list(range(ensemble_size))
                },
                "num_bodies": {
                    "value": kwargs.get("num_bodies")
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
        self.sweep_id = wandb.sweep(config)

        self.model = DeepEnsembleModel()

    def train(self, num_epochs):
        _submit = lambda x: subprocess.call(["sbatch", "--wait", "configs/sweep.sh"])

        os.environ['WANDB_SWEEP_ID'] = self.sweep_id
        with mp.Pool(5) as p:
            p.map(_submit, range(self.ensemble_size))

        save_dir = os.path.join(os.environ["LOGDIR"], os.environ['WANDB_SWEEP_ID'])
        model_paths = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

        self.ensemble = []
        for model_path in model_paths:
            model = torch.load(model_path)
            self.ensemble.append(model)

        import pprint
        pprint.pprint(self.ensemble)


def make_trainer(uq_type=None, **kwargs):
    if uq_type == 'swag':
        kwargs.pop('num_bodies', None)
        return SWAGTrainer(**kwargs)
    elif uq_type == 'deep-ensemble':
        return DeepEnsembleTrainer(**kwargs)
    else:
        kwargs.pop('num_bodies', None)
        return make_det_trainer(**kwargs)