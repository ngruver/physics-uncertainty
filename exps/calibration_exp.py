import os
import sys
import json
import copy
import pickle
import argparse
import altair as alt
import pandas as pd

import wandb
import torch
import numpy as np
from pprint import pprint

if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))

from oil.tuning.args import argupdated_config
from src.train.det_trainer import make_trainer
from src.models import CHNN
from src.systems.chain_pendulum import ChainPendulum
from src.systems.rigid_body import project_onto_constraints
		
parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', type=str,
					default='/misc/vlgscratch4/WilsonGroup/ngruver/logs/physics-uncertainty',
                   	help='directory containing configs and slurm scripts')
parser.add_argument('--exp_name', type=str, default='default',
                   	help='experiment name used for tracking')

def load_config(exp_dir, exp_name):
	config_fn = "{}.json".format(os.path.join(exp_dir, exp_name))
	
	with open(config_fn) as fd:
		config = json.load(fd)

	trainer_config = copy.deepcopy(make_trainer.__kwdefaults__)

	trainer_config["network"] = CHNN
	trainer_config["device"] = 'cuda:0' if torch.cuda.is_available() else None
	trainer_config["body"] = ChainPendulum(config["num_bodies"])
	trainer_config["C"] = config["segment_len"]
	trainer_config["num_epochs"] = config["num_epochs"]
	trainer_config["root_dir"] = exp_dir

	return config, trainer_config

def gen_compare_plot(chart_fn, config, trainer, posterior_samples=10):
	z0_orig = config["body"].sample_initial_conditions(1)

	N = 10
	eps = 2. * torch.rand_like(z0_orig.expand(N, -1, -1, -1)) - 1.
	z0 = project_onto_constraints(config["body"].body_graph,
	                              z0_orig.expand(N, -1, -1, -1) + 0.1 * eps, tol=1e-5)
	ts = torch.arange(0., N, config["body"].dt, device=z0_orig.device, dtype=z0_orig.dtype)

	true_zt = config["body"].integrate(z0_orig, ts, method='rk4')
	true_zt_chaos = config["body"].integrate(z0, ts, method='rk4')

	num_samples = posterior_samples
	pred_zt = []
	for _ in range(num_samples):
		trainer.model.sample()
		model = trainer.model
		if torch.cuda.is_available():
			model = model.cuda()
			z0_orig = z0_orig.cuda()
			ts = ts.cuda()
		with torch.no_grad():
			zt_pred = trainer.model.integrate_swag(z0_orig, ts, method='rk4')
		pred_zt.append(zt_pred)
	pred_zt = torch.cat(pred_zt, dim=0)

	alt.data_transformers.disable_max_rows()

	body_idx, dof_idx = 1, 1

	true_chart = alt.Chart(pd.DataFrame({
	    't': ts.cpu().numpy(),
	    'y': true_zt[..., 0, body_idx, dof_idx].mean(dim=0).cpu().numpy(),
	})).mark_line(color='black',strokeDash=[5,5]).encode(x='t:Q', y=alt.Y('y:Q'))

	pred_zt_mu = pred_zt[..., 0, body_idx, dof_idx].mean(dim=0)
	pred_zt_std = pred_zt[..., 0, body_idx, dof_idx].std(dim=0)
	pred_chart = alt.Chart(pd.DataFrame({
	    't': ts.cpu().numpy(),
	    'y': pred_zt_mu.cpu().numpy(),
	    'y_lo': (pred_zt_mu - 2. * pred_zt_std).cpu().numpy(),
	    'y_hi': (pred_zt_mu + 2. * pred_zt_std).cpu().numpy(),
	})).mark_line(color='red',opacity=0.5).encode(x='t:Q', y='y:Q')
	pred_err_chart = pred_chart.mark_area(opacity=0.1,color='red').encode(y='y_lo', y2='y_hi')

	true_zt_chaos_mu = true_zt_chaos[..., 0, body_idx, dof_idx].mean(dim=0)
	true_zt_chaos_std = true_zt_chaos[..., 0, body_idx, dof_idx].std(dim=0)
	chaos_chart = alt.Chart(pd.DataFrame({
	    't': ts.cpu().numpy(),
	    'y': true_zt_chaos_mu.cpu().numpy(),
	    'y_lo': (true_zt_chaos_mu - 2. * true_zt_chaos_std).cpu().numpy(),
	    'y_hi': (true_zt_chaos_mu + 2. * true_zt_chaos_std).cpu().numpy(),
	})).mark_line(color='blue',opacity=0.5).encode(x='t:Q', y='y:Q')
	chaos_err_chart = chaos_chart.mark_area(opacity=0.1,color='blue').encode(y='y_lo', y2='y_hi')

	chart = (chaos_err_chart + chaos_chart + true_chart | pred_err_chart + pred_chart + true_chart)
	chart.properties(title=f'Mass = {body_idx}, DoF = {dof_idx}; Chaos v/s Predictions')
	chart.save(chart_fn)

def main():
	args = parser.parse_args()

	config, trainer_config = load_config(args.exp_dir, args.exp_name)

	wandb.init(project="physics-uncertainty", 
			   name=config["exp_name"], config=config,
			   dir=os.path.join(config["exp_dir"],"wandb"))

	trainer = make_trainer(**trainer_config)
	
	wandb.watch(trainer.model)

	trainer.train(trainer_config['num_epochs'])

	chart_fn = "{}.html".format(os.path.join(args.exp_dir, args.exp_name))
	gen_compare_plot(chart_fn, trainer_config, trainer)#, config["posterior_samples"])
	wandb.log({"chart": wandb.Html(open(chart_fn))})

if __name__ == "__main__":
	main()