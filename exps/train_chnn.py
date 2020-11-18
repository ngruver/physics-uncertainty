import os
import sys
import wandb
import altair as alt
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch

from src.train.ensemble_trainer import make_trainer
from src.models import CHNN
from src.systems.chain_pendulum import ChainPendulum
from src.datasets import get_chaotic_eval_dataset

def generate_chart(ts, true_zt, true_zt_chaos, pred_zt, body_idx, dof_idx):
	true_chart = alt.Chart(pd.DataFrame({
	  't': ts.cpu().numpy(),
	  'y': true_zt.mean(dim=0).cpu().numpy(),
	})).mark_line(color='black',strokeDash=[5,5]).encode(x='t:Q', y=alt.Y('y:Q'))

	pred_zt_mu = pred_zt.mean(dim=0)
	pred_zt_std = pred_zt.std(dim=0)
	pred_chart = alt.Chart(pd.DataFrame({
	  't': ts.cpu().numpy(),
	  'y': pred_zt_mu.cpu().numpy(),
	  'y_lo': (pred_zt_mu - 2. * pred_zt_std).cpu().numpy(),
	  'y_hi': (pred_zt_mu + 2. * pred_zt_std).cpu().numpy(),
	})).mark_line(color='red',opacity=0.5).encode(x='t:Q', y='y:Q')
	pred_err_chart = pred_chart.mark_area(opacity=0.1,color='red').encode(y='y_lo', y2='y_hi')

	true_zt_chaos_mu = true_zt_chaos.mean(dim=0)
	true_zt_chaos_std = true_zt_chaos.std(dim=0)
	chaos_chart = alt.Chart(pd.DataFrame({
	  't': ts.cpu().numpy(),
	  'y': true_zt_chaos_mu.cpu().numpy(),
	  'y_lo': (true_zt_chaos_mu - 2. * true_zt_chaos_std).cpu().numpy(),
	  'y_hi': (true_zt_chaos_mu + 2. * true_zt_chaos_std).cpu().numpy(),
	})).mark_line(color='blue',opacity=0.5).encode(x='t:Q', y='y:Q')
	chaos_err_chart = chaos_chart.mark_area(opacity=0.1,color='blue').encode(y='y_lo', y2='y_hi')

	return (chaos_err_chart + chaos_chart + true_chart | pred_err_chart + pred_chart + true_chart).properties(title=f'Mass = {body_idx}, DoF = {dof_idx}; Chaos v/s Predictions')

def plot_ts(ts, z0_orig, true_zt, true_zt_chaos, pred_zt):
	alt.data_transformers.disable_max_rows()

	for i in range(z0_orig.size(0)):
		for b in tqdm(range(z0_orig.size(-2))):
	  		for dof in tqdm(range(z0_orig.size(-1)), leave=False):
	  			chart = generate_chart(ts, true_zt[i, :, 0, b, dof],
																 true_zt_chaos[:, i, :, 0, b, dof],
																 pred_zt[:, i, :, 0, b, dof], b, dof)
	  			wandb.log({f'i={i};b={b};dof={dof}': wandb.Html(chart.to_html())})

def calibration_metric(true_zt, pred_zt):
	pred_zt_mu = pred_zt.mean(dim=0)
	pred_zt_std = pred_zt.std(dim=0)	
	low = (pred_zt_mu - 2. * pred_zt_std).cpu().numpy()
	high = (pred_zt_mu + 2. * pred_zt_std).cpu().numpy()
	true_zt = true_zt.cpu().numpy()

	perc_in_interval = np.mean(np.logical_and(np.greater(true_zt, low), np.less(true_zt, high)))
	return perc_in_interval


def kl_metric(true_zt_chaos, pred_zt):
	true_zt_chaos_mu = true_zt_chaos.mean(dim=0)
	true_zt_chaos_std = true_zt_chaos.std(dim=0)	

	pred_zt_mu = pred_zt.mean(dim=0)
	pred_zt_std = pred_zt.std(dim=0)	

	kl_gaussian = ((pred_zt_std / (true_zt_chaos_std + 1e-7)) + 1e-7).log() + \
								(true_zt_chaos_std.pow(2) + (true_zt_chaos_mu - pred_zt_mu).pow(2)) / (2 * pred_zt_std.pow(2) + 1e-7) - 0.5

	return kl_gaussian.mean().item()

def compute_metrics(true_zt, true_zt_chaos, pred_zt):
	calibration_score = calibration_metric(true_zt, pred_zt)
	kl_score = kl_metric(true_zt_chaos, pred_zt)

	print(calibration_score)
	print(kl_score)
	#LOG TO WANDB HERE

def evaluate_uq(body, model, n_samples=10, device=None):
	evald = get_chaotic_eval_dataset(body, n_samples)

	model = model.to(device)

	ts = evald['ts'].to(device)
	z0_orig = evald['z0_orig'].to(device)
	true_zt = evald['true_zt'].to(device)
	true_zt_chaos = evald['true_zt_chaos'].to(device)

	true_zt_chaos = true_zt_chaos[:n_samples]

	pred_zt = model(z0_orig, ts, n_samples=n_samples)

	plot_ts(ts, z0_orig, true_zt, true_zt_chaos, pred_zt)

	compute_metrics(true_zt, true_zt_chaos, pred_zt)


def main(**cfg):
  wandb.init(config=cfg)

  cfg['device'] = cfg.get('device', None)
  if cfg['device'] is None:
  	cfg['device'] = 'cuda:0' if torch.cuda.is_available() else None

  body = ChainPendulum(cfg.get('num_bodies', 3))
  trainer = make_trainer(**cfg,
	network=CHNN, body=body, trainer_config=dict(log_dir=wandb.run.dir))

  trainer.train(cfg.get('num_epochs', 10))

  cfg['uq_type'] = cfg.get('uq_type', None)
  if cfg['uq_type'] is not None:
  	evaluate_uq(body, trainer.model, device=cfg['device'])

  save_dir = os.path.join(wandb.run.dir, 'model.pt')
  torch.save(trainer.model.state_dict(), save_dir)
  wandb.save(save_dir)

if __name__ == "__main__":
  os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')

  from fire import Fire
  Fire(main)