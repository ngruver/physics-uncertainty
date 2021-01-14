import os
import sys
import wandb
import altair as alt
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch

from src.train.ensemble_trainer import make_trainer
from src.models import HNN, CHNN, AleatoricCHNN, CNFCHNN
from src.systems.chain_pendulum import ChainPendulum
from src.systems.coupled_pendulum import CoupledPendulum
from src.systems.magnet_pendulum import MagnetPendulum
from src.datasets import get_chaotic_eval_dataset

def evaluate_uq(uq_type, body, model, eps_scale=1e-2, n_samples=5, device=None):
  n_init = 25
  evald = get_chaotic_eval_dataset(body, n_init=n_init, n_samples=n_samples, eps_scale=eps_scale)

  model = model.to(device)

  ts = evald['ts'].to(device)
  z0_orig = evald['z0_orig'].to(device)
  true_zt = evald['true_zt'].to(device)
  true_zt_chaos = evald['true_zt_chaos'].to(device)
  true_zt_chaos = true_zt_chaos[:n_samples]

  z0 = torch.cat([z0_orig.unsqueeze(0), true_zt_chaos[:,:,0,:,:,:]], 0)
  z0 = z0.reshape((n_samples + 1)*n_init, *z0.shape[2:])

  z0 = body.global2bodyCoords(z0.cpu().double()).float().to(device)

  if uq_type == 'output-uncertainty':
    pred_zt, var_zt = model(z0, ts, n_samples=n_samples)
    var_zt = var_zt.reshape((n_samples + 1), n_init, *var_zt.shape[1:])[0]
  else:
    pred_zt = model(z0, ts, n_samples=n_samples)

  # print(pred_zt.shape)
  # bs, Nlong, *rest = pred_zt.shape
  # flat_pred = body.body2globalCoords(pred_zt.reshape(bs * Nlong, *rest))
  # pred_zt = flat_pred.reshape(bs, Nlong, *flat_pred.shape[1:])

  if uq_type == 'swag' or uq_type == 'deep-ensemble':
    M, bs, Nlong, *rest = pred_zt.shape
    flat_pred = body.body2globalCoords(pred_zt.reshape(M * bs * Nlong, *rest))
    pred_zt = flat_pred.reshape(M, bs, Nlong, *flat_pred.shape[1:])

    pred_zt = pred_zt.reshape(n_samples, (n_samples + 1), n_init, *pred_zt.shape[2:])
    pred_zt, pred_zt_chaos = pred_zt[:,0], pred_zt[:,1:].mean(0)
  else:
    bs, Nlong, *rest = pred_zt.shape
    flat_pred = body.body2globalCoords(pred_zt.reshape(bs * Nlong, *rest))
    pred_zt = flat_pred.reshape(bs, Nlong, *flat_pred.shape[1:])

    pred_zt = pred_zt.reshape((n_samples + 1), n_init, *pred_zt.shape[1:])
    pred_zt, pred_zt_chaos = pred_zt[:1], pred_zt[1:]

  ## NOTE: Simply dump all data so that we can do offline plotting.
  data_dump = dict(
    ts=ts.cpu(),
    z0_orig=z0_orig.cpu(),
    true_zt=true_zt.cpu(),
    true_zt_chaos=true_zt_chaos.cpu(),
    pred_zt=pred_zt.cpu(),
    pred_zt_chaos=pred_zt_chaos.cpu()
  )
  if uq_type == 'output-uncertainty':
    data_dump['var_zt'] = var_zt.cpu()

  data_dump_file = os.path.join(wandb.run.dir, 'data.pt')
  print("Dumping to {}...".format(data_dump_file))
  torch.save(data_dump, data_dump_file)
  wandb.save(data_dump_file)

  # plot_ts(ts, z0_orig, true_zt, true_zt_chaos, pred_zt)

  # compute_metrics(ts, true_zt, true_zt_chaos, pred_zt)

def main(**cfg):
  print("CUDA AVAILABLE: {}".format(torch.cuda.is_available()))

  wandb.init(config=cfg)

  run_eval = cfg.pop('run_eval', True)
  eps_scale = cfg.pop('eps_scale', 1e-2)

  cfg['device'] = cfg.get('device', None)
  if cfg['device'] is None:
    cfg['device'] = 'cuda:0' if torch.cuda.is_available() else None

  body = ChainPendulum(cfg.get('num_bodies', 3))
  cfg['uq_type'] = cfg.get('uq_type', None)
  if False:
  	pass
  else:
    network = HNN

  trainer = make_trainer(**cfg,
      network=network, body=body, trainer_config=dict(log_dir=wandb.run.dir))

  trainer.train(cfg.get('num_epochs', 10))
  rel_errs, zt_pred = trainer._trainer.test_rollouts(True)

  save_dir = os.path.join(wandb.run.dir, 'model.pt')
  torch.save(trainer.model.state_dict(), save_dir)
  wandb.save(save_dir)

  cfg['uq_type'] = cfg.get('uq_type', None)
  
  evaluate_uq(cfg['uq_type'], body, trainer.model, eps_scale=eps_scale, device=cfg['device'])

if __name__ == "__main__":
  # os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')

  from fire import Fire
  Fire(main)