import os
import sys
import wandb
import tempfile
import altair as alt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch

from src.train.ensemble_trainer import make_trainer
from src.models import HNN, CHNN, AleatoricCHNN, CNFCHNN
from src.systems.chain_pendulum import ChainPendulum
from src.systems.coupled_pendulum import CoupledPendulum
from src.systems.magnet_pendulum import MagnetPendulum
from src.datasets import get_chaotic_eval_dataset

def generate_trace_chart(ts, true_zt, true_zt_chaos, pred_zt, body_idx, dof_idx):
  true_chart = alt.Chart(pd.DataFrame({
    't': ts.cpu().numpy(),
    'y': true_zt.cpu().numpy(),
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

def generate_err_chart(ts, true_zt, true_zt_chaos, pred_zt):
  bold_opacity = 0.8
  dotted_opacity = 0.4
  pred_color = 'red'
  chaos_color = 'blue'

  chaos_mrse = compute_rel_error(true_zt, true_zt_chaos.mean(dim=0)).squeeze(0).log().cpu().numpy()
  pred_mrse = compute_rel_error(true_zt, pred_zt.mean(dim=0)).squeeze(0).log().cpu().numpy()

  # pred_mean = swag_mrse.mean(0).cpu().numpy()
  # pred_std = swag_mrse.std(0).cpu().numpy()

  pred_chart = alt.Chart(pd.DataFrame({
      't': ts.cpu().numpy(),
      'y': pred_mrse,
      'chaos': chaos_mrse,
      # 'y_hi': pred_mean + 2. * pred_std,
      # 'y_lo': np.clip(swag_mean - 2. * swag_std, 0.0, np.inf),
  })).mark_line(color=pred_color,opacity=bold_opacity).encode(x='t', y='y')
  chaos_chart = pred_chart.mark_line(color=chaos_color, opacity=dotted_opacity, strokeDash=[2,2]).encode(x='t', y='chaos')
  chart = pred_chart + chaos_chart

  return chart

def plot_ts(ts, z0_orig, true_zt, true_zt_chaos, pred_zt):
  alt.data_transformers.disable_max_rows()

  ## Only plot 5 trajectories for viz.
  nsamps = min(z0_orig.size(0), 5)
  nbodies = z0_orig.size(-2)
  for i in range(nsamps):
    trace_chart = None

    for b in tqdm(range(nbodies)):
      trace_chart_x = generate_trace_chart(ts, true_zt[i, :, 0, b, 0],
                                     true_zt_chaos[:, i, :, 0, b, 0],
                                     pred_zt[:, i, :, 0, b, 0], b, 0)
      trace_chart_y = generate_trace_chart(ts, true_zt[i, :, 0, b, 1],
                                     true_zt_chaos[:, i, :, 0, b, 1],
                                     pred_zt[:, i, :, 0, b, 1], b, 1)
      if trace_chart is None:
        trace_chart = (trace_chart_x & trace_chart_y)
      else:
        trace_chart = trace_chart & (trace_chart_x & trace_chart_y)

    wandb.log({f'trace:i={i};nb={nbodies}': wandb.Html(trace_chart.to_html())})

    err_chart = generate_err_chart(ts, true_zt[i], true_zt_chaos[i], pred_zt[:,i])

    wandb.log({f'err:i={i};nb={nbodies}': wandb.Html(err_chart.to_html())})

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

def compute_mrse(ref, pred):
    '''
    N is the number of initial conditions.
    M is the number of samples in prediction
    The first dimension "2" corresponds to position + velocity.
    B is the number of bodies.
    The last dimension "2" corresponds to xy.

    Arguments:
    ref: N x T x 2 x B x 2
    pred: M x N x T x 2 x B x 2
    '''
    delta_z = ref - pred  # M x N x T x 2 x B x 2
    all_err = delta_z.pow(2).sum(dim=-1).sum(dim=-1).sum(dim=-1).sqrt()  # M x N x T

    return all_err

def compute_rel_error(ref, pred):
  '''
  N is the number of initial conditions.
  M is the number of samples in prediction
  The first dimension "2" corresponds to position + velocity.
  B is the number of bodies.
  The last dimension "2" corresponds to xy.

  Arguments:
    ref: N x T x 2 x B x 2
    pred: M x N x T x 2 x B x 2
  '''
  delta_z = ref.unsqueeze(0) - pred  # M x N x T x 2 x B x 2
  all_err = delta_z.pow(2).sum(dim=-1).sum(dim=-1).sum(dim=-1).sqrt()  # M x N x T

  sum_z = ref.unsqueeze(0) + pred  # M x N x T x 2 x B x 2
  pred_rel_err = all_err / sum_z.pow(2).sum(dim=-1).sum(dim=-1).sum(dim=-1).sqrt()  # M x N x T

  return pred_rel_err

def compute_likelihood(ref, pred):
  '''
  Likelihood of the reference under Gaussian estimated
  by the samples, factored over time.
  Arguments:
    ref: N x T x 2 x B x 2
    pred: M x N x T x 2 x B x 2
  '''
  batch_shape = pred.shape[:3]

  pred_mu = pred.view(*batch_shape, -1).mean(dim=0)
  pred_std = pred.view(*batch_shape, -1).std(dim=0) + 1e-6
  pred_std[torch.isnan(pred_std)] = 1e-6
  pred_dist = torch.distributions.MultivariateNormal(pred_mu, pred_std.diag_embed())

  log_prob = pred_dist.log_prob(ref.view(*ref.shape[:2], -1))  ## N x T
  return log_prob

def compute_geom_mean(ts, loss):
  n = ts.size(-1)
  loss = loss[:,:,:n//3]
  ts = ts[:n//3]
  t_range = ts.max() - ts.min()
  return torch.trapz((loss + 1e-8).log(), ts).div(t_range).exp()

def compute_metrics(ts, true_zt, true_zt_chaos, pred_zt):
  # calibration_score = calibration_metric(true_zt, pred_zt)
  # kl_score = kl_metric(true_zt_chaos, pred_zt)

  chaos_rel_err = compute_rel_error(true_zt, true_zt_chaos.mean(0))
  pred_rel_err = compute_rel_error(true_zt, pred_zt.mean(0))

  chaos_geom_mean = compute_geom_mean(ts, chaos_rel_err)
  pred_geom_mean = compute_geom_mean(ts, pred_rel_err)

  chaos_likelihood = compute_likelihood(true_zt, true_zt_chaos)
  pred_likelihood = compute_likelihood(true_zt, pred_zt)

  wandb.log({
    'chaos_lik': chaos_likelihood.sum(-1).mean(),
    'chaos_lik_std': chaos_likelihood.sum(-1).std(),
    'chaos_geom_mean': chaos_geom_mean.mean(),
    'chaos_geom_mean_std': chaos_geom_mean.std(),
    'pred_lik': pred_likelihood.sum(-1).mean(),
    'pred_lik_std': pred_likelihood.sum(-1).std(),
    'pred_geom_mean': pred_geom_mean.mean(),
    'pred_geom_mean_std': pred_geom_mean.std(),
  })

def evaluate_uq(uq_type, body, model, eps_scale=1e-2, n_samples=5, device=None):
  n_init = 25
  evald = get_chaotic_eval_dataset(body, n_init=n_init, n_samples=n_samples, eps_scale=eps_scale)

  model = model.to(device)

  ts = evald['ts'].to(device)
  z0_orig = evald['z0_orig'].to(device)
  true_zt = evald['true_zt'].to(device)
  true_zt_chaos = evald['true_zt_chaos'].to(device)
  true_zt_chaos = true_zt_chaos[:n_samples]

  z0 = torch.cat([z0_orig.unsqueeze(0), true_zt_chaos[:,:,0]], 0)
  z0 = z0.reshape((n_samples + 1)*n_init, *z0.shape[2:])

  if uq_type == 'output-uncertainty':
    pred_zt, var_zt = model(z0, ts, n_samples=n_samples)
    var_zt = var_zt.reshape((n_samples + 1), n_init, *var_zt.shape[1:])[0]
  else:
    pred_zt = model(z0, ts, n_samples=n_samples)

  if uq_type == 'swag' or uq_type == 'deep-ensemble':
    pred_zt = pred_zt.reshape(n_samples, (n_samples + 1), n_init, *pred_zt.shape[2:])
    pred_zt, pred_zt_chaos = pred_zt[:,0], pred_zt[:,1:].mean(0)
  else:
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

def evaluate_training(trainer, body, model_num=0):
  train_loader = trainer.dataloaders["train"]
  test_loader = trainer.dataloaders["test"]

  train_loss = 0
  for mb in train_loader:
    with torch.no_grad():
      train_loss += trainer.loss(mb).item()

  test_loss = 0
  for mb in test_loader:
    with torch.no_grad():
      test_loss += trainer.loss(mb).item()

  evald = get_chaotic_eval_dataset(body, n_init=25, n_samples=5)

  device = "cuda" if torch.cuda.is_available() else None 
  model = trainer.model.to(device)
  ts = evald['ts'].to(device)
  z0_orig = evald['z0_orig'].to(device)
  true_zt = evald['true_zt'].to(device)

  with torch.no_grad():
    pred_zt = trainer.model.integrate(z0_orig, ts)

  pred_rel_err = compute_rel_error(true_zt, pred_zt)
  pred_geom_mean = compute_geom_mean(ts, pred_rel_err).mean().item()

  eval_dict = {
    'train_loss_{}'.format(model_num): train_loss,
    'test_loss_{}'.format(model_num): test_loss,
    'rollout_geom_mean_{}'.format(model_num): pred_geom_mean,
  }

  return eval_dict


def main(**cfg):
  wandb.init(config=cfg)

  run_eval = cfg.pop('run_eval', True)
  eps_scale = cfg.pop('eps_scale', 1e-2)

  cfg['device'] = cfg.get('device', None)
  if cfg['device'] is None:
    cfg['device'] = 'cuda:0' if torch.cuda.is_available() else None

  body = ChainPendulum(cfg.get('num_bodies', 3))
  #CoupledPendulum(cfg.get('num_bodies', 3))
  #MagnetPendulum(cfg.get('num_bodies', 3))
  #ChainPendulum(cfg.get('num_bodies', 3))
  cfg['uq_type'] = cfg.get('uq_type', None)
  if cfg['uq_type'] == 'output-uncertainty':
    network = AleatoricCHNN
  elif cfg['uq_type'] == 'cnf':
    network = CNFCHNN
  else:
    network = CHNN

  trainer = make_trainer(**cfg,
      network=network, body=body, trainer_config=dict(log_dir=wandb.run.dir))

  # map_backwards(cfg['uq_type'], body, trainer.model, eps_scale=eps_scale, device=cfg['device'])

  # load_fn = "/misc/vlgscratch4/WilsonGroup/ngruver/src/physics-uncertainty/wandb/run-20210114_122958-1axj7lqk/files/model.pt"
  # #"/misc/vlgscratch4/WilsonGroup/ngruver/src/physics-uncertainty/wandb/offline-run-20210108_125717-1w6bfiaj/files/model.pt"
  # state_dict = torch.load(load_fn)
  # trainer.model.load_state_dict(state_dict)

  root_dir = os.path.join(os.environ["LOGDIR"], "chnn_ensemble_diversity")
  if not os.path.exists(root_dir):
    os.makedirs(root_dir)
  models_dir = tempfile.mkdtemp(dir=root_dir)

  num_epochs = cfg.get('num_epochs', 10)
  for i in range(num_epochs):
    trainer.train(1)

    save_dir = os.path.join(models_dir, 'model_{}.pt'.format(i))
    torch.save(trainer.model.state_dict(), save_dir)

    eval_dict = {}
    for idx, _trainer in enumerate(trainer._trainers):
      eval_dict.update(evaluate_training(_trainer, body, idx))

    wandb.log(eval_dict)

  save_dir = os.path.join(wandb.run.dir, 'model.pt')
  print("************* SAVE DIR: {} *************".format(save_dir))

  torch.save(trainer.model.state_dict(), save_dir)
  wandb.save(save_dir)

  wandb.log({"models_dir": models_dir})

  cfg['uq_type'] = cfg.get('uq_type', None)
  if run_eval:# and (cfg['uq_type'] is not None):
    evaluate_uq(cfg['uq_type'], body, trainer.model, eps_scale=eps_scale, device=cfg['device'])
    # map_backwards(cfg['uq_type'], body, trainer.model, eps_scale=eps_scale, device=cfg['device'])

if __name__ == "__main__":
  # os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')

  from fire import Fire
  Fire(main)