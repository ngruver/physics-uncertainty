import os
import wandb
import altair as alt
import pandas as pd
from tqdm.auto import tqdm
import torch

from src.train.det_trainer import make_trainer
from src.models import CHNN
from src.systems.chain_pendulum import ChainPendulum
from src.systems.rigid_body import project_onto_constraints


def plot_ts(body, model, n_samples=10, device=None):
  alt.data_transformers.disable_max_rows()

  z0_orig = body.sample_initial_conditions(1)

  eps = 2. * torch.rand_like(z0_orig.expand(10, -1, -1, -1)) - 1.
  z0 = project_onto_constraints(body.body_graph,
                                z0_orig.expand(10, -1, -1, -1) + 0.1 * eps, tol=1e-5)
  ts = torch.arange(0., 10.0, body.dt, device=z0_orig.device, dtype=z0_orig.dtype)

  true_zt = body.integrate(z0_orig, ts, method='rk4')
  true_zt_chaos = body.integrate(z0, ts, method='rk4')

  pred_zt = []
  for _ in range(n_samples):
      model.sample()
      model = model.to(device)
      z0_orig = z0_orig.to(device)
      ts = ts.to(device)
      
      with torch.no_grad():
        zt_pred = model.integrate_swag(z0_orig, ts, method='rk4')
      pred_zt.append(zt_pred)
  pred_zt = torch.cat(pred_zt, dim=0)

  body_idx, dof_idx = 1, 1

  def generate_chart(body_idx, dof_idx):
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

    return (chaos_err_chart + chaos_chart + true_chart | pred_err_chart + pred_chart + true_chart).properties(title=f'Mass = {body_idx}, DoF = {dof_idx}; Chaos v/s Predictions')
  
  for b in tqdm(range(body.n)):
    for dof in tqdm(range(2), leave=False):
      chart = generate_chart(b, dof)
      wandb.log({f'b={b};dof={dof}': wandb.Html(chart.to_html())})


def main(num_bodies=2, **cfg):
  wandb.init(project=os.environ['WANDB_PROJECT'], config=cfg)

  cfg['device'] = cfg.get('device', None)
  if cfg['device'] is None:
    cfg['device'] = 'cuda:0' if torch.cuda.is_available() else None

  body = ChainPendulum(num_bodies)
  trainer = make_trainer(**cfg,
    network=CHNN, body=body, trainer_config=dict(log_dir=wandb.run.dir))

  trainer.train(cfg.get('num_epochs', 10))

  plot_ts(body, trainer.model, device=cfg['device'])


if __name__ == "__main__":
  os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')

  from fire import Fire
  Fire(main)