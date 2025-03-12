'''
Adapted from
https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/train.py
'''


import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from typing import Callable, Iterable, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ReduceOp
from SAE.dataset_iterator import ActivationsDataloader
from SAE.sae import SparseAutoencoder, unit_norm_decoder_, unit_norm_decoder_grad_adjustment_
from SAE.sae_utils import SAETrainingConfig, Config

from types import SimpleNamespace
from typing import Optional, List
import json

import tqdm

def weighted_average(points: torch.Tensor, weights: torch.Tensor):
    weights = weights / weights.sum()
    return (points * weights.view(-1, 1)).sum(dim=0)


@torch.no_grad()
def geometric_median_objective(
    median: torch.Tensor, points: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:

    norms = torch.linalg.norm(points - median.view(1, -1), dim=1)  # type: ignore

    return (norms * weights).sum()


def compute_geometric_median(
    points: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    maxiter: int = 100,
    ftol: float = 1e-20,
    do_log: bool = False,
):
    """
    :param points: ``torch.Tensor`` of shape ``(n, d)``
    :param weights: Optional ``torch.Tensor`` of shape :math:``(n,)``.
    :param eps: Smallest allowed value of denominator, to avoid divide by zero.
        Equivalently, this is a smoothing parameter. Default 1e-6.
    :param maxiter: Maximum number of Weiszfeld iterations. Default 100
    :param ftol: If objective value does not improve by at least this `ftol` fraction, terminate the algorithm. Default 1e-20.
    :param do_log: If true will return a log of function values encountered through the course of the algorithm
    :return: SimpleNamespace object with fields
        - `median`: estimate of the geometric median, which is a ``torch.Tensor`` object of shape :math:``(d,)``
        - `termination`: string explaining how the algorithm terminated.
        - `logs`: function values encountered through the course of the algorithm in a list (None if do_log is false).
    """
    with torch.no_grad():

        if weights is None:
            weights = torch.ones((points.shape[0],), device=points.device)
        # initialize median estimate at mean
        new_weights = weights
        median = weighted_average(points, weights)
        objective_value = geometric_median_objective(median, points, weights)
        if do_log:
            logs = [objective_value]
        else:
            logs = None

        # Weiszfeld iterations
        early_termination = False
        pbar = tqdm.tqdm(range(maxiter))
        for _ in pbar:
            prev_obj_value = objective_value

            norms = torch.linalg.norm(points - median.view(1, -1), dim=1)  # type: ignore
            new_weights = weights / torch.clamp(norms, min=eps)
            median = weighted_average(points, new_weights)
            objective_value = geometric_median_objective(median, points, weights)

            if logs is not None:
                logs.append(objective_value)
            if abs(prev_obj_value - objective_value) <= ftol * objective_value:
                early_termination = True
                break

            pbar.set_description(f"Objective value: {objective_value:.4f}")

    median = weighted_average(points, new_weights)  # allow autodiff to track it
    return SimpleNamespace(
        median=median,
        new_weights=new_weights,
        termination=(
            "function value converged within tolerance"
            if early_termination
            else "maximum iterations reached"
        ),
        logs=logs,
    )

def maybe_transpose(x):
    return x.T if not x.is_contiguous() and x.T.is_contiguous() else x

import wandb

RANK = 0

class Logger:
    def __init__(self, sae_name, **kws):
        self.vals = {}
        self.enabled = (RANK == 0) and not kws.pop("dummy", False)
        self.sae_name = sae_name

    def logkv(self, k, v):
        if self.enabled:
            self.vals[f'{self.sae_name}/{k}'] = v.detach() if isinstance(v, torch.Tensor) else v
        return v

    def dumpkvs(self, step):
        if self.enabled:
            wandb.log(self.vals, step=step)
            self.vals = {}
    

class FeaturesStats:
    def __init__(self, dim, logger):
        self.dim = dim
        self.logger = logger
        self.reinit()

    def reinit(self):
        self.n_activated = torch.zeros(self.dim, dtype=torch.long, device="cuda")
        self.n = 0
    
    def update(self, inds):
        self.n += inds.shape[0]
        inds = inds.flatten().detach()
        self.n_activated.scatter_add_(0, inds, torch.ones_like(inds))

    def log(self):
        self.logger.logkv('activated', (self.n_activated / self.n + 1e-9).log10().cpu().numpy())

def training_loop_(
    aes, 
    train_acts_iter, 
    loss_fn, 
    log_interval, 
    save_interval,
    loggers,
    sae_cfgs,
):
    sae_packs = []
    for ae, cfg, logger in zip(aes, sae_cfgs, loggers):
        pbar = tqdm.tqdm(unit=" steps", desc="Training Loss: ")
        fstats = FeaturesStats(ae.n_dirs, logger)
        opt = torch.optim.Adam(ae.parameters(), lr=cfg.lr, eps=cfg.eps, fused=True)
        sae_packs.append((ae, cfg, logger, pbar, fstats, opt))
    
    for i, flat_acts_train_batch in enumerate(train_acts_iter):
        flat_acts_train_batch = flat_acts_train_batch.cuda()

        for ae, cfg, logger, pbar, fstats, opt in sae_packs:
            recons, info = ae(flat_acts_train_batch)
            loss = loss_fn(ae, cfg, flat_acts_train_batch, recons, info, logger)

            fstats.update(info['inds'])
            
            bs = flat_acts_train_batch.shape[0]
            logger.logkv('not-activated 1e4', (ae.stats_last_nonzero > 1e4 / bs).mean(dtype=float).item())
            logger.logkv('not-activated 1e6', (ae.stats_last_nonzero > 1e6 / bs).mean(dtype=float).item())
            logger.logkv('not-activated 1e7', (ae.stats_last_nonzero > 1e7 / bs).mean(dtype=float).item())

            logger.logkv('explained variance', explained_variance(recons, flat_acts_train_batch))
            logger.logkv('l2_div', (torch.linalg.norm(recons, dim=1) / torch.linalg.norm(flat_acts_train_batch, dim=1)).mean())

            if (i + 1) % log_interval == 0:
                fstats.log()
                fstats.reinit()
            
            if (i + 1) % save_interval == 0:
                ae.save_to_disk(f"{cfg.save_path}/{i + 1}")

            loss.backward()

            unit_norm_decoder_(ae)
            unit_norm_decoder_grad_adjustment_(ae)

            opt.step()
            opt.zero_grad()
            logger.dumpkvs(i)

            pbar.set_description(f"Training Loss {loss.item():.4f}")
            pbar.update(1)


    for ae, cfg, logger, pbar, fstats, opt in sae_packs:
        pbar.close()
        ae.save_to_disk(f"{cfg.save_path}/final")


def init_from_data_(ae, stats_acts_sample):
    ae.pre_bias.data = (
        compute_geometric_median(stats_acts_sample[:32768].float().cpu()).median.cuda().float()
    )


def mse(recons, x):
    # return ((recons - x) ** 2).sum(dim=-1).mean()
    return ((recons - x) ** 2).mean()

def normalized_mse(recon: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    # only used for auxk
    xs_mu = xs.mean(dim=0)

    loss = mse(recon, xs) / mse(
        xs_mu[None, :].broadcast_to(xs.shape), xs
    )

    return loss


def explained_variance(recons, x):
    """_summary_

    Args:
        recons (_type_): _description_
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Compute the variance of the difference
    diff = x - recons
    diff_var = torch.var(diff, dim=0, unbiased=False)

    # Compute the variance of the original tensor
    x_var = torch.var(x, dim=0, unbiased=False)

    # Avoid division by zero
    explained_var = 1 - diff_var / (x_var + 1e-8)

    return explained_var.mean()


def loss(ae: SparseAutoencoder, cfg_sae: SAETrainingConfig, flat_acts_train_batch, 
        recons, info, logger, mse_scale):
    # MSE
    logger.logkv("train_recons", mse_scale * mse(recons, flat_acts_train_batch))
    # AuxK
    + logger.logkv(
        "train_maxk_recons",
        cfg_sae.auxk_coef
        * normalized_mse(
            ae.decode_sparse(
                info["auxk_inds"],
                info["auxk_vals"],
            ),
            flat_acts_train_batch - recons.detach() + ae.pre_bias.detach(),
        ).nan_to_num(0),
    )
    


def main():
    cfg = Config(json.load(open('SAE/config.json')))

    dataloader = ActivationsDataloader(cfg.paths_to_latents, cfg.block_name, cfg.bs)

    acts_iter = dataloader.iterate()
    stats_acts_sample = torch.cat([
        next(acts_iter).cpu() for _ in range(10)
    ], dim=0)

    aes = [
        SparseAutoencoder(
            n_dirs_local=sae.n_dirs,
            d_model=sae.d_model,
            k=sae.k,
            auxk=sae.auxk,
            dead_steps_threshold=sae.dead_toks_threshold // cfg.bs,
        ).cuda()
        for sae in cfg.saes
    ]
    
    for ae in aes:
        init_from_data_(ae, stats_acts_sample)
    
    # TODO: what is this for? 
    mse_scale = (
        1 / ((stats_acts_sample.float().mean(dim=0) - stats_acts_sample.float()) ** 2).mean()
    )
    mse_scale = mse_scale.item()
    del stats_acts_sample

    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_name,
    )

    loggers = [Logger(
        sae_name=cfg_sae.sae_name,
        dummy=False,
    ) for cfg_sae in cfg.saes]

    training_loop_(
        aes,
        acts_iter,
        loss=lambda ae, cfg_sae, flat_acts_train_batch, recons, info, logger: loss(ae, cfg_sae, flat_acts_train_batch, recons, info, logger, mse_scale),
        sae_cfgs = cfg.saes,
        loggers=loggers,
        log_interval=cfg.log_interval,
        save_interval=cfg.save_interval,
    )


if __name__ == "__main__":
    main()