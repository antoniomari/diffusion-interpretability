'''
Adapted from
https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/model.py
'''

import torch
import torch.nn as nn
import os
import json

class SparseAutoencoder(nn.Module):
    """
    Top-K Autoencoder with sparse kernels. Implements:

        latents = relu(topk(encoder(x - pre_bias) + latent_bias))
        recons = decoder(latents) + pre_bias
    """

    def __init__(
        self,
        n_dirs_local: int,
        d_model: int,
        k: int,
        auxk: int | None,
        dead_steps_threshold: int,
    ):
        """Sparse Autoencoder constructor

        Args:
            n_dirs_local (int): # hidden dim of SAE
            d_model (int): # dim of model activation (input and output dim of SAE)
            k (int): #TODO: check
            auxk (int | None): #TODO: check
            dead_steps_threshold (int): #TODO: check
        """
    
        super().__init__()
        self.n_dirs_local = n_dirs_local
        self.d_model = d_model
        self.k = k
        self.auxk = auxk
        self.dead_steps_threshold = dead_steps_threshold

        self.encoder = nn.Linear(d_model, n_dirs_local, bias=False)
        self.decoder = nn.Linear(n_dirs_local, d_model, bias=False)

        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        self.latent_bias = nn.Parameter(torch.zeros(n_dirs_local))

        # buffer that stores how many iterations have passed since each latent unit was last activated
        # if this value is higher than a threshold, then that feature is "dead"? 
        self.stats_last_nonzero: torch.Tensor
        self.register_buffer("stats_last_nonzero", torch.zeros(n_dirs_local, dtype=torch.long))  #TODO: check

        def auxk_mask_fn(x: torch.Tensor):
            """ 
            Masks non-dead features, keeping active the dead ones.
            """
            # dead mask is 1 for each dimension that did not activate for more than threshold, 0 otherwise
            dead_mask = self.stats_last_nonzero > dead_steps_threshold
            x.data *= dead_mask  # inplace to save memory
            return x

        self.auxk_mask_fn = auxk_mask_fn

        ## initialization

        # "tied" init NOTE: since the cloning, we are not actually tieing the weights, are we?
        self.decoder.weight.data = self.encoder.weight.data.T.clone()

        # store decoder in column major layout for kernel: TODO: checkout if this operation works as intended
        self.decoder.weight.data = self.decoder.weight.data.T.contiguous().T

        unit_norm_decoder_(self)

    def save_to_disk(self, path: str):
        PATH_TO_CFG = 'config.json'
        PATH_TO_WEIGHTS = 'state_dict.pth'
        
        cfg = {
            "n_dirs_local": self.n_dirs_local,
            "d_model": self.d_model,
            "k": self.k,
            "auxk": self.auxk,
            "dead_steps_threshold": self.dead_steps_threshold,
        }

        os.makedirs(path, exist_ok=True)

        # save config (hidden_dim, input_dim, k, auxk, dead_steps_thresholds)
        with open(os.path.join(path, PATH_TO_CFG), 'w') as f:
            json.dump(cfg, f)
        
        # save model state dict
        torch.save({
            "state_dict": self.state_dict(),
        }, os.path.join(path, PATH_TO_WEIGHTS))


    @classmethod
    def load_from_disk(cls, path: str):
        PATH_TO_CFG = 'config.json'
        PATH_TO_WEIGHTS = 'state_dict.pth'

        with open(os.path.join(path, PATH_TO_CFG), 'r') as f:
            cfg = json.load(f)

        ae = cls(
            n_dirs_local=cfg["n_dirs_local"],
            d_model=cfg["d_model"],
            k=cfg["k"],
            auxk=cfg["auxk"],
            dead_steps_threshold=cfg["dead_steps_threshold"],
        )

        state_dict = torch.load(os.path.join(path, PATH_TO_WEIGHTS))["state_dict"]
        ae.load_state_dict(state_dict)

        return ae

    @property
    def n_dirs(self):
        return self.n_dirs_local

    def encode(self, x: torch.Tensor):
        x = x - self.pre_bias # x.shape = TODO: check (Batch?, d_model)
        latents_pre_act = self.encoder(x) + self.latent_bias  # shape = (Batch, n_dirs_local)

        # gets the K largest activations (and their indices) for each elem of the batch
        # vals: (batch, K), inds: (batch, K)
        vals, inds = torch.topk(
            latents_pre_act,
            k=self.k,
            dim=-1
        )   
        
        # The latents will be 0 for all non-topk activations, ReLU(x) for the topk.
        latents = torch.zeros_like(latents_pre_act)  # shape = (batch, hidden_dim)
        latents.scatter_(-1, inds, torch.relu(vals))  # for (b, k), v in (inds, vals): latents[b, k] = ReLU(v)
        # This is materializing the sparse tensor, in forward this does not happen (use sparse tensor implementation instead)

        return latents

    def forward(self, x: torch.Tensor):

        # NOTE: duplication of code
        x = x - self.pre_bias
        latents_pre_act: torch.Tensor = self.encoder(x) + self.latent_bias
        vals, inds = torch.topk(
            latents_pre_act,
            k=self.k,
            dim=-1
        )

        ## set num nonzero stat ## NOTE: these stats could be used for regularization 
        # and to adapt autoencoder dynamically to avoid dead neurons
        tmp = torch.zeros_like(self.stats_last_nonzero)

        # since tmp is 1d tensor, scatter_add_ expects 1d -> iterates on (i, v) in (inds, vals) incrementing tmp[i] += (v > 1e-3)
        # so tmp contains the number of activations of each latent feature for the current batch
        tmp.scatter_add_(
            0,
            inds.reshape(-1),
            (vals > 1e-3).to(tmp.dtype).reshape(-1),  # use threshold 1e-3, everything lower considered 0
        )

        # if feature was activated, the stats_last_nonzero is reset to 0, otherwise the value is kept
        self.stats_last_nonzero *= 1 - tmp.clamp(max=1)
        self.stats_last_nonzero += 1
        ## end stats ##

        ## Selects the top-auxK dead features 
        # NOTE: top-auxK are not used for reconstruction, find out how they are used.
        if self.auxk is not None:  # for auxk
            # IMPORTANT: has to go after stats update!
            # WARN: auxk_mask_fn can mutate latents_pre_act!
            auxk_vals, auxk_inds = torch.topk(
                self.auxk_mask_fn(latents_pre_act),
                k=self.auxk,
                dim=-1
            )
        else:
            auxk_inds = None
            auxk_vals = None

        ## end auxk

        # applies ReLU activation function to vals (and possibly auxk_vals)
        vals = torch.relu(vals)
        if auxk_vals is not None:
            auxk_vals = torch.relu(auxk_vals)

        # rows is batch size, cols is hidden_dim
        rows, cols = latents_pre_act.size()
        # [0, 1, Batch -1] => [[0], [1], ..., [B -1]] => (k=3) [[0, 0, 0], [1, 1, 1], ..., [B-1, B-1, B-1]] => [0, 0, 0, 1, 1, 1, ..., B-1, B-1, B-1]
        row_indices = torch.arange(rows).unsqueeze(1).expand(-1, self.k).reshape(-1)
        vals = vals.reshape(-1)  # [val_b=1_k=1, val_b=1_k=2, val_b=1_k=3, ...]
        inds = inds.reshape(-1)  # [ind_b=1_k=1, ind_b=1_k=2, ind_b=1_k=3, ...]

        indices = torch.stack([row_indices.to(inds.device), inds]) # creates couples (position_in_batch, index_of_val)
        # indices is a collection of 2d-indices that specify where a value is stored
        # sparse tensor, representing a tensor of shape (batch, hidden_dim) -> 
        # we specify each element for val in vals: T[b, k] = val (where (b, k) is corresponding element of indices
        sparse_tensor = torch.sparse_coo_tensor(indices, vals, torch.Size([rows, cols]))
        # Sparse matrix multiplication for optimized decoding
        recons = torch.sparse.mm(sparse_tensor, self.decoder.weight.T) + self.pre_bias


        return recons, {
            "inds": inds,
            "vals": vals,
            "auxk_inds": auxk_inds,
            "auxk_vals": auxk_vals,
        }

    
    def decode_sparse(self, inds: torch.Tensor, vals: torch.Tensor):
        # NOTE: duplicate code, TODO: reuse it?
        # inds (Batch, K), vals (Batch, K)
        rows, cols = inds.shape[0], self.n_dirs
        
        row_indices = torch.arange(rows).unsqueeze(1).expand(-1, inds.shape[1]).reshape(-1)
        vals = vals.reshape(-1)
        inds = inds.reshape(-1)

        indices = torch.stack([row_indices.to(inds.device), inds])

        sparse_tensor = torch.sparse_coo_tensor(indices, vals, torch.Size([rows, cols]))

        recons = torch.sparse.mm(sparse_tensor, self.decoder.weight.T) + self.pre_bias
        return recons

    @property
    def device(self):
        return next(self.parameters()).device


def unit_norm_decoder_(autoencoder: SparseAutoencoder) -> None:
    """
    Unit normalize the decoder weights of an autoencoder.
    """
    autoencoder.decoder.weight.data /= autoencoder.decoder.weight.data.norm(dim=0)


def unit_norm_decoder_grad_adjustment_(autoencoder) -> None:
    """project out gradient information parallel to the dictionary vectors - assumes that the decoder is already unit normed"""

    assert autoencoder.decoder.weight.grad is not None

    autoencoder.decoder.weight.grad +=\
        torch.einsum("bn,bn->n", autoencoder.decoder.weight.data, autoencoder.decoder.weight.grad) *\
        autoencoder.decoder.weight.data * -1
        