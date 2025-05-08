# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None


class Mamba_bigram(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        bias=False,
        use_fast_path=False,
        layer_idx=None,
        use_bigram_layers=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse of softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))  # Keep in fp32

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    # def bigram_embedding(self, hidden_states):
    #     """
    #     Compute bigram embedding: 
    #     Each token embedding is half from the previous token and half from itself.
    #     """
    #     prev_tokens = torch.cat([torch.zeros_like(hidden_states[:, :1, :]), hidden_states[:, :-1, :]], dim=1)
    #     return 0.5 * prev_tokens + 0.5 * hidden_states  # Mix 50% previous and 50% current token
    
    def bigram_embedding(self, h):
        """Applies bigram embedding (half from previous, half from current)."""
        s = h.size()
        h = h.reshape(s[0], -1)  
        d2 = s[2] // 2  # Half of embedding dimension
        h = h.roll(d2, 1)  # Shift right by `d2`
        h[:, :d2] = 0  # Zero out first part to prevent garbage values
        h = h.reshape(*s)  # Restore shape
        return h

    def quagram_concat(self, x):
        """
        Concatenates quartered embeddings from the previous 3 tokens and current token.
        Assumes embedding dimension D is divisible by 4.

        Args:
            x: Tensor of shape (B, L, D)

        Returns:
            Tensor of shape (B, L, D), with:
                - [:D//4]   from x_{t-3}
                - [D//4:2D//4] from x_{t-2}
                - [2D//4:3D//4] from x_{t-1}
                - [3D//4:]  from x_t
        """
        B, L, D = x.shape
        d4 = D // 4
        assert D % 4 == 0, "Embedding dim must be divisible by 4 for quagram_concat."

        x_t    = x
        x_tm1  = torch.roll(x, shifts=1, dims=1)
        x_tm2  = torch.roll(x, shifts=2, dims=1)
        x_tm3  = torch.roll(x, shifts=3, dims=1)

        # Prevent information leak at the start of the sequence
        x_tm1[:, 0, :] = 0
        x_tm2[:, :2, :] = 0
        x_tm3[:, :3, :] = 0

        # Concatenate selected quarters from each token
        out = torch.cat([
            x_tm3[:, :, :d4],          # First quarter from t-3
            x_tm2[:, :, d4:2*d4],      # Second quarter from t-2
            x_tm1[:, :, 2*d4:3*d4],    # Third quarter from t-1
            x_t[:, :, 3*d4:],          # Fourth quarter from t
        ], dim=-1)

        return out

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        # print(f"[INFO] Using MambaBigram: {hidden_states.shape}")
        batch, seqlen, dim = hidden_states.shape

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        x, z = xz.chunk(2, dim=1)

        # Continue with the rest of Mamba forward pass
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        
        assert self.activation in ["silu", "swish"]
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,  # Keep z from projection
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )

        y = rearrange(y, "b d l -> b l d")
        y = self.bigram_embedding(y)  # Apply bigram embedding
        out = self.out_proj(y)
        return out