import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import matplotlib.pyplot as plt


class PosDefiniteMamba(nn.Module):
    def __init__(self, dt_rank, d_inner, d_state, Wx_type="pos", threshold=0.7, orthotype="exp"):
        super(PosDefiniteMamba, self).__init__()
        self.dt_rank = dt_rank
        self.d_inner = d_inner
        self.d_state = d_state
        self.Wx_type = Wx_type
        self.threshold = float(threshold)

    def forward(self, X):
        """
            X: [dt_rank + d_state * 2, d_inner]
            diag: [d_state]
            out: x_proj = nn.Linear(d_inner, dt_rank + d_state * 2)
        """
        SD = X[:self.dt_rank, :]
        P = X[self.dt_rank:self.dt_rank+self.d_state, :]  # [d_state d_inner]
        diag = X[-self.d_state:,:][:, :1]
        if self.Wx_type == "neg":
            diag_softplus = -F.softplus(diag) - self.threshold
        else:
            diag_softplus = F.softplus(diag) + self.threshold
        P_triu_diag1 = torch.triu(P, diagonal=1).fill_diagonal_(1)
        SC = P_triu_diag1
        SB = diag_softplus * P_triu_diag1  # [d_state d_inner]

        out = torch.concat([SD, SB, SC], dim=0)
        
        # Cache for inspection
        self._last_SB = SB
        self._last_SC = SC
        self._last_diag = diag_softplus

        return out
    

def test_functional_PosDefiniteMamba(Wx_type="pos", save_path="plot.png"):
    dt_rank = 2
    d_state = 3
    d_inner = 6
    threshold = 0.7

    # Fix seed to ensure reproducibility
    torch.manual_seed(42)

    # Generate one shared input X
    X = torch.randn(dt_rank + 2 * d_state, d_inner)

    # Create and parametrize the Linear layer
    linear = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
    parametrize.register_parametrization(
        linear, "weight",
        PosDefiniteMamba(dt_rank=dt_rank, d_state=d_state, d_inner=d_inner, Wx_type=Wx_type, threshold=threshold)
    )

    # Run forward with the known input
    param_module = linear.parametrizations.weight[0]
    SB = param_module._last_SB
    SC = param_module._last_SC
    expected_scale = param_module._last_diag

    # Compute per-row scaling: SB ≈ scale * SC
    eps = 1e-8
    scale = SB / (SC + eps)
    scale_mean = scale.mean(dim=1, keepdim=True)

    print(f"[TEST] Wx_type: {Wx_type}")
    print(f"[TEST] Empirical scale per row:\n{scale_mean.squeeze()}")
    print(f"[TEST] Expected scale from diag:\n{expected_scale.squeeze()}")

    plt.figure()
    plt.plot(scale_mean.squeeze().detach().cpu().numpy(), label="empirical scale")
    plt.plot(expected_scale.squeeze().detach().cpu().numpy(), label="expected scale")
    plt.title(f"SB scaling vs. expected — Wx_type={Wx_type}")
    plt.xlabel("Row Index")
    plt.ylabel("Scale")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[SAVED] Plot saved to {save_path}")


# test_functional_PosDefiniteMamba("pos", save_path="pos_scaling.png")
# test_functional_PosDefiniteMamba("neg", save_path="neg_scaling.png")

# print(f"{torch.eye(4, 2)=}")

dt_rank = 2
d_inner = 3
d_state = 6

layer = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
parametrize.register_parametrization(
    layer, "weight",
    PosDefiniteMamba(dt_rank=dt_rank, d_inner=d_inner, d_state=d_state, Wx_type="pos", threshold=0.7)
)

X = torch.randn(dt_rank + 2 * d_state, d_inner)

W = layer.parametrizations.weight[0](X)

SB = layer.parametrizations.weight[0]._last_SB
SC = layer.parametrizations.weight[0]._last_SC

print("SB:\n", SB)
print("SC:\n", SC)