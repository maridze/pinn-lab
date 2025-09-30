import torch
import torch.nn as nn

# === базовый MLP ===
class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, widths=(64, 64, 64), spectral_norm=False):
        super().__init__()
        layers, last = [], in_dim
        for h in widths:
            lin = nn.Linear(last, h)
            if spectral_norm:
                lin = nn.utils.spectral_norm(lin)
            layers += [lin, nn.Tanh()]
            last = h
        out = nn.Linear(last, out_dim)
        if spectral_norm:
            out = nn.utils.spectral_norm(out)
        self.net = nn.Sequential(*layers, out)

    def forward(self, x):
        return self.net(x)

# === заготовки для будущих бэкононов ===
class SIREN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("SIREN backbone not yet implemented")

class FourierMLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Fourier Features backbone not yet implemented")

# === фабрика моделей ===
def build_model(cfg):
    name = cfg["model"].get("name", "mlp").lower()
    if name == "mlp":
        return MLP(
            widths=tuple(cfg["model"].get("widths", [64, 64, 64])),
            spectral_norm=cfg["model"].get("spectral_norm", False),
        )
    elif name == "siren":
        return SIREN()
    elif name == "fourier":
        return FourierMLP()
    else:
        raise ValueError(f"Unknown model: {name}")

