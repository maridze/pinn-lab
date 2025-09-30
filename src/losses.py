import torch.nn.functional as F
import torch

def data_loss(pred, target, cfg):
    kind = cfg["loss"].get("data", "huber")
    delta = float(cfg["loss"].get("huber_delta", 0.5))
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "l1":
        return F.l1_loss(pred, target)
    if kind == "huber":
        return F.huber_loss(pred, target, delta=delta)
    raise ValueError(f"Unknown data loss {kind}")

def physics_loss(model, x, t, cfg, residual_fn):
    R = residual_fn(model, x, t, **cfg.get("physics", {}))
    return torch.mean(R**2)

def compute_metrics(pred, y):
    mae = torch.mean(torch.abs(pred - y)).item()
    rmse = torch.sqrt(torch.mean((pred - y) ** 2)).item()
    return {"mae": mae, "rmse": rmse}

def resolve_loss_weights(cfg, step=0):
    # пока фиксированные веса
    w_data = float(cfg["loss"].get("data_weight", 1.0))
    w_phys = float(cfg["loss"].get("physics_weight", 1.0))
    return {"data": w_data, "phys": w_phys}
