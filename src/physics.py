import numpy as np, torch
from pathlib import Path

def burgers_residual(model, x, t, nu=0.01, create_graph=True):
    x = x.requires_grad_()
    t = t.requires_grad_()
    xt = torch.cat([x, t], dim=1)
    u = model(xt)

    u_x, = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                               create_graph=True, retain_graph=True)
    u_t, = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                               create_graph=create_graph, retain_graph=True)
    u_xx, = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                create_graph=create_graph, retain_graph=True)
    return u_t + u * u_x - nu * u_xx

# заготовки для будущего
def dr_residual(model, x, t, **kwargs):
    raise NotImplementedError("Diffusion-Reaction residual not yet implemented")

def ns_residual(model, x, t, **kwargs):
    raise NotImplementedError("Navier–Stokes residual not yet implemented")

def fixed_collocation(n, file_path, seed=42):
    p = Path(file_path)
    if p.exists():
        arr = np.load(p)
    else:
        rng = np.random.default_rng(seed)
        x = rng.random((n,1), dtype=np.float32)
        t = rng.random((n,1), dtype=np.float32)
        arr = np.concatenate([x,t], axis=1)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, arr)
    xt = torch.from_numpy(arr.astype(np.float32))
    return xt[:, :1], xt[:, 1:2]


RESIDUALS = {
    "burgers1d": burgers_residual,
    "dr2d": dr_residual,
    "ns2d": ns_residual,
}
