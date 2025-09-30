import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from src.utils import set_seed, pick_device, new_run_dir, get_cfg, CsvLogger
from src.datasets import make_loaders
from src.plots import plot_curves
from src.physics import burgers_residual, fixed_collocation
from src.model import MLP


def data_loss(pred, target, kind="huber", delta=0.5):
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "l1":
        return F.l1_loss(pred, target)
    if kind == "huber":
        return F.huber_loss(pred, target, delta=delta)
    raise ValueError(f"Unknown data loss kind: {kind}")


def main(cfg_path: str = "config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # --- deterministic setup & device ---
    seed = get_cfg(cfg, "seed", int, 42)
    set_seed(seed)
    device = pick_device(get_cfg(cfg, "device", str, "auto"))

    # --- data loaders ---
    dl_tr, dl_vl = make_loaders(cfg)

    # --- model ---
    widths = tuple(get_cfg(cfg, "model.widths", list, [64, 64, 64]))
    spectral_norm = bool(get_cfg(cfg, "model.spectral_norm", bool, False))
    model = MLP(widths=widths, spectral_norm=spectral_norm).to(device)

    # --- optimizer / train params (типобезопасно) ---
    lr = float(get_cfg(cfg, "train.lr", float, 1e-3))
    if not (lr > 0):
        raise ValueError(f"train.lr должно быть > 0, получено: {lr}")
    epochs = int(get_cfg(cfg, "train.epochs", int, 200))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # --- losses config ---
    loss_kind = str(get_cfg(cfg, "loss.data", str, "huber"))
    huber_delta = float(get_cfg(cfg, "loss.huber_delta", float, 0.5))
    phys_w = float(get_cfg(cfg, "loss.physics_weight", float, 1.0))

    # --- fixed collocation for val physics (репродьюсабельно) ---
    n_col = int(get_cfg(cfg, "data.collocation.n_points", int, 20000))
    col_file = str(get_cfg(cfg, "data.collocation.fixed_file", str, "data/splits/colloc_seed42.npy"))
    x_c, t_c = fixed_collocation(n_col, col_file, seed=seed)
    x_c, t_c = x_c.to(device), t_c.to(device)

    # --- logging ---
    run_dir = new_run_dir()
    logger = CsvLogger(
        run_dir / "log.csv",
        ["epoch", "tr_total", "tr_data", "tr_phys", "vl_total", "vl_data", "vl_phys"],
    )
    save_best = bool(get_cfg(cfg, "logging.save_best", bool, True))
    best, best_ep = float("inf"), -1

    for ep in range(1, epochs + 1):
        # ===== TRAIN =====
        model.train()
        trd = trp = trt = 0.0
        nb = 0
        for X, y in tqdm(dl_tr, desc=f"epoch {ep}", leave=False):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            l_data = data_loss(pred, y, loss_kind, huber_delta)

            x, t = X[:, :1], X[:, 1:2]
            # create_graph=False достаточно (мы не делаем higher-order по параметрам)
            R = burgers_residual(model, x, t, create_graph=False)
            l_phys = torch.mean(R ** 2)

            loss = l_data + phys_w * l_phys

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            trd += float(l_data.item())
            trp += float(l_phys.item())
            trt += float(loss.item())
            nb += 1

        trd /= max(1, nb)
        trp /= max(1, nb)
        trt /= max(1, nb)

        # ===== VAL =====
        model.eval()
        with torch.no_grad():
            vd = 0.0
            nbv = 0
            for X, y in dl_vl:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                l_data = data_loss(pred, y, loss_kind, huber_delta)
                vd += float(l_data.item())
                nbv += 1
            vd /= max(1, nbv)

        # фиксированные коллокации для вал. физики (бездефектно и стабильно)
        with torch.enable_grad():
            vp = torch.mean(burgers_residual(model, x_c, t_c, create_graph=False) ** 2).item()

        vt = vd + phys_w * vp

        logger.log(
            {
                "epoch": ep,
                "tr_total": trt,
                "tr_data": trd,
                "tr_phys": trp,
                "vl_total": vt,
                "vl_data": vd,
                "vl_phys": vp,
            }
        )

        if vt < best:
            best, best_ep = vt, ep
            if save_best:
                torch.save(model.state_dict(), run_dir / "best.pt")

    logger.close()

    if bool(get_cfg(cfg, "logging.make_plots", bool, False)):
        cols = get_cfg(cfg, "logging.plots.curves", list, 
                    ["tr_total","vl_total","tr_data","vl_data","tr_phys","vl_phys"])
        dpi  = int(get_cfg(cfg, "logging.plots.dpi", int, 200))
        fname= str(get_cfg(cfg, "logging.plots.filename", str, "curve.png"))
        out = plot_curves(run_dir, columns=cols, dpi=dpi, filename=fname)
        print("Saved plot:", out)
    (run_dir / "_meta.txt").write_text(f"best_val_total={best:.6e} at epoch {best_ep}\n")
    print("Done. Run dir:", run_dir)


if __name__ == "__main__":
    main()
