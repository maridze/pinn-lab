import os, random, numpy as np, torch, csv
from pathlib import Path

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pick_device(pref="auto"):
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref in ("auto", None) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def new_run_dir():
    import datetime
    d = Path("results/runs") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_cfg(cfg, path, cast=None, default=None):
    keys = path.split(".")
    cur = cfg
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    if cast:
        return cast(cur)
    return cur

class CsvLogger:
    def __init__(self, path, fieldnames):
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fieldnames)
        self.w.writeheader()
    def log(self, row: dict):
        self.w.writerow(row); self.f.flush()
    def close(self):
        self.f.close()

def export_table(df, name):
    out_csv = Path("results/tables") / f"{name}.csv"
    out_tex = Path("results/tables") / f"{name}.tex"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    try:
        df.to_latex(out_tex, index=False)
    except Exception:
        pass
    return out_csv

def save_predictions(model, loader, path):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X, y in loader:
            all_preds.append(model(X).cpu().numpy())
            all_true.append(y.cpu().numpy())
    np.savez(path, pred=np.vstack(all_preds), true=np.vstack(all_true))
