import os, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
try:
    import h5py
    H5OK = True
except Exception:
    H5OK = False

class BurgersDataset(Dataset):
    def __init__(self, h5_path=None, n_points=16000, split="train", seed=42):
        rng = np.random.default_rng(seed)
        if h5_path and os.path.exists(h5_path) and H5OK:
            with h5py.File(h5_path, "r") as f:
                grp = f.get(split)
                x = np.array(grp["x"]).reshape(-1,1).astype("float32")
                t = np.array(grp["t"]).reshape(-1,1).astype("float32")
                u = np.array(grp["u"])
                if u.ndim == 1: u = u.reshape(-1,1)
                u = u.astype("float32")
        else:
            x = rng.random((n_points,1), dtype=np.float32)
            t = rng.random((n_points,1), dtype=np.float32)
            c = 0.5
            u = (np.sin(np.pi*(x - c*t)) * np.exp(-t)).astype("float32")
        self.X = np.concatenate([x,t], axis=1)
        self.y = u
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])

def make_loaders(cfg):
    ds_tr = BurgersDataset(cfg["data"]["h5_path"], split="train", seed=1)
    ds_vl = BurgersDataset(cfg["data"].get("val_h5_path"), split="val", seed=2)
    bs = cfg["data"]["batch_size"]
    return (DataLoader(ds_tr, batch_size=bs, shuffle=True),
            DataLoader(ds_vl, batch_size=bs*2, shuffle=False))
