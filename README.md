# 🧪 pinn-lab

**pinn-lab** is a minimal and reproducible framework for **Physics-Informed Neural Networks (PINNs)** on **PDEBench**.  
It is designed as the foundation for research on **robust PINNs**, **multi-fidelity training**, and **adaptive loss balancing**, with extensible backbones such as MLP, SIREN, Fourier features, and attention-based PINNs.

---

## 🚀 Features

- Baseline PINN (MLP backbone) for Burgers 1D.  
- Support for **HDF5 datasets** (PDEBench) and synthetic fallback.  
- Fixed **collocation points** for reproducible validation.  
- Loss functions: MSE, L1, Huber.  
- CSV logging and automatic training curve plots.  
- Roadmap: Robust PINNs, multi-fidelity PINNs, adaptive balancing, and alternative backbones.  

---

## 📂 Repository structure

```
pinn-lab/
├─ config.yaml              # experiment configuration
├─ requirements.txt
├─ LICENSE
├─ README.md
├─ data/
│  ├─ raw/                  # raw PDEBench HDF5 files (not included)
│  ├─ processed/            # optional preprocessing
│  ├─ splits/               # fixed collocation points (included for reproducibility)
│  │   └─ colloc_seed42.npy
├─ results/
│  ├─ runs/                 # logs and checkpoints per run
│  ├─ figures/              # generated plots
│  └─ tables/               # summary tables
├─ src/
│  ├─ datasets.py           # datasets (Burgers, later DR/NS)
│  ├─ losses.py             # data/physics losses and metrics
│  ├─ model.py              # MLP + stubs for SIREN/Fourier/Attention
│  ├─ physics.py            # PDE residuals
│  ├─ plots.py              # plotting utilities
│  ├─ train.py              # main training loop
│  └─ utils.py              # helpers (seed, logging, etc.)
└─ scripts/
└─ experiments/          # experiment scripts (robust, multi-fidelity, adaptive, backbones)

```

---

## ⚙️ Installation

```bash
git clone https://github.com/maridze/pinn-lab.git
cd pinn-lab
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
````

> On Apple Silicon or ARM, you may need to upgrade pip first:
>
> ```bash
> pip install --upgrade pip setuptools wheel

---

## 📥 Data

* Download [PDEBench](https://github.com/pdebench/PDEBench) datasets (e.g., Burgers 1D).
* Place the `.h5` files into `data/raw/`.
* If no file is found, the code will use **synthetic data**.

⚠️ Large raw datasets are **not included in this repo**.
Only small reproducibility artifacts (`data/splits/*.npy`) are tracked.

---

## 📝 Example `config.yaml`

```yaml
seed: 42
device: "auto"          # auto|cuda|cpu
task: "burgers1d"

experiment:
  tag: "baseline"

data:
  h5_path: "data/raw/burgers_train.h5"   # path to PDEBench file (optional)
  val_h5_path: null
  batch_size: 256
  collocation:
    n_points: 20000
    fixed_file: "data/splits/colloc_seed42.npy"

model:
  name: "mlp"
  widths: [64, 64, 64]
  spectral_norm: false

train:
  epochs: 200
  lr: 1e-3

loss:
  data: "huber"          # mse|l1|huber
  huber_delta: 0.5
  physics_weight: 1.0
  adaptive:
    enabled: false
    method: null          # gradnorm|dwa|ours

eval:
  metrics: ["mae","rmse"]

logging:
  save_best: true
  csv_every: 1
  make_plots: true
  plots:
    curves: ["tr_total","vl_total","tr_data","vl_data","tr_phys","vl_phys"]
    dpi: 200
    filename: "curve.png"
```

---

## 📑 License

MIT License © 2025
