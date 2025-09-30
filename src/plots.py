import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

def plot_curves(run_dir, columns=None, dpi=200, filename="curve.png"):
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir/"log.csv")
    if columns is None:
        columns = ["tr_total","vl_total","tr_data","vl_data","tr_phys","vl_phys"]
    plt.figure()
    for col in columns:
        if col in df.columns:
            plt.plot(df["epoch"], df[col], label=col)
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    out = run_dir/filename
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    return out

def plot_predictions(pred_file, out_file="pred_vs_true.png"):
    import numpy as np
    data = np.load(pred_file)
    pred, true = data["pred"], data["true"]
    plt.figure()
    plt.scatter(true, pred, s=2, alpha=0.5)
    plt.xlabel("true"); plt.ylabel("pred")
    plt.plot([true.min(), true.max()], [true.min(), true.max()], "r--")
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    return out_file
