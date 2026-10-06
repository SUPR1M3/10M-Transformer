import os, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_pilot(out="figures/pilot_curves.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for path in sorted(glob.glob("results/pilot/*.json")):
        r = json.load(open(path))
        tokens, loss = zip(*r["curve"])
        ax.loglog(tokens, loss, label=f'{r.get("max_digits", "?")}d, N={r["N"]/1e3:.0f}k, lr={r["lr"]:g}')
    ax.set(xlabel="tokens seen", ylabel="test loss (answer tokens)", title="Pilot: loss vs tokens")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

def isoflop_valleys(df):
    """For each budget C: fit a parabola to log(loss) vs log(N), and return its minimum."""
    rows = []
    for C, g in df.groupby("C"):
        if len(g) < 3:
            continue
        x, y = np.log(g["N"]), np.log(g["loss"])
        c2, c1, c0 = np.polyfit(x, y, 2)
        log_n_opt = -c1 / (2 * c2)
        valid = c2 > 0 and x.min() < log_n_opt < x.max()      # a real valley, inside the grid
        n_opt = np.exp(log_n_opt)
        rows.append({"C": C, "N_opt": n_opt, "D_opt": C / (6 * n_opt),
                     "valid": valid, "coef": (c2, c1, c0)})
    return pd.DataFrame(rows)

def plot_isoflop(df, arch, out):
    v = isoflop_valleys(df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    budgets = sorted(df["C"].unique())

    for i, C in enumerate(budgets):
        g = df[df["C"] == C]
        col = plt.cm.viridis(i / max(1, len(budgets) - 1))
        ax1.scatter(g["N"], g["loss"], color=col, s=15)
        row = v[v["C"] == C]
        if len(row):
            c2, c1, c0 = row["coef"].iloc[0]
            xs = np.linspace(np.log(g["N"].min()), np.log(g["N"].max()), 100)
            ax1.plot(np.exp(xs), np.exp(c2 * xs**2 + c1 * xs + c0), color=col, label=f"C={C:.0e}")
            if row["valid"].iloc[0]:
                ax1.scatter(row["N_opt"], np.exp(c2 * np.log(row["N_opt"])**2
                            + c1 * np.log(row["N_opt"]) + c0), marker="*", s=120, color=col)
    ax1.set(xscale="log", yscale="log", xlabel="N (non-embedding params)",
            ylabel="final test loss", title=f"IsoFLOP profiles ({arch})")
    ax1.legend(fontsize=7)

    good = v[v["valid"]]
    a = None
    if len(good) >= 2:
        a, b = np.polyfit(np.log(good["C"]), np.log(good["N_opt"]), 1)
        cs = np.logspace(np.log10(good["C"].min()), np.log10(good["C"].max()), 50)
        ax2.loglog(good["C"], good["N_opt"], "o", label="valley minima")
        ax2.loglog(cs, np.exp(b) * cs**a, label=f"N_opt ∝ C^{a:.2f}  (D_opt ∝ C^{1 - a:.2f})")
        ax2.legend()
    ax2.set(xlabel="compute C (FLOPs)", ylabel="compute-optimal N", title="Compute-optimal model size")

    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return a

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    if glob.glob("results/pilot/*.json"):
        plot_pilot()
    if os.path.exists("results/runs.csv"):
        df = pd.read_csv("results/runs.csv")
        for arch, g in df.groupby("arch"):
            a = plot_isoflop(g, arch, f"figures/isoflop_{arch}.png")
            print(f"{arch}: N_opt ∝ C^{a:.3f}" if a is not None else f"{arch}: not enough valid valleys")