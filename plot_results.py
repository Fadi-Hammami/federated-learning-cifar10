"""
Generate the figures for the presentation from results/*.json

  python plot_results.py                       # uses A_iid + B_noniid_label (+ B_noniid_dir if present)
  python plot_results.py A_iid B_noniid_label  # explicit tags

Figures are saved in figures/.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"A_iid": "#2a6f97", "B_noniid_shards": "#c8553d", "B_noniid_dir": "#e3a72f"}
LABELS = {"A_iid": "A — IID", "B_noniid_shards": "B — non-IID (2 shards ≈ 2 classes/client)",
          "B_noniid_dir": "B' — non-IID Dirichlet (α=0.3)"}


def load(tag):
    with open(f"results/{tag}.json") as f:
        return json.load(f)


def main(tags):
    os.makedirs("figures", exist_ok=True)
    runs = {t: load(t) for t in tags if os.path.exists(f"results/{t}.json")}
    if not runs:
        print("no results found"); return

    # 1) accuracy + loss curves ------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for t, r in runs.items():
        h = r["history"]; c = COLORS.get(t, None); lab = LABELS.get(t, t)
        ax[0].plot(h["round"], np.array(h["test_acc"]) * 100, label=lab, color=c, lw=2)
        ax[1].plot(h["round"], h["test_loss"], label=lab, color=c, lw=2)
    ax[0].set(xlabel="Communication round", ylabel="Global test accuracy (%)", title="Accuracy of the global model")
    ax[0].axhline(60, ls="--", color="grey", lw=1); ax[0].text(0.5, 60.4, "60 %", color="grey", fontsize=8)
    ax[1].set(xlabel="Communication round", ylabel="Global test loss", title="Loss of the global model")
    for a in ax: a.grid(alpha=.3); a.legend()
    fig.tight_layout(); fig.savefig("figures/accuracy_loss.png", dpi=160); plt.close(fig)

    # 2) data distribution per client (class histogram heatmap) ----------------------
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4.5), squeeze=False)
    for a, (t, r) in zip(axes[0], runs.items()):
        hist = np.array(r["class_histogram"])
        im = a.imshow(hist, aspect="auto", cmap="Blues")
        a.set(xlabel="Class", ylabel="Client id", title=LABELS.get(t, t), xticks=range(10))
        plt.colorbar(im, ax=a, fraction=0.046, label="# samples")
    fig.tight_layout(); fig.savefig("figures/data_distribution.png", dpi=160); plt.close(fig)

    # 3) variance between clients (final global model on each client's data) --------
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / len(runs)
    for i, (t, r) in enumerate(runs.items()):
        accs = np.array([p["acc_on_local_data"] for p in r["per_client"]]) * 100
        x = np.arange(len(accs)) + i * width
        ax.bar(x, accs, width=width, color=COLORS.get(t), label=f"{LABELS.get(t, t)}  (std={accs.std():.1f})")
    ax.set(xlabel="Client id", ylabel="Accuracy of final global model\non client's local data (%)",
           title="Variance between clients", ylim=(0, 105))
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig("figures/client_variance.png", dpi=160); plt.close(fig)

    # 4) stability: std of client update norms per round ----------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    for t, r in runs.items():
        ax.plot(r["history"]["round"], r["history"]["update_norm_std"], color=COLORS.get(t), lw=2, label=LABELS.get(t, t))
    ax.set(xlabel="Communication round", ylabel="Std of ||w_k − w_global|| across clients",
           title="Client drift (training stability)")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig("figures/client_drift.png", dpi=160); plt.close(fig)

    # 5) summary table ---------------------------------------------------------------
    rows = []
    for t, r in runs.items():
        h = r["history"]; acc = np.array(h["test_acc"])
        def rounds_to(th):
            idx = np.argmax(acc >= th); return int(h["round"][idx]) if acc[idx] >= th else "—"
        last5 = acc[-5:] * 100
        cv = r["client_variance"]
        rows.append([LABELS.get(t, t), f"{acc[-1]*100:.2f}", f"{acc.max()*100:.2f}", rounds_to(.40), rounds_to(.50), rounds_to(.60),
                     f"{last5.std():.2f}", f"{cv['acc_local_std']*100:.1f}", f"{cv['acc_local_min']*100:.1f}",
                     f"{r['elapsed_sec']/60:.1f}"])
    header = ["Scenario", "Final acc %", "Best acc %", "Rounds→40%", "Rounds→50%", "Rounds→60%",
              "Std last 5 rounds", "Per-client acc std", "Worst client acc %", "Time (min)"]
    with open("figures/summary_table.md", "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for row in rows: f.write("| " + " | ".join(str(x) for x in row) + " |\n")
    print(open("figures/summary_table.md", encoding="utf-8").read())
    print("figures saved in figures/")


if __name__ == "__main__":
    tags = sys.argv[1:] or ["A_iid", "B_noniid_shards", "B_noniid_dir"]
    main(tags)
