"""Generate the parameter-budget figure used in the report (numbers measured on CPU)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = r"C:\Users\Arjun Aggarwal\Documents\deepseek-harness\default-workspace\videomamba_echonet\report\fig_params.png"

labels = [4, 8, 16, 32]
in_proj = [184320, 368640, 737280, 1474560]
out_proj = [110592, 221184, 442368, 884736]
dt = [74928, 149808, 299568, 599088]
alog = [149040, 298032, 596016, 1191984]
total = [519265, 1038049, 2075617, 4150753]

fig, ax = plt.subplots(figsize=(6.4, 3.5), dpi=200)
x = np.arange(len(labels))
b1 = np.array(in_proj) / 1e6
b2 = np.array(out_proj) / 1e6
b3 = np.array(dt) / 1e6
b4 = np.array(alog) / 1e6
ax.bar(x, b1, 0.6, label="in_proj LoRA", color="#4C72B0")
ax.bar(x, b2, 0.6, bottom=b1, label="out_proj LoRA", color="#55A868")
ax.bar(x, b3, 0.6, bottom=b1 + b2, label="Delta-LoRA (dt_proj)", color="#C44E52")
ax.bar(x, b4, 0.6, bottom=b1 + b2 + b3, label="A-LoRA (A_log)", color="#8172B2")
for xi, t in zip(x, total):
    ax.text(xi, t / 1e6 + 0.07, f"{t:,}\n({100 * t / 25418113:.1f}%)", ha="center",
            fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([f"r = {r}" for r in labels])
ax.set_ylabel("trainable parameters (millions)")
ax.set_ylim(0, 5.3)
ax.set_xlabel("FG-LoRA rank  (full fine-tuning = 25.42 M, off scale)")
ax.set_title("FG-LoRA parameter budget, VideoMamba-S", fontsize=9)
ax.legend(fontsize=7, ncol=2, loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.subplots_adjust(left=0.11, right=0.98, top=0.90, bottom=0.17)
fig.savefig(OUT)
print("wrote", OUT)
