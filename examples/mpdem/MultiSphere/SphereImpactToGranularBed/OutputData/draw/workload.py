import numpy as np
import matplotlib.pyplot as plt

# --- raw timing data (same unit) ---
data = {
    "Broad-phase\ncontact detection": 0.12,
    "Update potential\ncontact list": 0.009,
    "Narrow-phase\ncontact detection": 4.5,
    "P2G": 108.0,
    "G2P": 54.0,
    "Stress update": 131.0,
    "Nodal update": 5.2,
}

labels = list(data.keys())
times = np.array(list(data.values()), dtype=float)

# --- convert to percentage ---
total = times.sum()
perc = times / total * 100.0

# --- plot (style close to the sample) ---
plt.figure(figsize=(10.5, 3.8), dpi=150)
x = np.arange(len(labels))

bars = plt.bar(
    x, perc, width=0.62, color="#8FB6D8", edgecolor="#222222", linewidth=1.1  # light blue  # black-ish border
)

ax = plt.gca()
ax.set_ylabel("Proportion of runtime (%)", fontsize=12)
ax.set_xlabel("Workflow stage", fontsize=12)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)

# y-axis range and ticks similar to the example
ax.set_ylim(0, max(perc) * 1.15)
ax.yaxis.set_major_locator(plt.MaxNLocator(10))

# mimic "paper" style axes
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)
ax.tick_params(axis="both", which="major", width=1.2, length=5)

plt.tight_layout()

plt.savefig("workload" + ".svg")
plt.close()

# Optional: print percentages for verification
print("Total time:", total)
for k, p in zip(labels, perc):
    print(f"{k.replace(chr(10),' '):30s}: {p:6.2f}%")
