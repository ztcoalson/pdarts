import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn import functional as F

USE_REDUCE = 1

# Set Seaborn style
sns.set(style="whitegrid")
sns.set_palette("colorblind")
plt.rcParams.update({'font.size': 14})

# Load baseline gradients
baseline_grads = torch.load('/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/pdarts/results/search-clean_grads-20250120-184932/arch_grads.pt')

# Load other gradient histories
grad_variants = {
    "Clean": torch.load('/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/pdarts/results/search-clean_fun-20250121-192336/arch_grads.pt'),
    "GC 50%": torch.load('/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/pdarts/results/search-gc_grads2-50.0%-20250124-223539/arch_grads.pt'),
    "Noise 50%": torch.load('/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/pdarts/results/search-noise_grads-50.0%-20250120-224244/arch_grads.pt'),
    "RLF 50%": torch.load('/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/pdarts/results/search-rlf_grads-50.0%-20250120-185011/arch_grads.pt'),
    "CLF 50%": torch.load('/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/pdarts/results/search-clf_grads-50.0%-20250124-175005/arch_grads.pt')
}

# Compute cosine similarity for each variant
cosims = {}
for name, grads in grad_variants.items():
    cosims[name] = [
        F.cosine_similarity(b[USE_REDUCE].flatten(), g[USE_REDUCE].flatten(), dim=0).item()
        for b, g in zip(baseline_grads, grads) if not torch.all(b[USE_REDUCE] == 0)
    ]

# Prepare data for Seaborn
epochs = range(1, len(cosims['Clean']) + 1)
data = []
for name, distances in cosims.items():
    for epoch, distance in zip(epochs, distances):
        data.append({"Epoch": epoch, "Cosine Similarity": distance, "Variant": name})

import pandas as pd
df = pd.DataFrame(data)

# Plot with Seaborn
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="Epoch", y="Cosine Similarity", hue="Variant", marker="o")

# Customize the plot
plt.xlabel("Architecture Training Epoch")
plt.ylabel("Cosine Similarity with Clean Gradients")
plt.xticks([0, 10, 20, 30, 40])
plt.yticks(np.arange(-0.2, 1.1, 0.2))
plt.xlim(0, 46)
plt.ylim(-0.2, 1.0)
plt.legend(loc="lower left")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(f"grads_comparison_{'reduce' if USE_REDUCE else 'normal'}.png", dpi=300)
plt.show()
