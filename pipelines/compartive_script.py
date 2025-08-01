import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Configuration: replace with your actual filenames ----------
json_files = [
    "pipelines/corner_rrt_results/corner_rrt_results.json",
    "pipelines/edge_sampling_rrt/edge_sampling_rrt.json",
    "pipelines/standard_rrt_results/standard_rrt_results.json",
]
labels = ["path_length", "path_length", "path_length"]  # or derive from filenames
# ---------------------------------------------------------------------

def load_and_compute_avg_path_length(fname, only_success=True):
    with open(fname, "r") as f:
        data = json.load(f)
    path_lengths = []
    for entry in data:
        if only_success and not entry.get("success", False):
            continue
        if "path_length" in entry:
            path_lengths.append(entry["path_length"])
    if not path_lengths:
        return float("nan")  # no valid data
    return np.mean(path_lengths)

# Collect averages
avg_lengths = []
for fname in json_files:
    if not Path(fname).exists():
        raise FileNotFoundError(f"JSON file not found: {fname}")
    avg = load_and_compute_avg_path_length(fname, only_success=True)
    avg_lengths.append(avg)
    print(f"{fname}: average path_length (successful runs) = {avg:.2f}")

# Comparative bar plot
fig, ax = plt.subplots()
x = np.arange(len(labels))
ax.bar(x, avg_lengths)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Average Path Length")
ax.set_title("Comparison of Average Path Length Across JSONs")
for i, v in enumerate(avg_lengths):
    ax.text(i, v + 0.5, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.show()
