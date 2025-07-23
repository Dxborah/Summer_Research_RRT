import numpy as np
import matplotlib.pyplot as plt
from scripts import run_rrt_v1, run_rrt_v2


seeds = [42, 7, 1337, 2024, 1001]
results_v1 = []
results_v2 = []

for seed in seeds:
    steps1, time1, success1 = run_rrt_v1(seed)
    steps2, time2, success2 = run_rrt_v2(seed)

    results_v1.append((steps1, time1, success1))
    results_v2.append((steps2, time2, success2))

# Print summary
print("\n📘 Summary for RRT v1:")
for i, (steps, time, success) in enumerate(results_v1):
    print(f"Seed {seeds[i]}: Steps = {steps}, Time = {time:.4f}s, Success = {success}")

print("\n📗 Summary for RRT v2:")
for i, (steps, time, success) in enumerate(results_v2):
    print(f"Seed {seeds[i]}: Steps = {steps}, Time = {time:.4f}s, Success = {success}")

# Optional: plot results
steps_v1 = [s for s, _, _ in results_v1]
steps_v2 = [s for s, _, _ in results_v2]
times_v1 = [t for _, t, _ in results_v1]
times_v2 = [t for _, t, _ in results_v2]

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(seeds, steps_v1, label='v1 Steps', marker='o')
plt.plot(seeds, steps_v2, label='v2 Steps', marker='o')
plt.xlabel("Seed"); plt.ylabel("Steps"); plt.legend(); plt.title("RRT Steps")

plt.subplot(1, 2, 2)
plt.plot(seeds, times_v1, label='v1 Time', marker='o')
plt.plot(seeds, times_v2, label='v2 Time', marker='o')
plt.xlabel("Seed"); plt.ylabel("Time (s)"); plt.legend(); plt.title("RRT Runtime")

plt.tight_layout()
plt.show()
