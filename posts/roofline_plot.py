
# /// script
# dependencies = [
#   "matplotlib",
#   "numpy",
# ]
# ///

import matplotlib.pyplot as plt
import numpy as np

# Hardware specifications
peak_compute = 26.7 * 1000  # Convert to GFLOPS
peak_bandwidth = 360  # GB/s
critical_intensity = peak_compute / peak_bandwidth

# Kernel data points
kernels = [
    {"name": "Naive 3×3", "intensity": 0.25, "performance": 90, "color": "#f39c12"},
    {"name": "Shared 3×3", "intensity": 0.45, "performance": 162, "color": "#27ae60"}
]

# Generate roofline data
def generate_roofline_data():
    # Memory-bound region
    memory_intensities = np.arange(0.01, critical_intensity, 0.05)
    memory_performance = memory_intensities * peak_bandwidth

    # Compute-bound region
    compute_intensities = np.arange(critical_intensity, 1000, 5)
    compute_performance = np.full_like(compute_intensities, peak_compute)

    return memory_intensities, memory_performance, compute_intensities, compute_performance

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6.5))

# Explicitly set both axes to log_10 scale
ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)

# Generate roofline data
mem_i, mem_p, comp_i, comp_p = generate_roofline_data()

# Plot roofline curves
ax.loglog(mem_i, mem_p, color="#e74c3c", linewidth=3, label="Memory-bound")
ax.loglog(comp_i, comp_p, color="#3498db", linewidth=3, label="Compute-bound")

# Guide lines
ax.axhline(y=peak_compute, color="#3498db", linestyle="--", alpha=0.7, linewidth=1)
ax.axvline(x=critical_intensity, color="#999", linestyle="--", alpha=0.7, linewidth=1)

# Plot kernel points
for kernel in kernels:
    ax.loglog(kernel["intensity"], kernel["performance"],
             'o', color=kernel["color"], markersize=8,
             markeredgecolor="#333", markeredgewidth=2)

    # Add kernel labels with better positioning to avoid overlap
    if kernel["name"] == "Naive 3×3":
        offset_x = kernel["intensity"] * 0.7  # Move left
        offset_y = kernel["performance"] * 0.65  # Move down
    else:  # Shared 3×3
        offset_x = kernel["intensity"] * 1.4  # Move right
        offset_y = kernel["performance"] * 1.3  # Move up

    ax.annotate(kernel["name"],
               (kernel["intensity"], kernel["performance"]),
               xytext=(offset_x, offset_y),
               fontsize=11, fontweight="bold", ha="center",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Region labels - move Memory-bound to center and away from kernel dots
ax.text(1.5, peak_bandwidth * 1.5, "Memory-bound",
        color="#e74c3c", fontsize=14, fontweight="bold", ha="center")
ax.text(200, peak_compute * 0.8, "Compute-bound",
        color="#3498db", fontsize=14, fontweight="bold")

# Critical intensity annotation
ax.text(critical_intensity * 1.3, peak_compute * 1.1,
        f"I_crit = {critical_intensity:.1f}",
        color="#f39c12", fontsize=12, fontweight="bold")

# Formatting
ax.set_xlim(0.1, 1000)
ax.set_ylim(10, 30000)
ax.set_xlabel("Operational Intensity (FLOP/byte) - Log_10 Scale", fontsize=12)
ax.set_ylabel("Performance (GFLOP/s) - Log_10 Scale", fontsize=12)
ax.grid(True, alpha=0.3)

# Create custom legend for kernels
legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=k["color"], markersize=8,
                             label=k["name"], markeredgecolor="#333")
                  for k in kernels]
ax.legend(handles=legend_elements, loc="lower right")

plt.tight_layout()
plt.show()
plt.savefig('./posts/mojo_gpu_puzzles/p14_roofline_naive_and_shared.png')

# Print performance metrics
print("Performance Analysis:")
print("-" * 40)
for kernel in kernels:
    efficiency = (kernel["performance"] / peak_compute) * 100
    print(f"{kernel['name']}:")
    print(f"  Intensity: {kernel['intensity']} FLOP/byte")
    print(f"  Performance: {kernel['performance']} GFLOP/s")
    print(f"  Efficiency: {efficiency:.1f}% of peak")
    print()
