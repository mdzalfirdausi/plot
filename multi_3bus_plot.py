import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ==========================================
# CELL: All 20 Unique 3D Projections (6C3 in a 5x4 Grid)
# ==========================================

# 1. Parameter Calibration for the Hiskens & Davy [100] 3-Bus System
grid_res = 180  # 180x180 resolution ensures fast CPU execution on your local laptop
alpha_vals = np.linspace(-np.pi, np.pi, grid_res)
alpha1, alpha2 = np.meshgrid(alpha_vals, alpha_vals)
alpha3 = 0.0  # Reference slack bus angle

# 2. Forward AC Power Flow Equations (All 6 Variables Defined)
P1 = 0.98 * np.sin(alpha1 - alpha2) + 0.98 * np.sin(alpha1 - alpha3) + 0.0
P2 = 0.98 * np.sin(alpha2 - alpha1) + 0.98 * np.sin(alpha2 - alpha3)
P3 = 0.98 * np.sin(alpha3 - alpha1) + 0.98 * np.sin(alpha3 - alpha2)

Q1 = (
    1.05 * (1.0 - np.cos(alpha1 - alpha2))
    + 1.05 * (1.0 - np.cos(alpha1 - alpha3))
    - 0.5
)
Q2 = (
    1.05 * (1.0 - np.cos(alpha2 - alpha1))
    + 1.05 * (1.0 - np.cos(alpha2 - alpha3))
    - 0.5
)
Q3 = (
    1.05 * (1.0 - np.cos(alpha3 - alpha1))
    + 1.05 * (1.0 - np.cos(alpha3 - alpha2))
    - 0.5
)

variables = {
    "$P_{G1}$ (pu)": P1,
    "$Q_{G1}$ (pu)": Q1,
    "$P_{G2}$ (pu)": P2,
    "$Q_{G2}$ (pu)": Q2,
    "$P_{G3}$ (pu)": P3,
    "$Q_{G3}$ (pu)": Q3,
}

# Generate all 6C3 = 20 unique triplets of (X, Y, Z) variables
combinations = list(itertools.combinations(variables.keys(), 3))

# 3. Setup 5x4 Figure Canvas for all 20 unique 3D plots
fig, axes = plt.subplots(
    5, 4, figsize=(20, 24), subplot_kw={"projection": "3d"}, dpi=130
)
axes_flat = axes.flatten()


# Custom formatter: removes '.0' from integers while keeping decimal steps
def clean_ticks(x, pos):
    if abs(x - round(x)) < 1e-5:
        return f"{int(round(x))}"
    return f"{x:.1f}"


# 4. Loop Through All 20 Triplets and Render 3D Surfaces
for i, (label_x, label_y, label_z) in enumerate(combinations):
    ax = axes_flat[i]
    X_data = variables[label_x]
    Y_data = variables[label_y]
    Z_data = variables[label_z]

    # Plot 3D surface manifold for this exact variable triplet
    surf = ax.plot_surface(
        X_data,
        Y_data,
        Z_data,
        cmap="jet",
        edgecolor=(0, 0, 0, 0.15),
        linewidth=0.35,
        alpha=0.85,
        antialiased=True,
        rstride=6,  # Stride of 6 keeps 20 simultaneous 3D subplots responsive & fast
        cstride=6,
    )

    # Set consistent 3D viewing angle
    ax.view_init(elev=22, azim=40)

    # Apply clean tick formatting and limit tick count
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_major_locator(ticker.MaxNLocator(4))
        axis.set_major_formatter(ticker.FuncFormatter(clean_ticks))

    ax.set_xlabel(label_x, fontsize=10, labelpad=4)
    ax.set_ylabel(label_y, fontsize=10, labelpad=4)
    ax.set_zlabel(label_z, fontsize=10, labelpad=2)

# 5. Global Formatting and Export
cbar = fig.colorbar(
    surf, ax=axes, orientation="horizontal", shrink=0.35, aspect=30, pad=0.03
)
cbar.set_label("Z-Axis Variable Magnitude (pu)", fontsize=13)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(
    "ACOPF_3D_manifolds_6C3.pdf",
    format="pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.4,
)
plt.show()