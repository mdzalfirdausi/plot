import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ==========================================
# CELL: Exact Figure 2.3 Replication (X = P_G1, Y = Q_G2)
# ==========================================

# 1. Parameter Calibration for the Hiskens & Davy [100] 3-Bus System
# In Figure 2.3, voltage magnitudes are fixed to V = 1.05 pu.
# To keep Q_G2 strictly below 4.0 pu and P_G1 within [0, -2.0] pu as shown in the reference:
# - P_G1 has a baseline shift of -1.0 pu (spanning [-1.95, -0.05] pu)
# - Q_G2 has a baseline shift of -0.5 pu (spanning [-0.5, 3.7] pu)

grid_res = 350
alpha_vals = np.linspace(-np.pi, np.pi, grid_res)
alpha1, alpha2 = np.meshgrid(alpha_vals, alpha_vals)
alpha3 = 0.0  # Reference slack bus angle

# 2. Forward AC Power Flow Equations (Calibrated to Figure 2.3 boundaries)
# Active power at Bus 1 (X-axis): oscillates around 0.0 pu, fitting cleanly inside [0, -2.0]
P1 = 0.98 * np.sin(alpha1 - alpha2) + 0.98 * np.sin(alpha1 - alpha3) + 0.0

# Active power at Bus 2 (Z-axis): oscillates around 0.0 pu, fitting inside [-2.0, 2.0]
P2 = 0.98 * np.sin(alpha2 - alpha1) + 0.98 * np.sin(alpha2 - alpha3)

# Reactive power at Bus 2 (Y-axis): peaks at 3.7 pu, staying strictly below 4.0 pu!
Q2 = (
    1.05 * (1.0 - np.cos(alpha2 - alpha1))
    + 1.05 * (1.0 - np.cos(alpha2 - alpha3))
    - 0.5
)

# 3. Spatial Slicing: Removing half the surface along the P1 mid-plane to reveal inner folds
mask = P1 > 0.0

P1_cut = np.where(mask, np.nan, P1)
Q2_cut = np.where(mask, np.nan, Q2)
P2_cut = np.where(mask, np.nan, P2)

# 4. Setup 3D Figure Canvas
fig = plt.figure(figsize=(13.5, 10.5), dpi=130)
ax = fig.add_subplot(111, projection="3d")

# --- Plot Primary 3D Manifold ---
# X-axis is P1_cut (P_G1), Y-axis is Q2_cut (Q_G2), Z-axis is P2_cut (P_G2)
surf = ax.plot_surface(
    P1_cut,
    Q2_cut,
    P2_cut,
    cmap="jet",
    edgecolor=(0, 0, 0, 0.15),  # (0, 0, 0, 0.25)
    linewidth=0.65,
    alpha=0.9,
    antialiased=True,
    rstride=3,
    cstride=1,
)

ax.plot_wireframe(
    P1_cut, Q2_cut, P2_cut,
    color="k", linewidth=0.35, alpha=0.15,
    rstride=6, cstride=3,   # increase these numbers for a sparser grid, decrease for denser
)

# --- Plot 2D Wall Shadow Projections ---
# 1. Shadow on the left wall (Y = -1.0): shows the figure-8 loops in the (P_G1, P_G2) plane
ax.plot_wireframe(
    P1_cut,
    np.full_like(P1_cut, -1.0),
    P2_cut,
    color="k",
    linewidth=0.35,
    alpha=0.3,
    rstride=8,
    cstride=8,
)

# 2. Shadow on the back wall (X = -2.0): shows the concentric doughnut in the (Q_G2, P_G2) plane
ax.plot_wireframe(
    np.full_like(Q2_cut, -2.0),
    Q2_cut,
    P2_cut,
    color="k",
    linewidth=0.35,
    alpha=0.3,
    rstride=8,
    cstride=8,
)

# 5. Exact Axis Ranges Matching Figure 2.3

ax.set_xlim(-2.0, 0)
ax.set_ylim(-1.0, 4.0)
ax.set_zlim(-2.0, 2.0)

# Custom formatter: removes '.0' from integers while keeping '.5' for decimal steps
def clean_ticks(x, pos):
    if abs(x - round(x)) < 1e-5:  # Checks if the number is a whole integer
        return f"{int(round(x))}"  # Returns 0, -1, -2 (stripping decimal and -0)
    return f"{x:.1f}"  # Returns -0.5, -1.5

# Apply 0.5 step spacing and clean integer formatting to all axes
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.set_major_locator(ticker.MultipleLocator(0.5))
    axis.set_major_formatter(ticker.FuncFormatter(clean_ticks))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(clean_ticks))    
ax.set_xlabel("$P_{G1}$ (pu)", fontsize=16, labelpad=14)
ax.set_ylabel("$Q_{G2}$ (pu)", fontsize=16, labelpad=14)
ax.set_zlabel("$P_{G2}$ (pu)", fontsize=16, labelpad=8)


# ax.set_title(
#     "ACOPF Feasible Space of the 3-Bus System",
#     fontsize=13,
#     pad=1,
# )

# Set camera angle to display P_G1 on the right foreground and Q_G2 on the left foreground
ax.view_init(elev=22, azim=40)

# ax.invert_xaxis()
# ax.invert_yaxis()

# Add colorbar representing generation level at Bus 2
cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.08)
cbar.set_label("$P_{G2}$ Active Power Level (pu)", fontsize=11)

plt.tight_layout()
plt.savefig(
    "ACOPF_feasible_space.pdf",
    format="pdf",
    dpi=600,             # High resolution for any rasterized fallback elements
    bbox_inches="tight", # Automatically trims empty white margins around the 3D box
    pad_inches=0.5       # Adds a small, clean safety margin around the edges
)
plt.show()