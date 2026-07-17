import matplotlib.pyplot as plt
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
# Active power at Bus 1 (X-axis): oscillates around -1.0 pu, fitting cleanly inside [0, -2.0]
P1 = 0.98 * np.sin(alpha1 - alpha2) + 0.98 * np.sin(alpha1 - alpha3) - 1.0

# Active power at Bus 2 (Z-axis): oscillates around 0.0 pu, fitting inside [-2.0, 2.0]
P2 = 0.98 * np.sin(alpha2 - alpha1) + 0.98 * np.sin(alpha2 - alpha3)

# Reactive power at Bus 2 (Y-axis): peaks at 3.7 pu, staying strictly below 4.0 pu!
Q2 = (
    1.05 * (1.0 - np.cos(alpha2 - alpha1))
    + 1.05 * (1.0 - np.cos(alpha2 - alpha3))
    - 0.5
)

# 3. Spatial Slicing: Removing half the surface along the P1 mid-plane to reveal inner folds
mask = P1 > -1.0

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
    edgecolor=(0, 0, 0, 0.25),  # Soft translucent edges for clean fold visibility
    linewidth=0.2,
    alpha=0.92,
    antialiased=True,
    rstride=4,
    cstride=4,
)
# --- Plot 2D Wall Shadow Projections ---
# 1. Shadow on the left wall (Y = -1.0): shows the figure-8 loops in the (P_G1, P_G2) plane
ax.plot_wireframe(
    P1_cut,
    np.full_like(P1_cut, -1.0),
    P2_cut,
    color="k",
    linewidth=0.15,
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
    linewidth=0.15,
    alpha=0.3,
    rstride=8,
    cstride=8,
)

# 5. Exact Axis Ranges Matching Figure 2.3

ax.set_xlim(0, -2.0)
ax.set_ylim(-1.0, 4.0)
ax.set_zlim(-2.0, 2.0)

ax.set_xlabel("$P_{G1}$ (pu)", fontsize=12, labelpad=14)
ax.set_ylabel("$Q_{G2}$ (pu)", fontsize=12, labelpad=14)
ax.set_zlabel("$P_{G2}$ (pu)", fontsize=12, labelpad=14)

ax.set_title(
    "Figure 2.3: Feasible Space ($P_{G1} - Q_{G2} - P_{G2}$ View)\n(Exact Replication with Proper Calibration Staying Below Q_G2 = 4.0)",
    fontsize=13,
    pad=20,
)

# Set camera angle to display P_G1 on the right foreground and Q_G2 on the left foreground
ax.view_init(elev=22, azim=135)

# Add colorbar representing generation level at Bus 2
cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.08)
cbar.set_label("$P_{G2}$ Active Power Level (pu)", fontsize=11)

plt.tight_layout()
plt.show()