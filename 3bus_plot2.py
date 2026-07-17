import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# CELL: Corrected Figure 13 (True Spatial Cut in Power Space)
# ==========================================

# 1. Effective Network Parameters (Consistent with Fig 7 - 12 calibration)
B12 = 1.10  # Effective branch susceptance Gen1 - Gen2 (pu)
B13 = 1.12  # Effective branch susceptance Gen1 - Gen3 (pu)
B23 = 1.12  # Effective branch susceptance Gen2 - Gen3 (pu)
alpha3 = 0.0  # Reference slack bus angle (rad)

# 2. Define Dense State Grid over Full Angle Space [-pi, pi]
grid_res = 350
alpha_vals = np.linspace(-np.pi, np.pi, grid_res)
alpha1, alpha2 = np.meshgrid(alpha_vals, alpha_vals)

# 3. Forward AC Power Flow Equations
# Active power injection at Gen1 (P1) - The horizontal X-axis
P1 = B12 * np.sin(alpha1 - alpha2) + B13 * np.sin(alpha1 - alpha3)

# Active power injection at Gen2 (P2) - The vertical Z-axis
P2 = B12 * np.sin(alpha2 - alpha1) + B23 * np.sin(alpha2 - alpha3)

# Reactive power injection at Gen2 (Q2) - The depth Y-axis
Q2 = (
    (1.0 - np.cos(alpha2 - alpha1)) * B12
    + (1.0 - np.cos(alpha2 - alpha3)) * B23
    + 0.82
)

# 4. CRITICAL FIX: Spatial Slicing in Power Space (P1) instead of Angle Space (alpha1)
# Removing the front half of the 3D volume (P1 > 0.05) cleanly slices open the outer
# maximum loadability envelope while leaving ALL internal folding layers intact!
mask = P1 > 0.05

P1_cut = np.where(mask, np.nan, P1)
Q2_cut = np.where(mask, np.nan, Q2)
P2_cut = np.where(mask, np.nan, P2)

# 5. Setup 3D Figure Canvas
fig = plt.figure(figsize=(13.5, 10.5), dpi=130)
ax = fig.add_subplot(111, projection="3d")

# --- Plot Primary 3D Manifold with Molzahn's 'Jet' Colormap ---
surf = ax.plot_surface(
    P1_cut,
    Q2_cut,
    P2_cut,
    cmap="jet",
    edgecolor="k",
    linewidth=0.1,
    alpha=0.95,
    antialiased=True,
    rstride=2,
    cstride=2,
)

# --- Plot 2D Wall Shadow Projections (Section V-B3) ---
# 1. P1 - P2 projection on the back wall (fixing Q2 at back boundary: 3.8 pu)
ax.plot_wireframe(
    P1_cut,
    np.full_like(P1_cut, 3.8),
    P2_cut,
    color="gray",
    linewidth=0.15,
    alpha=0.25,
    rstride=4,
    cstride=4,
)

# 2. Q2 - P2 projection on the left wall (fixing P1 at back boundary: -2.5 pu)
ax.plot_wireframe(
    np.full_like(Q2_cut, -2.5),
    Q2_cut,
    P2_cut,
    color="gray",
    linewidth=0.15,
    alpha=0.25,
    rstride=4,
    cstride=4,
)

# 6. Exact Formatting & Scaling
ax.set_xlim(0, -2.0)
ax.set_ylim(-1.0, 4.0)
ax.set_zlim(-2.0, 2.0)

ax.set_xlabel("$P_{G1}$ (pu)", fontsize=12, labelpad=14)
ax.set_ylabel("$Q_{G2}$ (pu)", fontsize=12, labelpad=14)
ax.set_zlabel("$P_{G2}$ (pu)", fontsize=12, labelpad=14)

ax.set_title(
    "Fig. 13. Power Flow Solution Space ($P_1 - Q_2 - P_2$ View)\n(True Spatial Cut Exposing Internal Folds and Toroidal Void)",
    fontsize=13,
    pad=20,
)

# Set camera angle to look directly into the open cut face (Elev: 26°, Azim: -52°)
ax.view_init(elev=20, azim=170)

# Add colorbar representing generation level at Bus 2
cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.08)
cbar.set_label("$P_{G2}$ Active Power Level (pu)", fontsize=11)

plt.tight_layout()
plt.show()