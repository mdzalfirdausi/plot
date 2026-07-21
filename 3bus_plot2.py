import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# Figure 2.3 Replication (P_G1 - Q_G2 - P_G2 view)
# Axis orientation matched to the published figure:
#   - P_G2 (vertical) labeled on the left edge
#   - Q_G2 running along the bottom-left, increasing toward the back-left
#   - P_G1 running along the bottom-right, increasing (toward 0) toward the front
# ==========================================

grid_res = 350
alpha_vals = np.linspace(-np.pi, np.pi, grid_res)
alpha1, alpha2 = np.meshgrid(alpha_vals, alpha_vals)
alpha3 = 0.0  # reference slack bus angle

# Forward AC power flow equations (calibrated to Figure 2.3 boundaries)
P1 = 0.98 * np.sin(alpha1 - alpha2) + 0.98 * np.sin(alpha1 - alpha3) - 1.0
P2 = 0.98 * np.sin(alpha2 - alpha1) + 0.98 * np.sin(alpha2 - alpha3)
Q2 = (
    1.05 * (1.0 - np.cos(alpha2 - alpha1))
    + 1.05 * (1.0 - np.cos(alpha2 - alpha3))
    - 0.5
)

# Spatial slicing: remove half the surface along the P1 mid-plane to reveal inner folds
mask = P1 > -1.0
P1_cut = np.where(mask, np.nan, P1)
Q2_cut = np.where(mask, np.nan, Q2)
P2_cut = np.where(mask, np.nan, P2)

fig = plt.figure(figsize=(13.5, 10.5), dpi=130)
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    P1_cut, Q2_cut, P2_cut,
    cmap="jet",
    edgecolor=(0, 0, 0, 0.25),
    linewidth=0.2,
    alpha=0.92,
    antialiased=True,
    rstride=4, cstride=4,
)

# Wall shadow projections
ax.plot_wireframe(P1_cut, np.full_like(P1_cut, -1.0), P2_cut,
                   color="k", linewidth=0.15, alpha=0.3, rstride=8, cstride=8)
ax.plot_wireframe(np.full_like(Q2_cut, -2.0), Q2_cut, P2_cut,
                   color="k", linewidth=0.15, alpha=0.3, rstride=8, cstride=8)

ax.set_xlim(0, -2.0)
ax.set_ylim(-1.0, 4.0)
ax.set_zlim(-2.0, 2.0)

ax.set_xlabel("$P_{G1}$ (pu)", fontsize=12, labelpad=14)
ax.set_ylabel("$Q_{G2}$ (pu)", fontsize=12, labelpad=14)
ax.set_zlabel("$P_{G2}$ (pu)", fontsize=12, labelpad=14)

ax.set_title(
    "Figure 2.3: Feasible Space ($P_{G1}$-$Q_{G2}$-$P_{G2}$ View)",
    fontsize=13, pad=20,
)

# View angle matched to the published figure's axis layout:
# P_G2 vertical axis on the left, Q_G2 bottom-left, P_G1 bottom-right
ax.view_init(elev=22, azim=-55)

cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.08)
cbar.set_label("$P_{G2}$ Active Power Level (pu)", fontsize=11)

plt.tight_layout()
plt.show()