#!/usr/bin/env python3
"""
WB5 Five-Bus OPF Feasible Space Computation
Reproduces Figure 3 from Molzahn (2017) "Computing the Feasible Spaces of Optimal Power Flow Problems"

The approach discretizes the control variables (P_G5, V_G5, V_G1) and solves the
power flow equations at each grid point, then filters solutions that satisfy
all OPF constraints to map out the complete feasible space.
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# WB5 System Data (per unit, 100 MVA base)
# =============================================================================
# Bus layout (0-indexed):
#   Bus 0  →  Slack / Generator G1
#   Bus 1  →  PQ load  (1.30 pu real, 0.20 pu reactive)
#   Bus 2  →  PQ load  (1.30 pu real, 0.20 pu reactive)
#   Bus 3  →  PQ load  (0.65 pu real, 0.10 pu reactive)
#   Bus 4  →  PV  / Generator G5  (no load)

n = 5

# Branches: (from_bus, to_bus, R_pu, X_pu, B_total_pu)
branch_data = [
    (0, 1, 0.04, 0.09, 0.00),
    (0, 2, 0.05, 0.10, 0.00),
    (1, 3, 0.55, 0.90, 0.45),
    (2, 4, 0.55, 0.90, 0.45),
    (3, 4, 0.06, 0.10, 0.00),
    (1, 2, 0.07, 0.09, 0.00),
]

# Build Y-bus
Y = np.zeros((n, n), dtype=complex)
for (i, j, r, x, bc) in branch_data:
    ys = 1.0 / (r + 1j * x)      # series admittance
    ysh = 1j * bc / 2.0           # half-line charging susceptance
    Y[i, i] += ys + ysh
    Y[j, j] += ys + ysh
    Y[i, j] -= ys
    Y[j, i] -= ys

G, B = Y.real, Y.imag

# Load demands (pu)
Pd = np.array([0.00, 1.30, 1.30, 0.65, 0.00])
Qd = np.array([0.00, 0.20, 0.20, 0.10, 0.00])

# Bounds
V_MIN, V_MAX = 0.95, 1.05
QG_MIN, QG_MAX = -0.30, 18.0

# Known OPF solutions (from Molzahn 2017)
GLOBAL_SOL = (1.81, 2.21, -0.30)   # (P_G1, P_G5, Q_G5) pu
LOCAL_SOL = (2.46, 0.98, -0.30)

# =============================================================================
# Power Flow Solver
# =============================================================================
def bus_powers(Vm, theta):
    """Calculate bus active and reactive power injections."""
    d = theta[:, None] - theta[None, :]
    P = (Vm[:, None] * Vm[None, :] * (G * np.cos(d) + B * np.sin(d))).sum(axis=1)
    Q = (Vm[:, None] * Vm[None, :] * (G * np.sin(d) - B * np.cos(d))).sum(axis=1)
    return P, Q


def solve_pf(P_G5, V_G1, V_G5, x0):
    """
    Newton-Raphson power flow for WB5.
    
    Parameters:
    -----------
    P_G5 : float — active power injection at bus 4 (G5), pu
    V_G1 : float — voltage magnitude at slack bus (bus 0), pu
    V_G5 : float — voltage magnitude at PV bus (bus 4), pu
    x0   : array — initial guess [theta1, theta2, theta3, theta4, V1, V2, V3]
    
    Returns:
    --------
    dict or None — solution with P_G1, Q_G1, Q_G5, Vm if converged
    """
    # Net power specs at non-slack buses
    P_spec = np.array([-Pd[1], -Pd[2], -Pd[3], P_G5])
    Q_spec = np.array([-Qd[1], -Qd[2], -Qd[3]])
    
    def residuals(x):
        theta = np.array([0., x[0], x[1], x[2], x[3]])
        Vm = np.array([V_G1, x[4], x[5], x[6], V_G5])
        P, Q = bus_powers(Vm, theta)
        return [
            P_spec[0] - P[1], P_spec[1] - P[2],
            P_spec[2] - P[3], P_spec[3] - P[4],
            Q_spec[0] - Q[1], Q_spec[1] - Q[2], Q_spec[2] - Q[3],
        ]
    
    sol, info, ier, _ = fsolve(residuals, x0, full_output=True, xtol=1e-10)
    if ier != 1 or np.max(np.abs(info['fvec'])) > 1e-6:
        return None
    
    theta = np.array([0., sol[0], sol[1], sol[2], sol[3]])
    Vm = np.array([V_G1, sol[4], sol[5], sol[6], V_G5])
    P, Q = bus_powers(Vm, theta)
    
    return {
        'P_G1': P[0], 'Q_G1': Q[0], 'Q_G5': Q[4],
        'Vm': Vm, 'theta': theta
    }


# =============================================================================
# Grid Sweep
# =============================================================================
# Discretization parameters as described in the paper
# ΔP = 1 MW = 0.01 pu, ΔV = 0.001 pu  (finer grid for better visualization)

P5_vals = np.arange(0.0, 4.01, 0.02)      # 201 values
V5_vals = np.arange(0.95, 1.051, 0.002)   # 51 values
V1_vals = np.arange(0.95, 1.051, 0.002)   # 51 values

print(f"Grid: {len(P5_vals)} × {len(V5_vals)} × {len(V1_vals)} = "
      f"{len(P5_vals) * len(V5_vals) * len(V1_vals):,} points")

# Multiple starting points to capture disconnected components
X0_LIST = [
    [0.0, 0.0, 0.0, 0.0, 1.00, 1.00, 1.00],   # flat start
    [-0.1, -0.2, -0.3, -0.4, 0.97, 0.97, 0.97],
    [0.1, 0.2, 0.1, 0.2, 1.02, 1.02, 1.02],
    [0.0, 0.0, 0.0, -1.0, 1.00, 1.00, 1.00],
    [0.0, 0.0, 0.0, 1.0, 1.00, 1.00, 1.00],
    [-0.5, -0.5, -0.3, -1.5, 0.95, 0.95, 0.95],
    [0.5, 0.5, 0.3, 0.5, 1.05, 1.05, 1.05],
    [-0.2, -0.3, -0.1, -2.0, 0.96, 0.96, 0.96],
    [0.2, 0.3, 0.2, 1.5, 1.04, 1.04, 1.04],
]

feasible_points = []

print("Starting grid sweep...")
total = len(P5_vals) * len(V5_vals) * len(V1_vals)
count = 0

for i, P5 in enumerate(P5_vals):
    for V5 in V5_vals:
        for V1 in V1_vals:
            count += 1
            if count % 5000 == 0:
                print(f"  Progress: {count:,} / {total:,} "
                      f"({100*count/total:.1f}%) — {len(feasible_points):,} feasible")

            seen = set()
            for x0 in X0_LIST:
                res = solve_pf(P5, V1, V5, x0)
                if res is None:
                    continue
                
                P1, Q1, Q5 = res['P_G1'], res['Q_G1'], res['Q_G5']
                Vm = res['Vm']
                
                # De-duplicate: same solution from different starts
                key = tuple(np.round([P1, Q1, Q5, *Vm], 3))
                if key in seen:
                    continue
                seen.add(key)
                
                # Feasibility checks (with small tolerance)
                EPS = 1e-4
                if P1 < -EPS:
                    continue
                if not (QG_MIN - EPS <= Q1 <= QG_MAX + EPS):
                    continue
                if not (QG_MIN - EPS <= Q5 <= QG_MAX + EPS):
                    continue
                if not all(V_MIN - EPS <= v <= V_MAX + EPS for v in Vm):
                    continue
                
                cost = 400.0 * P1 + 100.0 * P5
                feasible_points.append([P1, P5, Q5, cost])

print(f"\nFeasible points collected: {len(feasible_points):,}")

# Convert to numpy array
pts = np.array(feasible_points)

# =============================================================================
# Create the Plot (Figure 3 from the paper)
# =============================================================================
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')

P1 = pts[:, 0]
P5 = pts[:, 1]
Q5 = pts[:, 2]
cost = pts[:, 3]

# Sort points by cost for better color mapping
idx = np.argsort(cost)
P1_s, P5_s, Q5_s, cost_s = P1[idx], P5[idx], Q5[idx], cost[idx]

# Scatter plot of feasible space colored by cost
sc = ax.scatter(
    P5_s, P1_s, Q5_s,
    c=cost_s,
    cmap='Reds',
    norm=Normalize(vmin=cost_s.min(), vmax=cost_s.max()),
    s=3,
    alpha=0.6,
    linewidths=0,
)

# Gray plane for Q_G5 = -0.30 (reactive power lower limit)
pg5g, pg1g = np.meshgrid(
    np.linspace(-0.5, 4.5, 30),
    np.linspace(-0.5, 5.5, 30)
)
qg5g = np.full_like(pg5g, QG_MIN)
ax.plot_surface(
    pg5g, pg1g, qg5g,
    alpha=0.25,
    color='gray',
    zorder=0,
    label=r'$Q_{G5} \geq -0.30$ pu'
)

# Global and local optima (from paper)
gP1, gP5, gQ5 = GLOBAL_SOL
lP1, lP5, lQ5 = LOCAL_SOL

ax.scatter([gP5], [gP1], [gQ5],
           color='lime', s=350, marker='*',
           zorder=10, depthshade=False,
           label='Global optimum (1.81, 2.21, -0.30) pu')

ax.scatter([lP5], [lP1], [lQ5],
           color='dodgerblue', s=200, marker='v',
           zorder=10, depthshade=False,
           label='Local optimum (+14.3% cost)')

# Add text annotations for the solutions
ax.text(gP5 + 0.15, gP1 - 0.1, gQ5 - 0.05,
        'Global optimum', color='lime', fontsize=10, ha='left')
ax.text(lP5 + 0.15, lP1 + 0.1, lQ5 + 0.05,
        'Local optimum', color='dodgerblue', fontsize=10, ha='left')

# Colorbar
cbar = plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.12, aspect=20)
cbar.set_label('Generation Cost ($/hr)', fontsize=12)
cbar.ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'${x:,.0f}')
)

# Labels and formatting
ax.set_xlabel(r'$P_{G5}$ (per unit)', labelpad=12, fontsize=13)
ax.set_ylabel(r'$P_{G1}$ (per unit)', labelpad=12, fontsize=13)
ax.set_zlabel(r'$Q_{G5}$ (per unit)', labelpad=12, fontsize=13)

ax.set_title(
    'Feasible Space of the WB5 Five-Bus OPF Problem\n'
    r'Gray plane: $Q_{G5} \geq -0.30$ pu (splits space into two disconnected components)',
    fontsize=12,
    pad=18
)

# Set viewing angle to match paper figure
ax.view_init(elev=20, azim=-55)

# Axis limits
ax.set_xlim([-0.8, 4.5])
ax.set_ylim([-0.2, 5.5])
ax.set_zlim([-0.55, 0.8])

# Legend
ax.legend(loc='upper left', fontsize=10, framealpha=0.85)

plt.tight_layout()
plt.savefig('WB5_feasible_space.png', dpi=200, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'WB5_feasible_space.png'")