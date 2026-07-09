# =============================================================================
# CELL 1: Generalized System Setup using Custom MATPOWER Text Parser
# =============================================================================
import os
import re
import numpy as np

def parse_matpower_matrix(content, matrix_name):
    """Extracts a matrix from the .m file content using regex and cleans out comments."""
    pattern = rf'mpc\.{matrix_name}\s*=\s*\[(.*?)\];'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        return []
    
    matrix_str = match.group(1)
    # Remove inline comments (anything after %)
    matrix_str = re.sub(r'%.*', '', matrix_str)
    
    rows = []
    # Split by lines or semicolons to safely handle formatted .m files
    for line in matrix_str.replace(';', '\n').split('\n'):
        clean_line = line.strip()
        if clean_line:
            rows.append([float(val) for val in clean_line.split()])
            
    return np.array(rows)

def load_case_data(filepath):
    """Reads baseMVA, bus, gen, and branch data directly from a plain-text .m file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Case file not found: {filepath}")
        
    with open(filepath, 'r') as f:
        content = f.read()
        
    # 1. Extract baseMVA
    base_match = re.search(r'mpc\.baseMVA\s*=\s*([\d\.]+);', content)
    baseMVA = float(base_match.group(1)) if base_match else 100.0
    
    # 2. Extract standard matrices using your parser
    bus_data = parse_matpower_matrix(content, 'bus')
    gen_data = parse_matpower_matrix(content, 'gen')
    branch_data = parse_matpower_matrix(content, 'branch')
    
    return baseMVA, bus_data, gen_data, branch_data

# --- LOAD SPECIFIC CASE FILE ---
# Update this path if your WB5.m file is in a different folder
filepath = 'WB5.m' 
baseMVA, bus_data, gen_data, branch_data = load_case_data(filepath)

# 1. Convert MATLAB 1-based indices to Python 0-based indices
bus_data[:, 0] -= 1
gen_data[:, 0] -= 1
branch_data[:, 0] -= 1
branch_data[:, 1] -= 1

# 2. Convert physical powers (MW / MVAR) to per-unit using baseMVA
# Bus load demand (Pd, Qd -> cols 2 and 3)
bus_data[:, 2] /= baseMVA
bus_data[:, 3] /= baseMVA

# Gen dispatch and limits (Pg, Qg, Qmax, Qmin, Pmax, Pmin -> cols 1, 2, 3, 4, 8, 9)
gen_data[:, [1, 2, 3, 4, 8, 9]] /= baseMVA

# Branch thermal limits (rateA -> col 5). If 0, replace with unconstrained limit.
if branch_data.shape[1] > 5:
    branch_data[:, 5] /= baseMVA
    branch_data[branch_data[:, 5] == 0, 5] = 1e10

# 3. Build Admittance Matrix (Ybus)
n_buses = len(bus_data)
Ybus = np.zeros((n_buses, n_buses), dtype=complex)
for row in branch_data:
    f, t = int(row[0]), int(row[1])
    r, x, bc = row[2], row[3], row[4]
    y_s = 1.0 / (r + 1j * x)
    y_sh = 1j * bc / 2.0
    Ybus[f, f] += y_s + y_sh
    Ybus[t, t] += y_s + y_sh
    Ybus[f, t] -= y_s
    Ybus[t, f] -= y_s

print(f"✔ Cell 1 Executed: Read {filepath} directly into per-unit NumPy arrays.")
print(f"  System baseMVA: {baseMVA} MVA | Extracted Buses: {n_buses} | Generators: {len(gen_data)}")

# =============================================================================
# CELL 2: Step 1 - Automated Control Space Discretization (Section III-A)
# =============================================================================
import numpy as np

# 1. Automatically identify active online generators (G)
active_gens = gen_data[gen_data[:, 7] == 1]
gen_buses = active_gens[:, 0].astype(int)

# 2. Automatically select Slack Bus (S) per Footnote 3: Max (P_max - P_min)
p_ranges = active_gens[:, 8] - active_gens[:, 9] # P_max - P_min
slack_idx_in_gen = np.argmax(p_ranges)           # Index in gen_data
slack_bus = int(active_gens[slack_idx_in_gen, 0])
non_slack_gens = np.delete(active_gens, slack_idx_in_gen, axis=0)

print(f"--- Automated Topology Analysis ---")
print(f"Total Buses: {n_buses} | Generator Buses (G): {gen_buses.tolist()}")
print(f"Selected Slack Bus (S): Bus {slack_bus+1} (Max P_range = {p_ranges[slack_idx_in_gen]:.2f} pu)")
print(f"Non-Slack Generators (G \\ S): {(non_slack_gens[:, 0] + 1).astype(int).tolist()}")
print("-----------------------------------\n")

# 3. Dynamically assemble the independent control vector u and its bounds
control_names = []
u_min = []
u_max = []

# Calculate total system active power demand to set intelligent plotting windows
total_load_pu = np.sum(bus_data[:, 2])

# Rule A: Discretize active power P_Gi for all non-slack generators (i in G \ S)
for gen in non_slack_gens:
    bus_id = int(gen[0])
    control_names.append(f"P_G{bus_id+1}")
    u_min.append(gen[9]) # P_min (col 9)
    
    # EXPLORATION WINDOW OVERRIDE:
    # Structural P_max in WB5 is 50.0 pu. Sweeping up to 50.0 pu when total load is 3.25 pu
    # wastes over 90% of grid solves. We intelligently clip the search window to 4.0 pu!
    plot_p_ceiling = min(gen[8], max(4.0, total_load_pu * 1.25))
    u_max.append(plot_p_ceiling)

# Rule B: Discretize voltage magnitude V_i for ALL generators (i in G)
for gen in active_gens:
    bus_id = int(gen[0])
    bus_row = bus_data[bus_data[:, 0] == bus_id][0]
    control_names.append(f"V_G{bus_id+1}")
    
    # FIXED: In standard 13-column MATPOWER bus matrices, Vmin is col 12 and Vmax is col 11!
    # If using a compact 6-column matrix, Vmin is col 5 and Vmax is col 4.
    vmin_idx = 12 if len(bus_row) >= 13 else 5
    vmax_idx = 11 if len(bus_row) >= 13 else 4
    
    u_min.append(bus_row[vmin_idx])
    u_max.append(bus_row[vmax_idx])

u_min = np.array(u_min)
u_max = np.array(u_max)
num_controls = len(control_names)

print(f"Automated Control Vector u ({num_controls} variables extracted):")
for i in range(num_controls):
    print(f"  [{i+1}] {control_names[i]}: Search Window [{u_min[i]:.3f}, {u_max[i]:.3f}] pu")

# 4. Programmatically build the N-dimensional discretization grid
N_res = 15  # Resolution per axis

# Generate a list of 1D linear sweeps for each extracted variable
d_sweeps = [np.linspace(u_min[i], u_max[i], N_res) for i in range(num_controls)]

# Create the N-dimensional meshgrid dynamically
mesh_grids = np.meshgrid(*d_sweeps, indexing='ij')

# Flatten all grids and stack them into an (M x d) array of candidate vectors
candidate_controls = np.vstack([grid.ravel() for grid in mesh_grids]).T

print(f"\n✔ Step 1 Complete: Programmatically generated {len(candidate_controls):,} grid coordinates.")
print("\nPreview of first 5 automated candidate control vectors (u^k):")
print("-" * 65)
for k in range(5):
    formatted_vals = " | ".join([f"{control_names[i]} = {candidate_controls[k, i]:.3f}" for i in range(num_controls)])
    print(f"  Candidate {k+1:04d}: {formatted_vals}")
print("-" * 65)

# =============================================================================
# CELL 3: Step 2 - Formulating the Square Polynomial System (Section III-B)
# =============================================================================
# We create a universal function that maps 2n-2 unknown voltage states (x)
# to 2n-2 quadratic power flow residuals for any candidate control vector (u_k).

def get_state_mapping(n_buses, slack_bus):
    """Creates indexing maps to move between full 2n voltages and 2n-2 unknown states."""
    all_buses = np.arange(n_buses)
    unknown_buses = np.delete(all_buses, slack_bus)
    return unknown_buses

def unpack_voltages(state_x, u_k, n_buses, slack_bus, active_gens, control_names):
    """Reconstructs full Vd and Vq arrays (size n) from the 2n-2 unknown states and fixed u_k."""
    unknown_buses = get_state_mapping(n_buses, slack_bus)
    
    Vd = np.zeros(n_buses)
    Vq = np.zeros(n_buses)
    
    # 1. Assign the 2n-2 unknown state variables to non-slack buses
    num_unknown = len(unknown_buses)
    Vd[unknown_buses] = state_x[:num_unknown]
    Vq[unknown_buses] = state_x[num_unknown:]
    
    # 2. Assign fixed Slack Bus voltages (V_q = 0, V_d = V_slack from u_k)
    slack_control_name = f"V_G{slack_bus+1}"
    slack_idx_in_u = control_names.index(slack_control_name)
    Vd[slack_bus] = u_k[slack_idx_in_u]
    Vq[slack_bus] = 0.0
    
    return Vd, Vq

def power_flow_residuals(state_x, u_k, bus_data, gen_data, Ybus, slack_bus, active_gens, non_slack_gens, control_names):
    """
    Evaluates the 2n-2 quadratic polynomial equations (Equation 6 in Molzahn 2017).
    Returns an array of 2n-2 residuals: F(x) = 0.
    """
    n_buses = len(bus_data)
    G, B = Ybus.real, Ybus.imag
    Vd, Vq = unpack_voltages(state_x, u_k, n_buses, slack_bus, active_gens, control_names)
    
    # Calculate active (P_calc) and reactive (Q_calc) injections at all buses
    # Equations (2a) and (2b) in paper
    P_calc = np.zeros(n_buses)
    Q_calc = np.zeros(n_buses)
    for i in range(n_buses):
        for k in range(n_buses):
            P_calc[i] += Vd[i] * (G[i,k]*Vd[k] - B[i,k]*Vq[k]) + Vq[i] * (B[i,k]*Vd[k] + G[i,k]*Vq[k])
            Q_calc[i] += Vd[i] * (-B[i,k]*Vd[k] - G[i,k]*Vq[k]) + Vq[i] * (G[i,k]*Vd[k] - B[i,k]*Vq[k])
            
    residuals = []
    unknown_buses = get_state_mapping(n_buses, slack_bus)
    
    for i in unknown_buses:
        # Check if this bus is a generator
        is_gen = i in active_gens[:, 0]
        
        if is_gen:
            # RULE 1: Non-Slack Generator -> Match active power P_Gi from grid u_k (Eq 6a)
            p_control_name = f"P_G{i+1}"
            p_target = u_k[control_names.index(p_control_name)]
            residuals.append(P_calc[i] - p_target)
            
            # RULE 2: All Generators -> Match squared voltage magnitude |V_i|^2 from grid u_k (Eq 6b)
            v_control_name = f"V_G{i+1}"
            v_target = u_k[control_names.index(v_control_name)]
            residuals.append((Vd[i]**2 + Vq[i]**2) - (v_target**2))
            
        else:
            # RULE 3: PQ Load Buses -> Match fixed active and reactive load demand (Eq 6c, 6d)
            p_load = bus_data[bus_data[:, 0] == i, 2][0]
            q_load = bus_data[bus_data[:, 0] == i, 3][0]
            residuals.append(P_calc[i] - (-p_load))  # Generation minus load = 0 -> P_calc = -P_load
            residuals.append(Q_calc[i] - (-q_load))
            
    return np.array(residuals)

# --- Verification Test on the First Grid Candidate ---
test_u = candidate_controls[0]
test_x0 = np.ones(2 * n_buses - 2)  # Flat voltage guess of 1.0 for Vd, 0.0 for Vq

# FIXED: Corrected argument order -> pass test_x0 (state_x) first, test_u (u_k) second!
test_res = power_flow_residuals(test_x0, test_u, bus_data, gen_data, Ybus, slack_bus, active_gens, non_slack_gens, control_names)

print(f"✔ Step 2 Complete: Generalized square polynomial system built successfully.")
print(f"  System Size: {len(test_res)} equations and {len(test_x0)} unknown state variables.")
print(f"  Sample residual norm with flat voltage guess: {np.linalg.norm(test_res):.4f}")

# =============================================================================
# CELL 4 (HPC PRODUCTION EDITION): Full Grid Sweep & Disk Saving
# =============================================================================
import numpy as np
import pickle
import time
import phcpy
from phcpy.solver import solve
from phcpy.solutions import coordinates

# --- HELPER 1: Calculate transmission line apparent power flows ---
def compute_branch_flows(Vd, Vq, Ybus, branch_data):
    """Calculates apparent power flows S_lm (in MVA/pu) across all transmission lines."""
    n_lines = len(branch_data)
    S_max_calc = np.zeros(n_lines)
    V_cplx = Vd + 1j * Vq
    
    for l in range(n_lines):
        f = int(branch_data[l, 0])
        t = int(branch_data[l, 1])
        r, x = branch_data[l, 2], branch_data[l, 3]
        bc = branch_data[l, 4] if branch_data.shape[1] > 4 else 0.0
        
        y_s = 1.0 / (r + 1j * x)
        y_sh = 1j * bc / 2.0
        
        I_fr = (V_cplx[f] - V_cplx[t]) * y_s + V_cplx[f] * y_sh
        I_to = (V_cplx[t] - V_cplx[f]) * y_s + V_cplx[t] * y_sh
        
        S_fr = np.abs(V_cplx[f] * np.conj(I_fr))
        S_to = np.abs(V_cplx[t] * np.conj(I_to))
        S_max_calc[l] = max(S_fr, S_to)
        
    return S_max_calc

# --- HELPER 2: Filter real roots against OPF inequality constraints ---
def filter_feasible_point(state_x, u_k, bus_data, gen_data, branch_data, Ybus, slack_bus, active_gens, control_names):
    """Checks if a converged real power flow solution satisfies all OPF inequality constraints (Eq 5)."""
    n_buses = len(bus_data)
    G, B = Ybus.real, Ybus.imag
    tol = 1e-3  # Boundary tolerance for discretized grid evaluation
    
    unknown_buses = np.delete(np.arange(n_buses), slack_bus)
    Vd, Vq = np.zeros(n_buses), np.zeros(n_buses)
    num_unknown = len(unknown_buses)
    Vd[unknown_buses] = state_x[:num_unknown]
    Vq[unknown_buses] = state_x[num_unknown:]
    
    slack_control_name = f"V_G{slack_bus+1}"
    Vd[slack_bus] = float(u_k[control_names.index(slack_control_name)])
    Vq[slack_bus] = 0.0
    
    V_mag = np.sqrt(Vd**2 + Vq**2)
    
    P_inj, Q_inj = np.zeros(n_buses), np.zeros(n_buses)
    for i in range(n_buses):
        for k in range(n_buses):
            P_inj[i] += Vd[i] * (G[i,k]*Vd[k] - B[i,k]*Vq[k]) + Vq[i] * (B[i,k]*Vd[k] + G[i,k]*Vq[k])
            Q_inj[i] += Vd[i] * (-B[i,k]*Vd[k] - G[i,k]*Vq[k]) + Vq[i] * (G[i,k]*Vd[k] - B[i,k]*Vq[k])
            
    P_gen = P_inj + bus_data[:, 2]
    Q_gen = Q_inj + bus_data[:, 3]
    
    # FILTER A: Load Bus Voltage Magnitude Limits (Eq 5d)
    for bus_row in bus_data:
        i = int(bus_row[0])
        if i not in active_gens[:, 0]:
            vmin = bus_row[12] if len(bus_row) >= 13 else bus_row[5]
            vmax = bus_row[11] if len(bus_row) >= 13 else bus_row[4]
            if not (vmin - tol <= V_mag[i] <= vmax + tol):
                return False, P_gen, Q_gen, V_mag, None
                
    # FILTER B: Generator Active & Reactive Limits (Eq 5b, 5c)
    for gen in active_gens:
        i = int(gen[0])
        if not (gen[4] - tol <= Q_gen[i] <= gen[3] + tol):
            return False, P_gen, Q_gen, V_mag, None
        if not (gen[9] - tol <= P_gen[i] <= gen[8] + tol):
            return False, P_gen, Q_gen, V_mag, None
            
    # FILTER C: Apparent Power Line Flow Limits (Eq 5e, 5f)
    S_flows = compute_branch_flows(Vd, Vq, Ybus, branch_data)
    if branch_data.shape[1] > 5:
        for l in range(len(branch_data)):
            limit = branch_data[l, 5]
            if limit > 0 and S_flows[l] > limit + tol:
                return False, P_gen, Q_gen, V_mag, S_flows

    return True, P_gen, Q_gen, V_mag, S_flows

# --- HELPER 3: Build PHCpack Algebraic Strings ---
def add_monomial(coeff, var1, var2):
    final_coeff = coeff
    symbols = []
    
    if isinstance(var1, float): final_coeff *= var1
    else: symbols.append(var1)
    if isinstance(var2, float): final_coeff *= var2
    else: symbols.append(var2)
        
    if abs(final_coeff) < 1e-10: return ""
    sign_str = "+ " if final_coeff >= 0 else "- "
    abs_c = abs(final_coeff)
    
    if len(symbols) == 0: return f"{sign_str}{abs_c:.8f}"
    elif len(symbols) == 1: return f"{sign_str}{abs_c:.8f}*{symbols[0]}"
    else:
        if symbols[0] == symbols[1]: return f"{sign_str}{abs_c:.8f}*{symbols[0]}^2"
        else: return f"{sign_str}{abs_c:.8f}*{symbols[0]}*{symbols[1]}"

def build_phcpy_system_strings(u_k, bus_data, gen_data, Ybus, slack_bus, active_gens, control_names):
    n_buses = len(bus_data)
    G, B = Ybus.real, Ybus.imag
    unknown_buses = np.delete(np.arange(n_buses), slack_bus)
    v_slack = float(u_k[control_names.index(f"V_G{slack_bus+1}")])
    
    def get_Vd(k): return v_slack if k == slack_bus else f"Vd{k+1}"
    def get_Vq(k): return 0.0 if k == slack_bus else f"Vq{k+1}"

    poly_equations = []
    for i in unknown_buses:
        p_terms, q_terms = [], []
        for k in range(n_buses):
            vd_i, vq_i = get_Vd(i), get_Vq(i)
            vd_k, vq_k = get_Vd(k), get_Vq(k)
            
            p_terms.extend([add_monomial(G[i,k], vd_i, vd_k), add_monomial(-B[i,k], vd_i, vq_k),
                            add_monomial(B[i,k], vq_i, vd_k), add_monomial(G[i,k], vq_i, vq_k)])
            q_terms.extend([add_monomial(-B[i,k], vd_i, vd_k), add_monomial(-G[i,k], vd_i, vq_k),
                            add_monomial(G[i,k], vq_i, vd_k), add_monomial(-B[i,k], vq_i, vq_k)])
            
        def clean_expr(terms_list):
            expr = " ".join([t for t in terms_list if t != ""])
            if expr.startswith("+ "): return expr[2:]
            elif expr.startswith("- "): return "-" + expr[2:]
            return expr

        P_calc_str, Q_calc_str = clean_expr(p_terms), clean_expr(q_terms)
        
        is_gen = i in active_gens[:, 0]
        if is_gen:
            p_target = float(u_k[control_names.index(f"P_G{i+1}")])
            p_target_str = f"- {p_target:.8f}" if p_target >= 0 else f"+ {abs(p_target):.8f}"
            poly_equations.append(f"{P_calc_str} {p_target_str};")
            
            v_target = float(u_k[control_names.index(f"V_G{i+1}")])
            poly_equations.append(f"Vd{i+1}^2 + Vq{i+1}^2 - {v_target**2:.8f};")
        else:
            p_load = float(bus_data[bus_data[:, 0] == i, 2][0])
            q_load = float(bus_data[bus_data[:, 0] == i, 3][0])
            p_load_str = f"+ {p_load:.8f}" if p_load >= 0 else f"- {abs(p_load):.8f}"
            q_load_str = f"+ {q_load:.8f}" if q_load >= 0 else f"- {abs(q_load):.8f}"
            
            poly_equations.append(f"{P_calc_str} {p_load_str};")
            poly_equations.append(f"{Q_calc_str} {q_load_str};")
            
    var_names = [f"Vd{i+1}" for i in unknown_buses] + [f"Vq{i+1}" for i in unknown_buses]
    return poly_equations, var_names

# --- HELPER 4: Official Documentation-Faithful Root Parser ---
def parse_phcpy_real_roots(raw_solutions, var_names):
    real_roots = []
    if not raw_solutions: return real_roots
    
    for sol in raw_solutions:
        try:
            vars_list, vals_list = coordinates(sol)
            sol_dict = dict(zip(vars_list, vals_list))
            if not all(v in sol_dict for v in var_names): continue
                
            cplx_vals = [sol_dict[v] for v in var_names]
            if all(abs(c.imag) < 1e-3 for c in cplx_vals):
                real_roots.append(np.array([c.real for c in cplx_vals]))
        except Exception:
            continue
    return real_roots

# =============================================================================
# EXECUTE FULL PRODUCTION SWEEP ACROSS ENTIRE GRID
# =============================================================================
# We evaluate ALL coordinates in candidate_controls
eval_grid = candidate_controls
total_points = len(eval_grid)
print(f"Starting Production NPHC Sweep across {total_points:,} coordinates...")
print(f"Time started: {time.strftime('%X')}\n")

feasible_points = []
start_time = time.time()

for k, u_k in enumerate(eval_grid):
    pols, var_names = build_phcpy_system_strings(u_k, bus_data, gen_data, Ybus, slack_bus, active_gens, control_names)
    raw_complex_solutions = solve(pols)
    real_roots = parse_phcpy_real_roots(raw_complex_solutions, var_names)
    
    num_feas = 0
    for sol_x in real_roots:
        is_feas, P_gen, Q_gen, V_mag, S_flows = filter_feasible_point(
            sol_x, u_k, bus_data, gen_data, branch_data, Ybus, slack_bus, active_gens, control_names
        )
        if is_feas:
            # Generation Cost Formula for WB5: Cost = 400*P_G1 + 100*P_G5 ($/h)
            cost = 400.0 * P_gen[0] + 100.0 * P_gen[4]
            feasible_points.append({
                'u_k': u_k, 'P_gen': P_gen, 'Q_gen': Q_gen, 'V_mag': V_mag, 'cost': cost
            })
            num_feas += 1
            break # Found the feasible operating point for this grid coordinate
            
    # Log progress cleanly every 50 evaluations or whenever a point is found
    if (k + 1) % 50 == 0 or num_feas > 0:
        elapsed_sec = time.time() - start_time
        rate = (k + 1) / elapsed_sec
        est_rem_min = ((total_points - (k + 1)) / rate) / 60.0
        print(f"  [Progress {(k+1):4d}/{total_points:,} | {(k+1)/total_points*100:5.1f}%] P_G5={u_k[0]:.2f} pu | Real Roots: {len(real_roots):2d} | Feasible Total: {len(feasible_points):3d} | ETA: {est_rem_min:.1f} min")

print(f"\n✔ True NPHC Production Sweep Complete!")
print(f"  Total Time Elapsed: {(time.time() - start_time)/60:.2f} minutes")
print(f"  Strictly Feasible OPF Operating Points Found: {len(feasible_points):,}")

# --- SAVE RESULTS TO DISK FOR HPC SAFETY ---
output_filename = "wb5_feasible_points.pkl"
with open(output_filename, "wb") as f:
    pickle.dump(feasible_points, f)
print(f"✔ Data successfully persisted to disk: '{output_filename}' (Safe to plot!)")