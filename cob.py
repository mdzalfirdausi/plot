"""
cob.py — Replicating Molzahn (2017) Fig 3 Feasible Space Ribbon
Fixed: Unpacks double_track tuple & removes complex float warnings.
"""
import argparse
import os
import re
import time
import random
import pickle
import numpy as np
import multiprocessing as mp

from phcpy.solver import solve
from phcpy.solutions import coordinates
from phcpy.trackers import double_track as track

parser = argparse.ArgumentParser(description="Run NPHC Homotopy Solver.")
parser.add_argument("--file", type=str, required=True, help="Path to MATPOWER file (.m)")
args = parser.parse_args()

# =============================================================================
# PART 1: SYSTEM SETUP & PARSING
# =============================================================================
def parse_matpower_matrix(content, matrix_name):
    pattern = rf'mpc\.{matrix_name}\s*=\s*\[(.*?)\];'
    match = re.search(pattern, content, re.DOTALL)
    if not match: return []
    matrix_str = re.sub(r'%.*', '', match.group(1))
    rows = []
    for line in matrix_str.replace(';', '\n').split('\n'):
        clean_line = line.strip()
        if clean_line: rows.append([float(val) for val in clean_line.split()])
    return np.array(rows)

def load_case_data(filepath):
    if not os.path.exists(filepath): raise FileNotFoundError(f"Case file not found: {filepath}")
    with open(filepath, 'r') as f: content = f.read()
    base_match = re.search(r'mpc\.baseMVA\s*=\s*([\d\.]+);', content)
    baseMVA = float(base_match.group(1)) if base_match else 100.0
    return baseMVA, parse_matpower_matrix(content, 'bus'), parse_matpower_matrix(content, 'gen'), parse_matpower_matrix(content, 'branch')

script_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(script_dir, args.file)
print(f"Loading system data from {filepath}...")
baseMVA, bus_data, gen_data, branch_data = load_case_data(filepath)

bus_data[:, 0] -= 1
gen_data[:, 0] -= 1
branch_data[:, 0] -= 1
branch_data[:, 1] -= 1

bus_data[:, 2] /= baseMVA
bus_data[:, 3] /= baseMVA
gen_data[:, [1, 2, 3, 4, 8, 9]] /= baseMVA

if branch_data.shape[1] > 5:
    branch_data[:, 5] /= baseMVA
    branch_data[branch_data[:, 5] == 0, 5] = 1e10

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

# =============================================================================
# PART 2: CONTROL BOUNDS (Restored full physical voltage range)
# =============================================================================
active_gens = gen_data[gen_data[:, 7] == 1]
p_ranges = active_gens[:, 8] - active_gens[:, 9]
slack_idx_in_gen = np.argmax(p_ranges)
slack_bus = int(active_gens[slack_idx_in_gen, 0])
non_slack_gens = np.delete(active_gens, slack_idx_in_gen, axis=0)

control_names, u_min, u_max = [], [], []

for gen in non_slack_gens:
    bus_id = int(gen[0])
    control_names.append(f"P_G{bus_id+1}")
    u_min.append(0.50)  
    u_max.append(3.50)

for gen in active_gens:
    bus_id = int(gen[0])
    bus_row = bus_data[bus_data[:, 0] == bus_id][0]
    control_names.append(f"V_G{bus_id+1}")
    vmin_idx = 12 if len(bus_row) >= 13 else 5
    vmax_idx = 11 if len(bus_row) >= 13 else 4
    u_min.append(bus_row[vmin_idx])
    u_max.append(bus_row[vmax_idx])

u_min, u_max = np.array(u_min), np.array(u_max)
num_controls = len(control_names)

# =============================================================================
# PART 3: HOMOTOPY CONTINUATION & FILTERING
# =============================================================================
def compute_branch_flows(Vd, Vq, Ybus, branch_data):
    n_lines = len(branch_data)
    S_max_calc = np.zeros(n_lines)
    V_cplx = Vd + 1j * Vq
    for l in range(n_lines):
        f, t = int(branch_data[l, 0]), int(branch_data[l, 1])
        r, x = branch_data[l, 2], branch_data[l, 3]
        bc = branch_data[l, 4] if branch_data.shape[1] > 4 else 0.0
        y_s, y_sh = 1.0 / (r + 1j * x), 1j * bc / 2.0
        I_fr = (V_cplx[f] - V_cplx[t]) * y_s + V_cplx[f] * y_sh
        I_to = (V_cplx[t] - V_cplx[f]) * y_s + V_cplx[t] * y_sh
        S_max_calc[l] = max(np.abs(V_cplx[f] * np.conj(I_fr)), np.abs(V_cplx[t] * np.conj(I_to)))
    return S_max_calc

def filter_feasible_point(state_x, u_k, bus_data, gen_data, branch_data, Ybus, slack_bus, active_gens, control_names, tol=1e-3):
    n_buses = len(bus_data)
    G, B = Ybus.real, Ybus.imag
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
            P_inj[i] += Vd[i]*(G[i, k]*Vd[k] - B[i, k]*Vq[k]) + Vq[i]*(B[i, k]*Vd[k] + G[i, k]*Vq[k])
            Q_inj[i] += Vd[i]*(-B[i, k]*Vd[k] - G[i, k]*Vq[k]) + Vq[i]*(G[i, k]*Vd[k] - B[i, k]*Vq[k])

    P_gen, Q_gen = P_inj + bus_data[:, 2], Q_inj + bus_data[:, 3]

    for bus_row in bus_data:
        i = int(bus_row[0])
        if i not in active_gens[:, 0]:
            vmin = bus_row[12] if len(bus_row) >= 13 else bus_row[5]
            vmax = bus_row[11] if len(bus_row) >= 13 else bus_row[4]
            if not (vmin - tol <= V_mag[i] <= vmax + tol): return False, P_gen, Q_gen, V_mag, None

    for gen in active_gens:
        i = int(gen[0])
        if not (gen[4] - tol <= Q_gen[i] <= gen[3] + tol): return False, P_gen, Q_gen, V_mag, None
        if not (gen[9] - tol <= P_gen[i] <= gen[8] + tol): return False, P_gen, Q_gen, V_mag, None

    S_flows = compute_branch_flows(Vd, Vq, Ybus, branch_data)
    if branch_data.shape[1] > 5:
        for l in range(len(branch_data)):
            limit = branch_data[l, 5]
            if limit > 0 and S_flows[l] > limit + tol: return False, P_gen, Q_gen, V_mag, S_flows

    return True, P_gen, Q_gen, V_mag, S_flows

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
    else: return (f"{sign_str}{abs_c:.8f}*{symbols[0]}^2" if symbols[0] == symbols[1] else f"{sign_str}{abs_c:.8f}*{symbols[0]}*{symbols[1]}")

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
            vd_i, vq_i, vd_k, vq_k = get_Vd(i), get_Vq(i), get_Vd(k), get_Vq(k)
            p_terms.extend([add_monomial(G[i, k], vd_i, vd_k), add_monomial(-B[i, k], vd_i, vq_k), add_monomial(B[i, k], vq_i, vd_k), add_monomial(G[i, k], vq_i, vq_k)])
            q_terms.extend([add_monomial(-B[i, k], vd_i, vd_k), add_monomial(-G[i, k], vd_i, vq_k), add_monomial(G[i, k], vq_i, vd_k), add_monomial(-B[i, k], vq_i, vq_k)])

        def clean_expr(t_list):
            expr = " ".join([t for t in t_list if t != ""])
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
        except Exception: continue
    return real_roots

def evaluate_grid_point_cheater(args):
    """Worker uses Parameter Homotopy Tracking"""
    k, u_k, pols_generic, generic_seeds = args
    pols_target, var_names = build_phcpy_system_strings(u_k, bus_data, gen_data, Ybus, slack_bus, active_gens, control_names)
    
    try:
        # CRITICAL FIX: Unpack tuple to separate gamma from solutions list!
        _, target_complex_solutions = track(pols_target, pols_generic, generic_seeds)
        real_roots = parse_phcpy_real_roots(target_complex_solutions, var_names)
    except Exception:
        return None
    
    for sol_x in real_roots:
        is_feas, P_gen, Q_gen, V_mag, S_flows = filter_feasible_point(
            sol_x, u_k, bus_data, gen_data, branch_data, Ybus, slack_bus, active_gens, control_names
        )
        if is_feas:
            cost = 400.0 * P_gen[0] + 100.0 * P_gen[4]
            return {'u_k': u_k, 'P_gen': P_gen, 'Q_gen': Q_gen, 'V_mag': V_mag, 'cost': cost}
    return None

# =============================================================================
# PART 4: MASTER EXECUTION BLOCK
# =============================================================================
if __name__ == '__main__':
    # 250 steps on P_G5, 20 steps on voltages = 100,000 High-Density Coordinates!
    N_res_P, N_res_V = 250, 20  
    d_sweeps = [np.linspace(u_min[i], u_max[i], N_res_P if "P_G" in control_names[i] else N_res_V) for i in range(num_controls)]
    mesh_grids = np.meshgrid(*d_sweeps, indexing='ij')
    candidate_controls = np.vstack([grid.ravel() for grid in mesh_grids]).T
    total_points = len(candidate_controls)

    print(f"Loading system data from {filepath}...")
    print(f"High-Density Grid complete: {total_points:,} coordinates generated.")

    print("\n--- PHASE 1: CHEATER'S HOMOTOPY PREPROCESSING ---")
    # CRITICAL FIX: Real random numbers prevent ComplexWarning when casting float()
    # double_track automatically applies a random complex gamma constant to prevent singularities!
    u_generic = np.array([u_min[i] + random.random() * (u_max[i] - u_min[i]) for i in range(num_controls)])
    pols_generic, _ = build_phcpy_system_strings(u_generic, bus_data, gen_data, Ybus, slack_bus, active_gens, control_names)
    
    print("Solving generic complex system from scratch. Please wait...")
    generic_seeds = solve(pols_generic)
    num_seeds = len(generic_seeds) if generic_seeds else 0
    print(f"✔ Generic Start System Solved! Found {num_seeds} valid generic seed paths to track.")

    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count()))
    print(f"\n[Auto-Scale Engine] Spawning {num_workers} parallel workers for parameter sweep...")
    print(f"Time started: {time.strftime('%X')}\n")

    start_time = time.time()
    feasible_points = []
    completed_count = 0

    tasks = [(k, u_k, pols_generic, generic_seeds) for k, u_k in enumerate(candidate_controls)]

    with mp.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(evaluate_grid_point_cheater, tasks, chunksize=20):
            completed_count += 1
            if result is not None:
                feasible_points.append(result)
                
            if completed_count % 2000 == 0 or completed_count == total_points:
                elapsed_sec = time.time() - start_time
                rate = completed_count / elapsed_sec
                est_rem_min = ((total_points - completed_count) / rate) / 60.0
                print(f"  [Progress {completed_count:6d}/{total_points:,} | {completed_count/total_points*100:5.1f}%] Feasible Total: {len(feasible_points):4d} | Rate: {rate:.1f} pts/sec | ETA: {est_rem_min:.1f} min", flush=True)

    print(f"\n✔ Parameter Sweep Complete!")
    print(f"  Total Time Elapsed: {(time.time() - start_time)/60:.2f} minutes")
    print(f"  Strictly Feasible OPF Operating Points Found: {len(feasible_points):,}")

    output_filename = "wb5_feasible_points_FINAL.pkl"
    with open(output_filename, "wb") as f:
        pickle.dump(feasible_points, f)
    print(f"✔ Combined dataset persisted to disk: '{output_filename}'")