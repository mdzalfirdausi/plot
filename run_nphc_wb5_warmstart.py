"""
run_nphc_wb5_warmstart_v2.py
==========================================================================
Robust warm-started NPHC for WB5 feasible-space (Molzahn 2017, Fig 3).

KEY: Uses coefficient-parameter homotopy (phcpy.trackers) to hot-start
      every grid point off its neighbor's solutions, instead of solving
      ab initio (full Bezout-bound) at every point. ~10-40x faster.

ROBUST IMPORTS: Tries multiple phcpy function names to handle API variations.
"""

import os
import re
import time
import pickle
import numpy as np
import multiprocessing as mp

import phcpy
from phcpy.solver import solve
from phcpy.solutions import coordinates

# Try to import the tracking functions with fallback names
try:
    from phcpy.trackers import standard_double_track as tracker_func
    TRACKER_NAME = "standard_double_track"
    print("✓ Using phcpy.trackers.standard_double_track")
except ImportError:
    try:
        from phcpy.trackers import track as tracker_func
        TRACKER_NAME = "track"
        print("✓ Using phcpy.trackers.track")
    except ImportError:
        try:
            from phcpy.solver import track as tracker_func
            TRACKER_NAME = "solver.track"
            print("✓ Using phcpy.solver.track")
        except ImportError:
            print("⚠ Warning: No tracking function found; will use fresh solve() at every point")
            print("  (this will be slower, but should still work)")
            tracker_func = None
            TRACKER_NAME = "none"

# =============================================================================
# PART 1: SYSTEM SETUP & PARSING
# =============================================================================
def parse_matpower_matrix(content, matrix_name):
    pattern = rf'mpc\.{matrix_name}\s*=\s*\[(.*?)\];'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return []
    matrix_str = re.sub(r'%.*', '', match.group(1))
    rows = []
    for line in matrix_str.replace(';', '\n').split('\n'):
        clean_line = line.strip()
        if clean_line:
            rows.append([float(val) for val in clean_line.split()])
    return np.array(rows)


def load_case_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Case file not found: {filepath}")
    with open(filepath, 'r') as f:
        content = f.read()
    base_match = re.search(r'mpc\.baseMVA\s*=\s*([\d\.]+);', content)
    baseMVA = float(base_match.group(1)) if base_match else 100.0
    return (baseMVA,
            parse_matpower_matrix(content, 'bus'),
            parse_matpower_matrix(content, 'gen'),
            parse_matpower_matrix(content, 'branch'))


filepath = 'WB5.m'
print(f"\n[1/5] Loading {filepath}...")
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
# PART 2: CONTROL SPACE DISCRETIZATION
# =============================================================================
print("[2/5] Discretizing control space...")

active_gens = gen_data[gen_data[:, 7] == 1]
p_ranges = active_gens[:, 8] - active_gens[:, 9]
slack_idx_in_gen = np.argmax(p_ranges)
slack_bus = int(active_gens[slack_idx_in_gen, 0])
non_slack_gens = np.delete(active_gens, slack_idx_in_gen, axis=0)

control_names = []
u_min, u_max = [], []
total_load_pu = np.sum(bus_data[:, 2])

for gen in non_slack_gens:
    bus_id = int(gen[0])
    control_names.append(f"P_G{bus_id+1}")
    u_min.append(gen[9])
    plot_p_ceiling = min(gen[8], max(4.0, total_load_pu * 1.25))
    u_max.append(plot_p_ceiling)

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

# FINER GRID: Now affordable if using tracking; adjust N_res down if it's too slow
N_res = 40
d_sweeps = [np.linspace(u_min[i], u_max[i], N_res) for i in range(num_controls)]
mesh_grids = np.meshgrid(*d_sweeps, indexing='ij')
candidate_controls = np.vstack([grid.ravel() for grid in mesh_grids]).T

print(f"  Grid: {len(candidate_controls):,} coordinates ({N_res} pts/axis)")
print(f"  Controls: {control_names}")

# =============================================================================
# PART 3: BUILD SYSTEM STRINGS / FILTER FEASIBILITY
# =============================================================================
print("[3/5] Building helper functions...")


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
        S_max_calc[l] = max(np.abs(V_cplx[f] * np.conj(I_fr)),
                             np.abs(V_cplx[t] * np.conj(I_to)))
    return S_max_calc


def filter_feasible_point(state_x, u_k, bus_data, gen_data, branch_data, Ybus,
                           slack_bus, active_gens, control_names, tol=1e-3):
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
            if not (vmin - tol <= V_mag[i] <= vmax + tol):
                return False, P_gen, Q_gen, V_mag, None

    for gen in active_gens:
        i = int(gen[0])
        if not (gen[4] - tol <= Q_gen[i] <= gen[3] + tol):
            return False, P_gen, Q_gen, V_mag, None
        if not (gen[9] - tol <= P_gen[i] <= gen[8] + tol):
            return False, P_gen, Q_gen, V_mag, None

    S_flows = compute_branch_flows(Vd, Vq, Ybus, branch_data)
    if branch_data.shape[1] > 5:
        for l in range(len(branch_data)):
            limit = branch_data[l, 5]
            if limit > 0 and S_flows[l] > limit + tol:
                return False, P_gen, Q_gen, V_mag, S_flows

    return True, P_gen, Q_gen, V_mag, S_flows


def add_monomial(coeff, var1, var2):
    final_coeff = coeff
    symbols = []
    if isinstance(var1, float):
        final_coeff *= var1
    else:
        symbols.append(var1)
    if isinstance(var2, float):
        final_coeff *= var2
    else:
        symbols.append(var2)

    if abs(final_coeff) < 1e-10:
        return ""
    sign_str = "+ " if final_coeff >= 0 else "- "
    abs_c = abs(final_coeff)

    if len(symbols) == 0:
        return f"{sign_str}{abs_c:.8f}"
    elif len(symbols) == 1:
        return f"{sign_str}{abs_c:.8f}*{symbols[0]}"
    else:
        return (f"{sign_str}{abs_c:.8f}*{symbols[0]}^2" if symbols[0] == symbols[1]
                else f"{sign_str}{abs_c:.8f}*{symbols[0]}*{symbols[1]}")


def build_phcpy_system_strings(u_k, bus_data, gen_data, Ybus, slack_bus,
                                active_gens, control_names):
    n_buses = len(bus_data)
    G, B = Ybus.real, Ybus.imag
    unknown_buses = np.delete(np.arange(n_buses), slack_bus)
    v_slack = float(u_k[control_names.index(f"V_G{slack_bus+1}")])

    def get_Vd(k):
        return v_slack if k == slack_bus else f"Vd{k+1}"

    def get_Vq(k):
        return 0.0 if k == slack_bus else f"Vq{k+1}"

    poly_equations = []
    for i in unknown_buses:
        p_terms, q_terms = [], []
        for k in range(n_buses):
            vd_i, vq_i, vd_k, vq_k = get_Vd(i), get_Vq(i), get_Vd(k), get_Vq(k)
            p_terms.extend([add_monomial(G[i, k], vd_i, vd_k), add_monomial(-B[i, k], vd_i, vq_k),
                             add_monomial(B[i, k], vq_i, vd_k), add_monomial(G[i, k], vq_i, vq_k)])
            q_terms.extend([add_monomial(-B[i, k], vd_i, vd_k), add_monomial(-G[i, k], vd_i, vq_k),
                             add_monomial(G[i, k], vq_i, vd_k), add_monomial(-B[i, k], vq_i, vq_k)])

        def clean_expr(t_list):
            expr = " ".join([t for t in t_list if t != ""])
            if expr.startswith("+ "):
                return expr[2:]
            elif expr.startswith("- "):
                return "-" + expr[2:]
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


def parse_phcpy_real_roots(raw_solutions, var_names, imag_tol=1e-3):
    real_roots = []
    if not raw_solutions:
        return real_roots
    for sol in raw_solutions:
        try:
            vars_list, vals_list = coordinates(sol)
            sol_dict = dict(zip(vars_list, vals_list))
            if not all(v in sol_dict for v in var_names):
                continue
            cplx_vals = [sol_dict[v] for v in var_names]
            if all(abs(c.imag) < imag_tol for c in cplx_vals):
                real_roots.append(np.array([c.real for c in cplx_vals]))
        except Exception:
            continue
    return real_roots


# =============================================================================
# PART 4: CHUNK WORKER WITH TRACKING ATTEMPT
# =============================================================================
print("[4/5] Launching solver...")
REFRESH_EVERY = 20


def evaluate_chunk(chunk):
    """
    Walk one contiguous slice of the grid sequentially.
    - First point: ab initio solve() (expensive)
    - Subsequent points: try to warm-start via tracking (cheap)
    - Fallback: if tracking fails/unavailable, use fresh solve()
    """
    results = []
    prev_pols, prev_sols = None, None
    failed_tracks = 0

    for step, (k, u_k) in enumerate(chunk):
        pols, var_names = build_phcpy_system_strings(
            u_k, bus_data, gen_data, Ybus, slack_bus, active_gens, control_names)

        need_ab_initio = (
            prev_sols is None
            or step % REFRESH_EVERY == 0
            or tracker_func is None
        )

        if need_ab_initio:
            raw_sols = solve(pols)
        else:
            # Try warm-start tracking
            try:
                raw_sols = tracker_func(pols, prev_pols, prev_sols)
                
                # Safety net: if solution count dropped by >50%, recapture via ab initio
                if raw_sols and prev_sols and len(raw_sols) < 0.5 * len(prev_sols):
                    raw_sols = solve(pols)
                    
            except Exception as e:
                # Tracking failed; fall back to ab initio
                raw_sols = solve(pols)
                failed_tracks += 1

        prev_pols, prev_sols = pols, raw_sols

        real_roots = parse_phcpy_real_roots(raw_sols, var_names)
        for sol_x in real_roots:
            is_feas, P_gen, Q_gen, V_mag, S_flows = filter_feasible_point(
                sol_x, u_k, bus_data, gen_data, branch_data, Ybus,
                slack_bus, active_gens, control_names)
            if is_feas:
                cost = 400.0 * P_gen[0] + 100.0 * P_gen[4]
                results.append({
                    'u_k': u_k, 'P_gen': P_gen, 'Q_gen': Q_gen,
                    'V_mag': V_mag, 'cost': cost
                })
                break  # Only keep first feasible root at each grid point

    return results, failed_tracks


# =============================================================================
# PART 5: EXECUTE IN PARALLEL
# =============================================================================
if __name__ == "__main__":
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count()))
    total_points = len(candidate_controls)

    # Split into contiguous chunks (contiguous in meshgrid-ravel order).
    # NOTE: np.array_split() can't be called directly on a list of
    # (index, control_vector) tuples -- numpy tries to coerce that into a
    # single rectangular array and fails ("inhomogeneous shape") because
    # each control_vector is itself an array. Instead, split the plain
    # integer index range first, then attach the control vectors after.
    idx_splits = np.array_split(np.arange(total_points), num_workers)
    chunks = [
        [(int(i), candidate_controls[i]) for i in idx_chunk]
        for idx_chunk in idx_splits if len(idx_chunk) > 0
    ]

    print(f"[5/5] Executing NPHC sweep")
    print(f"      {total_points:,} grid points / {len(chunks)} chunks / {num_workers} CPUs")
    print(f"      Tracker: {TRACKER_NAME}")
    print(f"      Warm-start refresh every {REFRESH_EVERY} points")
    print(f"      Started: {time.strftime('%X')}\n")

    start_time = time.time()
    feasible_points = []
    total_failed_tracks = 0

    with mp.Pool(processes=len(chunks)) as pool:
        for chunk_result, failed_count in pool.imap_unordered(evaluate_chunk, chunks):
            feasible_points.extend(chunk_result)
            total_failed_tracks += failed_count
            elapsed = (time.time() - start_time) / 60.0
            rate = len(feasible_points) / max(0.01, elapsed)
            print(f"      [chunk done | {elapsed:.1f}m] "
                  f"feasible: {len(feasible_points):,} | "
                  f"rate: {rate:.1f} pts/min", flush=True)

    total_time_sec = time.time() - start_time
    print(f"\n" + "="*70)
    print(f"COMPLETE! Total Time: {total_time_sec/60:.2f} min")
    print(f"Strictly Feasible OPF Operating Points: {len(feasible_points):,}")
    if total_failed_tracks > 0:
        print(f"(Note: {total_failed_tracks} tracking attempts fell back to ab initio solve)")
    print("="*70)

    output_filename = "wb5_feasible_points_warmstart.pkl"
    with open(output_filename, "wb") as f:
        pickle.dump(feasible_points, f)
    print(f"\n✓ Saved to: '{output_filename}'")
    print(f"  Ready for plotting (Cell 5)!")