# move particles

import libs
import params
import tools
from data_init import chunk_to_seconds, iter_minute_chunks

def _wrap_J(J, j_max, j_period):
    """Wrap longitudinal index J into [1..j_max] using the domain's periodicity."""
    if J < 1:
        return J + j_period
    if J > j_max:
        return J - j_period
    return J

def _clamp_I(I, i_max):
    """Clamp radial/latitudinal index I to [1..i_max]."""
    return max(1.0, min(float(I), float(i_max)))

def compute_adaptive_steps(I, J, j_max, j_period, interp_COLAT, interp_ALOCT_sin, interp_ALOCT_cos):
    initial_delta = 0.01  # Initial step size for numerical derivatives
    grid_spacing = 1.0    # Grid spacing in I and J (since I and J are integers in the dataset)
    target_fraction = 0.001 # Target step size as a smaller fraction of the characteristic length scale

    # Compute dtheta/dI (first pass)
    colat_minus = interp_COLAT([J, I - initial_delta])[0] * libs.np.pi / 180.0
    colat_plus = interp_COLAT([J, I + initial_delta])[0] * libs.np.pi / 180.0
    dtheta_dI = (colat_plus - colat_minus) / (2 * initial_delta)

    # Compute dphi/dJ with wrapping (first pass)
    J_minus = J - initial_delta
    J_plus = J + initial_delta
    if J_minus < 1:
        J_minus += j_period
    if J_plus > j_max:
        J_plus -= j_period
    
    # Reconstruct ALOCT from sin and cos components
    sin_minus = interp_ALOCT_sin([J_minus, I])[0]
    cos_minus = interp_ALOCT_cos([J_minus, I])[0]
    phi_minus = libs.np.arctan2(sin_minus, cos_minus) % (2 * libs.np.pi)  # Normalize to [0, 2π)
    
    sin_plus = interp_ALOCT_sin([J_plus, I])[0]
    cos_plus = interp_ALOCT_cos([J_plus, I])[0]
    phi_plus = libs.np.arctan2(sin_plus, cos_plus) % (2 * libs.np.pi)
    
    # Compute the periodic difference
    delta_phi = phi_plus - phi_minus
    if delta_phi > libs.np.pi:
        delta_phi -= 2 * libs.np.pi
    elif delta_phi < -libs.np.pi:
        delta_phi += 2 * libs.np.pi
    dphi_dJ = delta_phi / (2 * initial_delta)

    # Ensure non-zero derivatives to avoid division by zero
    dtheta_dI = max(abs(dtheta_dI), 1e-6)
    dphi_dJ = max(abs(dphi_dJ), 1e-6)

    # Estimate characteristic length scales
    L_theta = 1.0 / dtheta_dI  # Length scale in I direction (units of I)
    L_phi = 1.0 / dphi_dJ      # Length scale in J direction (units of J)

    # Compute adaptive step sizes as a fraction of the characteristic length
    delta_I = target_fraction * L_theta
    delta_J = target_fraction * L_phi

    # Clamp to reasonable values relative to the grid spacing
    min_delta = 0.01 * grid_spacing  # Minimum step: 1% of a grid cell
    max_delta = 0.1 * grid_spacing   # Maximum step: 20% of a grid cell
    delta_I = max(min_delta, min(delta_I, max_delta))
    delta_J = max(min_delta, min(delta_J, max_delta))

    # Recompute dtheta/dI and dphi/dJ with adaptive step sizes
    colat_minus = interp_COLAT([J, I - delta_I])[0] * libs.np.pi / 180.0
    colat_plus = interp_COLAT([J, I + delta_I])[0] * libs.np.pi / 180.0
    dtheta_dI = (colat_plus - colat_minus) / (2 * delta_I)

    J_minus = J - delta_J
    J_plus = J + delta_J
    if J_minus < 1:
        J_minus += j_period
    if J_plus > j_max:
        J_plus -= j_period
    
    # Reconstruct ALOCT from sin and cos components
    sin_minus = interp_ALOCT_sin([J_minus, I])[0]
    cos_minus = interp_ALOCT_cos([J_minus, I])[0]
    phi_minus = libs.np.arctan2(sin_minus, cos_minus) % (2 * libs.np.pi)
    
    sin_plus = interp_ALOCT_sin([J_plus, I])[0]
    cos_plus = interp_ALOCT_cos([J_plus, I])[0]
    phi_plus = libs.np.arctan2(sin_plus, cos_plus) % (2 * libs.np.pi)
    
    # Compute the periodic difference
    delta_phi = phi_plus - phi_minus
    if delta_phi > libs.np.pi:
        delta_phi -= 2 * libs.np.pi
    elif delta_phi < -libs.np.pi:
        delta_phi += 2 * libs.np.pi
    dphi_dJ = delta_phi / (2 * delta_J)

    # Ensure non-zero derivatives
    sign_th  = 1.0 if dtheta_dI>=0 else -1.0
    sign_phi = 1.0 if dphi_dJ  >=0 else -1.0
    dtheta_dI = sign_th  * max(abs(dtheta_dI), 1e-6)
    dphi_dJ   = sign_phi * max(abs(dphi_dJ), 1e-6)

    return delta_I, delta_J, dtheta_dI, dphi_dJ

def compute_dVeff_derivatives(I, J, delta_I, delta_J, j_max, j_period, interp_V, interp_VM, species, lambda_val):

    sign = 1.0 if species == 'p' else -1.0

    # Compute dV_eff/dI
    V_minus_I = interp_V([J, I - delta_I])[0]
    VM_minus_I = interp_VM([J, I - delta_I])[0]
    V_plus_I = interp_V([J, I + delta_I])[0]
    VM_plus_I = interp_VM([J, I + delta_I])[0]
    V_eff_minus_I = V_minus_I + sign * lambda_val * VM_minus_I
    V_eff_plus_I = V_plus_I + sign * lambda_val * VM_plus_I
    dV_eff_dI = (V_eff_plus_I - V_eff_minus_I) / (2 * delta_I)

    # Compute dV_eff/dJ with wrapping
    J_minus = J - delta_J
    J_plus = J + delta_J
    if J_minus < 1:
        J_minus += j_period
    if J_plus > j_max:
        J_plus -= j_period
    V_minus_J = interp_V([J_minus, I])[0]
    VM_minus_J = interp_VM([J_minus, I])[0]
    V_plus_J = interp_V([J_plus, I])[0]
    VM_plus_J = interp_VM([J_plus, I])[0]
    V_eff_minus_J = V_minus_J + sign * lambda_val * VM_minus_J
    V_eff_plus_J = V_plus_J + sign * lambda_val * VM_plus_J
    dV_eff_dJ = (V_eff_plus_J - V_eff_minus_J) / (2 * delta_J)

    return dV_eff_dI, dV_eff_dJ

def compute_dI_dt_dJ_dt(dV_eff_dI, dV_eff_dJ, theta, dtheta_dI, dphi_dJ):
    # Compute B_r
    Br = -2 * params.B0 * (params.RE / params.r)**3 * libs.np.cos(theta)
    # Compute fac
    fac = params.r**2 * abs(Br) * libs.np.sin(theta) * dphi_dJ * dtheta_dI
    fac = max(abs(fac), params.epsilon) * libs.np.sign(fac or 1)

    # Compute dI/dt and dJ/dt
    dI_dt = ( 1.0 / fac) * dV_eff_dJ
    dJ_dt = (-1.0 / fac) * dV_eff_dI

    return dI_dt, dJ_dt
    
def update_position_new(
    I, J, metadata, int_method, j_max, j_period,
    chunk_interpolators, interpolators, chunk, species, lambda_val
):
    # Normalize integration method (fix typo) and set substep dt from metadata cadence
    int_method = (int_method or params.int_method).lower()
    num_substeps = params.RK4_sub_step
    substep_dt = (-float(metadata.get('seconds_per_chunk', 60.0))) / float(num_substeps)

    current_I = float(I)
    current_J = float(J)

    # --- Hoist all per-minute interpolators (fields frozen within a minute) ---
    interp_COLAT     = chunk_interpolators[chunk]['COLAT']
    interp_ALOCT_sin = chunk_interpolators[chunk]['ALOCT_sin']
    interp_ALOCT_cos = chunk_interpolators[chunk]['ALOCT_cos']
    interp_V         = interpolators[chunk]['V']
    interp_VM        = interpolators[chunk]['VM']

    for _ in range(num_substeps):
        colat_deg = interp_COLAT([current_J, current_I])[0]
        if libs.np.isnan(colat_deg):
            return None, None
        theta = colat_deg * libs.np.pi / 180.0

        # local geometry
        dI1, dJ1, dth1, dph1 = compute_adaptive_steps(
            current_I, current_J, j_max, j_period, interp_COLAT, interp_ALOCT_sin, interp_ALOCT_cos
        )
        dVeI1, dVeJ1 = compute_dVeff_derivatives(
            current_I, current_J, dI1, dJ1, j_max, j_period, interp_V, interp_VM, species, lambda_val
        )
        k1_I, k1_J = compute_dI_dt_dJ_dt(dVeI1, dVeJ1, theta, dth1, dph1)
        if libs.np.isnan(k1_I) or libs.np.isnan(k1_J):
            return None, None

        if int_method == 'euler':
            new_I = current_I + substep_dt * k1_I
            new_J = current_J + substep_dt * k1_J

        elif int_method == 'rk4':
            # k2
            I2 = current_I + 0.5 * substep_dt * k1_I
            J2 = _wrap_J(current_J + 0.5 * substep_dt * k1_J, j_max, j_period)
            colat2 = interp_COLAT([J2, I2])[0]
            if libs.np.isnan(colat2):
                return None, None
            th2 = colat2 * libs.np.pi / 180.0
            dI2, dJ2, dth2, dph2 = compute_adaptive_steps(I2, J2, j_max, j_period, interp_COLAT, interp_ALOCT_sin, interp_ALOCT_cos)
            dVeI2, dVeJ2 = compute_dVeff_derivatives(I2, J2, dI2, dJ2, j_max, j_period, interp_V, interp_VM, species, lambda_val)
            k2_I, k2_J = compute_dI_dt_dJ_dt(dVeI2, dVeJ2, th2, dth2, dph2)
            if libs.np.isnan(k2_I) or libs.np.isnan(k2_J):
                return None, None

            # k3
            I3 = current_I + 0.5 * substep_dt * k2_I
            J3 = _wrap_J(current_J + 0.5 * substep_dt * k2_J, j_max, j_period)
            colat3 = interp_COLAT([J3, I3])[0]
            if libs.np.isnan(colat3):
                return None, None
            th3 = colat3 * libs.np.pi / 180.0
            dI3, dJ3, dth3, dph3 = compute_adaptive_steps(I3, J3, j_max, j_period, interp_COLAT, interp_ALOCT_sin, interp_ALOCT_cos)
            dVeI3, dVeJ3 = compute_dVeff_derivatives(I3, J3, dI3, dJ3, j_max, j_period, interp_V, interp_VM, species, lambda_val)
            k3_I, k3_J = compute_dI_dt_dJ_dt(dVeI3, dVeJ3, th3, dth3, dph3)
            if libs.np.isnan(k3_I) or libs.np.isnan(k3_J):
                return None, None

            # k4
            I4 = current_I + substep_dt * k3_I
            J4 = _wrap_J(current_J + substep_dt * k3_J, j_max, j_period)
            colat4 = interp_COLAT([J4, I4])[0]
            if libs.np.isnan(colat4):
                return None, None
            th4 = colat4 * libs.np.pi / 180.0
            dI4, dJ4, dth4, dph4 = compute_adaptive_steps(I4, J4, j_max, j_period, interp_COLAT, interp_ALOCT_sin, interp_ALOCT_cos)
            dVeI4, dVeJ4 = compute_dVeff_derivatives(I4, J4, dI4, dJ4, j_max, j_period, interp_V, interp_VM, species, lambda_val)
            k4_I, k4_J = compute_dI_dt_dJ_dt(dVeI4, dVeJ4, th4, dth4, dph4)
            if libs.np.isnan(k4_I) or libs.np.isnan(k4_J):
                return None, None

            new_I = current_I + (substep_dt / 6.0) * (k1_I + 2.0 * k2_I + 2.0 * k3_I + k4_I)
            new_J = current_J + (substep_dt / 6.0) * (k1_J + 2.0 * k2_J + 2.0 * k3_J + k4_J)

        else:
            raise ValueError(f"Unknown int_method '{int_method}'")

        # enforce topology each substep
        new_J = _wrap_J(new_J, j_max, j_period)
        new_I = _clamp_I(new_I, metadata["i_max"])

        current_I, current_J = new_I, new_J

    return current_I, current_J

# [move_particles.py] - AFTER

# Delete the old backtrack_particles function and replace it with this:

def move_particles_one_step(
    local_particles,        # List of particle dictionaries for this rank
    current_chunk,          # The integer chunk index for the current step
    metadata,
    interpolators,          # Interpolators for V, VM for the current chunk
    chunk_interpolators,    # Interpolators for geometry for the current chunk
    is_inside_ellipse       # The ellipse checking function
):
    """
    Advances a list of particles by one time step (-60s).
    Returns the updated list of particle dictionaries.
    """
    j_max = metadata['j_max']
    j_period = j_max - 3
    updated_particles = []

    for p in local_particles:
        # If particle is already stopped, just pass it through
        if not p.get('active', True):
            updated_particles.append(p)
            continue

        # Get field interpolators for this specific minute
        interp_X = chunk_interpolators['XMIN']
        interp_Y = chunk_interpolators['YMIN']

        # Perform the integration for one step
        new_I, new_J = update_position_new(
            p['I'], p['J'], metadata, params.int_method, j_max, j_period,
            {current_chunk: chunk_interpolators}, # update_position_new expects a dict
            {current_chunk: interpolators},
            current_chunk,
            p['species'], p['lambda']
        )

        # Create a copy of the particle to update
        p_new = p.copy()

        # Check for integration failure
        if new_I is None or libs.np.isnan(new_I) or new_J is None or libs.np.isnan(new_J):
            p_new.update({'active': False, 'stopped': True, 'final_X': p['X'], 'final_Y': p['Y'], 'stop_chunk': current_chunk})
            updated_particles.append(p_new)
            continue

        # Check if the particle left the boundary
        new_X = float(interp_X([new_J, new_I])[0])
        new_Y = float(interp_Y([new_J, new_I])[0])

        if libs.np.isnan(new_X) or libs.np.isnan(new_Y) or not is_inside_ellipse(new_X, new_Y):
            p_new.update({'active': False, 'stopped': True, 'final_X': p['X'], 'final_Y': p['Y'], 'stop_chunk': current_chunk})
        else:
            # Commit the new state
            p_new.update({'I': new_I, 'J': new_J, 'X': new_X, 'Y': new_Y})

        updated_particles.append(p_new)

    return updated_particles

