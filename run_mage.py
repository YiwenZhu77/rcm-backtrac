#!/usr/bin/env python3
"""Run Sina's backtracker on MAGE output — single process, uses original move_particles.py."""

import numpy as np
import h5py
import time
import sys
import os

# Add rcm_backtrac to path
sys.path.insert(0, os.path.dirname(__file__))

import libs
import params
import move_particles
import data_init
import tools

# ============ Config ============
N_STEPS = params.last_chunk  # 240 steps backward
OUT_TRAJ = '/path/to/pilot-data/sina_orig_1k_traj.npy'
os.makedirs(params.save_ani_file_path, exist_ok=True)

# ============ Load field data ============
print(f'Loading field data from {params.hdf5_file_path}...')
t0 = time.time()

f = h5py.File(params.hdf5_file_path, 'r')
metadata = data_init.load_metadata(f)
print(f'  Grid: i_max={metadata["i_max"]}, j_max={metadata["j_max"]}, chunks={metadata["num_chunks"]}')

# Pre-load all chunks
# precompute_interpolation_grids returns both geometry (XMIN,YMIN,COLAT,ALOCT) and field (V,VM) interps
all_interps = {}  # chunk -> {field -> interpolator}

for ci in range(N_STEPS + 1):
    chunk_data = data_init.load_chunk(f, ci, metadata, verbose=False)
    interps = data_init.precompute_interpolation_grids(None, {ci: chunk_data}, metadata)
    all_interps[ci] = interps[ci]

    if ci % 50 == 0:
        print(f'  Loaded chunk {ci}/{N_STEPS}')

f.close()
print(f'  Done in {time.time()-t0:.1f}s')

# ============ Load particles ============
print(f'Loading particles from {params.particle_file_path}...')
f_p = h5py.File(params.particle_file_path, 'r')
pnum = f_p['particles_numeric'][:]  # (N, 4): idx, I, J, lambda
ptimes = f_p['particles_time'][:]
pspecies = f_p['particles_species'][:]
ppressure = f_p['particles_pressure'][:]
f_p.close()

j_max = metadata['j_max']
j_period = j_max - 3

# Build ellipse checker
x0_e, y0_e, semimajor_e, semiminor_e, is_inside_ellipse = tools.plot_ellipse_boundary(
    a1=params.a1, a2=params.a2, semiminor=params.semiminor, plot=False, return_mask=True
)

# Initialize particle dicts
particles = []
for i in range(len(pnum)):
    idx, I, J, lam = pnum[i]
    chunk = params.last_chunk

    interp_X = all_interps[chunk]['XMIN']
    interp_Y = all_interps[chunk]['YMIN']
    x = float(interp_X([J, I])[0])
    y = float(interp_Y([J, I])[0])

    if np.isnan(x) or np.isnan(y):
        continue

    species = 'e' if int(pspecies[i]) == 1 else 'p'
    effective_lambda = 0.0 if params.set_lambda_to_zero else float(lam)

    particles.append({
        'idx': int(idx),
        'species': species,
        'I': float(I),
        'J': float(J),
        'X': x,
        'Y': y,
        'lambda': effective_lambda,
        'pressure': float(ppressure[i]),
        'chunk': chunk,
        'active': True,
        'stopped': False,
        'final_X': None,
        'final_Y': None,
        'group': 'initial',
    })

print(f'  Initialized {len(particles)} particles')

# ============ Integrate ============
print(f'Integrating {N_STEPS} steps backward...')

# Store trajectories
traj = np.full((N_STEPS + 1, len(particles), 2), np.nan)
for pi, p in enumerate(particles):
    traj[0, pi] = [p['X'], p['Y']]

t_start = time.time()
for step in range(N_STEPS):
    current_chunk = params.last_chunk - step
    if current_chunk < 0:
        break

    # Sina's move_particles_one_step expects:
    #   interpolators = {field: interp} for V, VM
    #   chunk_interpolators = {field: interp} for XMIN, YMIN, COLAT, ALOCT_sin, ALOCT_cos
    # Both come from the same precompute_interpolation_grids output
    interps_this = all_interps[current_chunk]

    particles = move_particles.move_particles_one_step(
        local_particles=particles,
        current_chunk=current_chunk,
        metadata=metadata,
        interpolators=interps_this,
        chunk_interpolators=interps_this,
        is_inside_ellipse=is_inside_ellipse,
    )

    for pi, p in enumerate(particles):
        if p.get('active', False) and not p.get('stopped', False):
            traj[step + 1, pi] = [p['X'], p['Y']]

    if step % 30 == 0:
        n_active = sum(1 for p in particles if p.get('active', False) and not p.get('stopped', False))
        print(f'  Step {step}/{N_STEPS}: {n_active} active, elapsed {time.time()-t_start:.0f}s')

t_total = time.time() - t_start
print(f'  Done in {t_total:.1f}s = {t_total/60:.1f} min')

# ============ Save ============
# Map back to 1k indices
res = np.full((N_STEPS + 1, 1000, 2), np.nan)
for pi, p in enumerate(particles):
    idx = p['idx']
    if 0 <= idx < 1000:
        res[:, idx] = traj[:, pi]

np.save(OUT_TRAJ, res)

n60 = np.sum(np.isfinite(res[60, :, 0]))
n120 = np.sum(np.isfinite(res[120, :, 0]))
n240 = np.sum(np.isfinite(res[min(240, N_STEPS), :, 0]))
print(f'\nSurvival: 60min={n60}, 120min={n120}, 240min={n240}')
print(f'Saved to {OUT_TRAJ}')
