# particle_init.py

import params
import libs
import tools
from typing import List, Dict, Tuple, Optional, Set
from data_init import seconds_to_chunk, is_minute_aligned

# Initialize MPI
comm = libs.MPI.COMM_WORLD
rank: int = comm.Get_rank()
size: int = comm.Get_size()


def save_initialization_png_mpi(
    local_particles,              # list[dict] from initialize_particles() on this rank
    metadata: Dict,
    chunk_data_dict: Dict[int, Dict],
    output_path: Optional[str] = None,
) -> None:
    """
    Gather particles across ranks and save an initialization scatter PNG.
    Uses existing X/Y in particle dicts; falls back to interpolating from I,J if needed.
    """
    if output_path is None:
        output_path = params.initial_setup_path

    # Gather all particles to rank 0
    parts_all = comm.gather(local_particles, root=0)
    if rank != 0:
        return

    # Flatten
    particles = [p for plist in parts_all for p in plist]
    if not particles:
        libs.logging.warning("[rank 0] No particles to plot; skipping PNG save.")
        return

    # Ensure output directory exists
    out_dir = libs.os.path.dirname(output_path)
    if out_dir:
        libs.os.makedirs(out_dir, exist_ok=True)

    xs, ys, specs, groups = [], [], [], []
    I_vals = libs.np.arange(1, metadata["i_max"] + 1, dtype=libs.np.float64)
    J_vals = libs.np.arange(1, metadata["j_max"] + 1, dtype=libs.np.float64)

    for p in particles:
        # p is a dict produced by initialize_particles()
        x = p.get('X')
        y = p.get('Y')
        sp = p.get('species', 'e')
        if (x is None or y is None) and ('I' in p and 'J' in p and 'chunk' in p):
            chunk = p['chunk']
            cd = chunk_data_dict.get(chunk, {})
            if 'XMIN' in cd and 'YMIN' in cd:
                try:
                    interp_X = libs.RegularGridInterpolator(
                        (J_vals, I_vals), cd['XMIN'], method='linear',
                        bounds_error=False, fill_value=libs.np.nan
                    )
                    interp_Y = libs.RegularGridInterpolator(
                        (J_vals, I_vals), cd['YMIN'], method='linear',
                        bounds_error=False, fill_value=libs.np.nan
                    )
                    x = float(interp_X([p['J'], p['I']])[0])
                    y = float(interp_Y([p['J'], p['I']])[0])
                except Exception:
                    x, y = None, None
        if x is None or y is None or libs.np.isnan(x) or libs.np.isnan(y):
            continue
        xs.append(x); ys.append(y); specs.append(sp)
        groups.append(p.get('group', 'initial'))

    if not xs:
        libs.logging.warning("[rank 0] All particle positions invalid; skipping PNG save.")
        return

    fig = libs.plt.figure(figsize=(8, 8))
    ax = libs.plt.gca()

    xs    = libs.np.array(xs   ); ys     = libs.np.array(ys    )
    specs = libs.np.array(specs); groups = libs.np.array(groups)

    styles = [
            ('e', 'initial', '^', 'Electrons (init)', dict(alpha=0.8)),
            ('e', 'lost'   , '^', 'Electrons (lost)', dict(alpha=0.9, facecolors='none', edgecolors='k')),
            ('p', 'initial', 'o', 'Protons   (init)', dict(alpha=0.8)),
            ('p', 'lost'   , 'o', 'Protons   (lost)', dict(alpha=0.9, facecolors='none', edgecolors='k')),
    ]
    for sp, grp, marker, label, kw in styles:
        m = (specs == sp) & (groups == grp)
        if m.any():
            ax.scatter(xs[m], ys[m], marker=marker, label=label, **kw)

    # Ellipse boundary + Earth
    try:
        tools.plot_ellipse_boundary(
            a1=params.a1, a2=params.a2, semiminor=params.semiminor,
            plot=True, return_mask=False, ax=ax
        )
    except Exception:
        pass
    try:
        tools.add_circle(ax, fill_black=True, zorder=10)
    except Exception:
        pass

    ax.set_xlabel('XMIN')
    ax.set_ylabel('YMIN')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.set_title('Final positions of particles')
    ax.set_xlim([10, -25])
    ax.set_ylim([15, -15])
    ax.legend(loc='upper right')

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    libs.plt.close(fig)
    libs.logging.info(f"[rank 0] Saved initialization plot to {output_path}")

def read_initial_particles(particle_file_path: str, metadata: Dict, max_initial_particles: Optional[int] = None) -> List[Tuple]:
    """
    Read initial particles from an HDF5 file for backtracking.

    Args:
        particle_file_path: Path to HDF5 file containing particle data.
        metadata: Dictionary with grid metadata (i_max, j_max).
        max_initial_particles: Maximum number of particles to read (default: None).

    Returns:
        List of tuples (time, idx, species, I, J, lambda_val, pressure).
    """
    start_time = libs.time.time()
    process = libs.psutil.Process()
    mem_before = process.memory_info().rss / 1024**2

    initial_particles = []
    # counter = 0

    if rank != 0:
        libs.logging.debug(f"Rank {rank}: Skipping read_initial_particles (only rank 0 reads)")
        return []


    try:
        if not libs.os.path.exists(particle_file_path) or not libs.os.access(particle_file_path, libs.os.R_OK):
            libs.logging.error(f"HDF5 file {particle_file_path} is not accessible")
            raise IOError(f"HDF5 file {particle_file_path} is not accessible")

        with libs.h5py.File(particle_file_path, 'r') as f:

            total_particles = len(f['particles_numeric']) if 'particles_numeric' in f else 0

            start_idx = 0
            end_idx = total_particles

            libs.logging.debug(f"Rank {rank}: Initial particles - total: {total_particles}, start_idx: {start_idx}, end_idx: {end_idx}")

            required_datasets = ['particles_numeric', 'particles_species', 'particles_pressure', 'particles_time']
            if not all(ds in f for ds in required_datasets):
                libs.logging.error(f"HDF5 file {particle_file_path} missing required datasets: {required_datasets}")
                raise KeyError(f"HDF5 file must contain {required_datasets}")

            particles_numeric = f['particles_numeric'][start_idx:end_idx]
            particles_species = f['particles_species'][start_idx:end_idx]
            particles_pressure = f['particles_pressure'][start_idx:end_idx]
            particles_time = f['particles_time'][start_idx:end_idx]

            if not (len(particles_numeric) == len(particles_species) == len(particles_pressure) == len(particles_time)):
                libs.logging.error(f"Rank {rank}: Mismatch in dataset lengths in {particle_file_path}")
                raise ValueError("Mismatch in lengths of particle datasets")

            species_map = {1: 'e', 2: 'p'}
            for i in range(len(particles_numeric)):
                # if max_initial_particles is not None and counter >= max_initial_particles:
                #    libs.logging.debug(f"Rank {rank}: Reached max_initial_particles {max_initial_particles}, stopping")
                #    break
                idx, I, J, lambda_val = particles_numeric[i]
                particle_time = particles_time[i]
                species_int = particles_species[i]
                pressure = particles_pressure[i]

                libs.logging.debug(
                    f"Rank {rank}: Particle {idx} - time: {particle_time}, idx: {idx}, species: {species_int}, "
                    f"I: {I}, J: {J}, lambda: {lambda_val}, pressure: {pressure}"
                )

                try:
                    particle_time = float(particle_time)
                    idx = int(idx)
                    I = float(I)
                    J = float(J)
                    lambda_val = float(lambda_val)
                    species_int = int(species_int)
                    pressure = float(pressure)
                except (TypeError, ValueError) as e:
                    libs.logging.warning(f"Rank {rank}: Invalid type in initial particle fields: {str(e)}, skipping")
                    continue

                if any(libs.np.isnan(x) for x in [particle_time, idx, species_int, I, J, lambda_val, pressure]):
                    libs.logging.warning(f"Rank {rank}: Particle {idx} has NaN values, skipping")
                    continue

                if species_int not in species_map:
                    libs.logging.warning(f"Rank {rank}: Invalid species {species_int} for particle {idx}, skipping")
                    continue
                species = species_map[species_int]

                if not (1 <= I <= metadata["i_max"] and 1 <= J <= metadata["j_max"]):
                    libs.logging.warning(f"Rank {rank}: Particle {idx} ({species}) has invalid position (I={I}, J={J}), skipping")
                    continue

                initial_particles.append((particle_time, idx, species, I, J, lambda_val, pressure))
                # counter += 1

            libs.logging.info(f"Rank {rank}: Read {initial_particles} initial particles from {particle_file_path}")

            # --- NEW SHUFFLE AND SLICE LOGIC ---
            shuffle_initial = getattr(params, "shuffle_initial", False)
            if max_initial_particles is not None and len(initial_particles) > max_initial_particles:
                if shuffle_initial:
                    libs.logging.info(f"Rank {rank}: Shuffling and selecting {max_initial_particles} particles...")
                    libs.random.shuffle(initial_particles)
                    initial_particles = initial_particles[:max_initial_particles]
                    libs.logging.info(f"Rank {rank}: Randomly selected {len(initial_particles)} particles to process.")
                else:
                    # Sequential "first-N" selection
                    libs.logging.info(f"Rank {rank}: Selecting the first {max_initial_particles} particles (sequential)...")
                    initial_particles = initial_particles[:max_initial_particles]
                    libs.logging.info(f"Rank {rank}: Sequentially selected {len(initial_particles)} particles.")

    except Exception as e:
        libs.logging.error(f"Rank {rank}: Failed to read initial particles from {particle_file_path}: {str(e)}", exc_info=True)
        raise

    elapsed_time = libs.time.time() - start_time
    mem_after = process.memory_info().rss / 1024**2
    libs.logging.debug(f"Rank {rank}: read_initial_particles took {elapsed_time:.2f}s, memory used: {mem_after - mem_before:.2f} MB")

    return initial_particles

def read_lost_particles(loss_file_path: str, max_lost_particles: Optional[int] = None) -> List[Tuple]:
    """
    Read lost particles from an HDF5 file that may be in one of two layouts:
      (1) top-level datasets: loss_time, idx, particle_type, I, J, lambda_val, pressure
      (2) grouped layout: grid_*/{particles_numeric, particles_species, particles_pressure, particles_time}
    Returns a list of tuples (particle_time, idx, species, I, J, lambda_val, pressure).
    """
    start_time = libs.time.time()
    process = libs.psutil.Process()
    mem_before = process.memory_info().rss / 1024**2

    loss_data: List[Tuple] = []
    counter = 0

    # Config: whether/how to offset lost-particle IDs to avoid collisions with initial IDs
    apply_offset = getattr(params, "apply_lost_id_offset", True)
    offset_val   = int(getattr(params, "lost_id_offset", 10000))

    if rank != 0:
        libs.logging.debug(f"Rank {rank}: Skipping read_lost_particles (only rank 0 reads)")
        return []

    try:
        if not libs.os.path.exists(loss_file_path) or not libs.os.access(loss_file_path, libs.os.R_OK):
            libs.logging.error(f"HDF5 file {loss_file_path} is not accessible")
            raise IOError(f"HDF5 file {loss_file_path} is not accessible")

        with libs.h5py.File(loss_file_path, 'r') as loss_f:

            def push(particle_time, idx_val, species, I_val, J_val, lam_val, pressure_val):
                nonlocal counter
                if max_lost_particles is not None and counter >= max_lost_particles:
                    return False
                # Optional cadence check (we keep your minute logic as-is)
                if not is_minute_aligned(particle_time):
                    libs.logging.warning(
                        f"Rank {rank}: Lost particle time {particle_time} not minute-aligned; will coerce downstream."
                    )
                loss_data.append((
                    float(particle_time),
                    int(idx_val),
                    species,
                    float(I_val),
                    float(J_val),
                    float(lam_val),
                    float(pressure_val),
                ))
                counter += 1
                return True

            # Layout (1): Top-level datasets exist
            top_level = all(k in loss_f for k in ['loss_time','idx','particle_type','I','J','lambda_val','pressure'])
            if top_level:
                loss_time = loss_f['loss_time'][:]
                idx_arr   = loss_f['idx'][:]
                ptype     = loss_f['particle_type'][:]
                I_arr     = loss_f['I'][:]
                J_arr     = loss_f['J'][:]
                lam_arr   = loss_f['lambda_val'][:]
                pres_arr  = loss_f['pressure'][:]

                total = len(loss_time)
                libs.logging.debug(f"Rank {rank}: Lost particles (top-level) total: {total}")

                for i in range(total):
                    species = 'e' if int(ptype[i]) == 1 else 'p'
                    idx_val = int(idx_arr[i])
                    if apply_offset:
                        idx_val += offset_val
                    if not push(loss_time[i], idx_val, species, I_arr[i], J_arr[i], lam_arr[i], pres_arr[i]):
                        break

            else:
                # Layout (2): grouped under grid_* keys
                group_keys = [g for g in loss_f.keys() if g.startswith('grid_')]
                libs.logging.debug(f"Rank {rank}: Lost particles grouped layout, groups: {group_keys}")
                for g in group_keys:
                    grp = loss_f[g]
                    if not all(k in grp for k in ['particles_numeric','particles_species','particles_pressure','particles_time']):
                        libs.logging.warning(f"Rank {rank}: Group {g} missing required datasets; skipping")
                        continue

                    # particles_numeric columns: idx, I, J, lambda
                    num   = grp['particles_numeric'][:]
                    spec  = grp['particles_species'][:]
                    pres  = grp['particles_pressure'][:]
                    times = grp['particles_time'][:]

                    total = len(num)
                    for i in range(total):
                        idx_val, I_val, J_val, lam_val = num[i]
                        s = spec[i]
                        species = s.decode('utf-8') if isinstance(s, bytes) else ('e' if int(s) == 1 else 'p')

                        idx_val = int(idx_val)
                        if apply_offset:
                            idx_val += offset_val

                        if not push(times[i], idx_val, species, I_val, J_val, lam_val, pres[i]):
                            break

            libs.logging.info(f"Rank {rank}: Read {counter} lost particles from {loss_file_path}")

    except Exception as e:
        libs.logging.error(f"Rank {rank}: Failed to read lost particles from {loss_file_path}: {str(e)}", exc_info=True)
        raise

    elapsed_time = libs.time.time() - start_time
    mem_after = process.memory_info().rss / 1024**2
    libs.logging.debug(f"Rank {rank}: read_lost_particles took {elapsed_time:.2f}s, memory used: {mem_after - mem_before:.2f} MB")

    return loss_data

def distribute_particles(
    particle_file_path: str,
    loss_file_path: str,
    metadata: Dict,
    max_initial_particles: Optional[int] = None,
    max_lost_particles: Optional[int] = None
) -> Tuple[List[Tuple], Set[int]]:
    """
    Read and distribute particles across ranks, compute required chunks.

    Args:
        particle_file_path: Path to initial particles HDF5 file.
        loss_file_path: Path to lost particles HDF5 file.
        metadata: Dictionary with grid metadata.
        max_initial_particles: Maximum initial particles to read.
        max_lost_particles: Maximum lost particles to read.

    Returns:
        Tuple of (local particles, required chunk indices).
    """
    start_time = libs.time.time()
    process = libs.psutil.Process()
    mem_before = process.memory_info().rss / 1024**2

    all_particles = []
    required_chunks: Set[int] = set()

    if rank == 0:
        try:
            libs.logging.info(f"Rank {rank}: Reading initial particles")
            all_initial_particles = read_initial_particles(particle_file_path, metadata, max_initial_particles)

            libs.logging.info(f"Rank {rank}: Reading lost particles")
            all_lost_particles = read_lost_particles(loss_file_path, max_lost_particles)

            # Optional sanity check: verify there are no overlapping IDs between initial and lost sets.
            apply_offset = getattr(params, "apply_lost_id_offset", True)
            if apply_offset:
                init_ids = {int(p[1]) for p in all_initial_particles}  # idx at position 1
                lost_ids = {int(p[1]) for p in all_lost_particles}
                dup = init_ids & lost_ids
                if dup:
                    sample = sorted(list(dup))[:10]
                    libs.logging.warning(
                        f"Rank {rank}: Lost-id offset is ON, but duplicates still found (showing up to 10): {sample}"
                    )

            all_particles = all_initial_particles + all_lost_particles

            for p in all_particles:
                birth_time = float(p[0])
                # Enforce cadence alignment; warn and continue if slightly off.
                if not is_minute_aligned(birth_time):
                    libs.logging.warning(
                        f"Rank {rank}: birth_time {birth_time} not minute-aligned; coercing via seconds_to_chunk(round)."
                    )
                chunk = seconds_to_chunk(birth_time, rounding="round")

                if params.first_chunk <= chunk <= params.last_chunk:
                    required_chunks.add(chunk)
                else:
                    libs.logging.warning(
                        f"Rank {rank}: Particle {p[1]} ({p[2]}): Invalid birth time {birth_time}s (chunk {chunk}), skipping"
                    )

            libs.logging.debug(f"Rank {rank}: All particles: {len(all_particles)}, IDs: {[p[1] for p in all_particles]}")
            libs.logging.debug(f"Rank {rank}: Required chunks: {sorted(required_chunks)}")
        except Exception as e:
            libs.logging.error(f"Rank {rank}: Failed to read particles: {str(e)}", exc_info=True)
            raise
    else:
        all_particles = []

    # Broadcast required chunks
    required_chunks = comm.bcast(required_chunks, root=0)
    mins = sorted(required_chunks)
    if not mins:
        if rank == 0:
            libs.logging.error("No particles survived reading/filtering; required_chunks is empty. "
                               "Check R_MIN/R_MAX units and filtering logic.")
        return [], set()  # or raise a clean, descriptive error
    full = list(range(min(mins), max(mins) + 1))
    if mins != full:
        if rank == 0:
            libs.logging.warning("Fixing non-contiguous required_chunks: %s -> %s", mins, full)
        required_chunks = set(full)
    required_chunks = comm.bcast(required_chunks, root=0)

    libs.logging.debug(f"Rank {rank}: Received required chunks: {sorted(required_chunks)}")

    # Compute distribution sizes
    N = len(all_particles) if rank == 0 else 0
    N = comm.bcast(N, root=0)
    counts = [(N // size) + (1 if i < N % size else 0) for i in range(size)]
    displacements = [sum(counts[:i]) for i in range(size)]
    fields_per_particle = 7  # time, idx, species, I, J, lambda_val, pressure

    # Flatten particles to a 1D array of floats
    if rank == 0:
        particle_array = libs.np.array(
            [val for p in all_particles for val in (
                p[0], float(p[1]), 1.0 if p[2] == 'e' else 2.0, p[3], p[4], p[5], p[6]
            )],
            dtype=libs.np.float64
        )
        libs.logging.debug(f"Rank {rank}: particle_array shape: {particle_array.shape}, sample: {particle_array[:fields_per_particle*2]}")
    else:
        particle_array = None

    # Allocate buffer for local particles
    local_particle_count = counts[rank]
    local_particles = libs.np.empty(local_particle_count * fields_per_particle, dtype=libs.np.float64)
    libs.logging.debug(f"Rank {rank}: local_particles shape: {local_particles.shape}")

    # Scatter particles
    scatter_counts = [c * fields_per_particle for c in counts]
    scatter_displacements = [d * fields_per_particle for d in displacements]
    libs.logging.debug(f"Rank {rank}: Scattering N={N}, counts={scatter_counts}, displacements={scatter_displacements}")
    comm.Scatterv([particle_array, scatter_counts, scatter_displacements, libs.MPI.DOUBLE], local_particles, root=0)

    # Reshape and convert back to list of tuples
    local_particles = local_particles.reshape(local_particle_count, fields_per_particle)
    species_map = {1.0: 'e', 2.0: 'p'}
    local_particles = [
        (
            p[0],              # time
            int(p[1]),         # idx
            species_map[p[2]], # species
            p[3],              # I
            p[4],              # J
            p[5],              # lambda_val
            p[6]               # pressure
        )
        for p in local_particles
    ]
    libs.logging.debug(f"Rank {rank}: Received {len(local_particles)} local particles: {[p[1] for p in local_particles]}")

    elapsed_time = libs.time.time() - start_time
    mem_after = process.memory_info().rss / 1024**2
    libs.logging.debug(f"Rank {rank}: distribute_particles took {elapsed_time:.2f}s, memory used: {mem_after - mem_before:.2f} MB")

    return local_particles, required_chunks

def particle_generator(loss_data: List[Tuple], metadata: Dict, chunk_data_dict: Dict) -> List[Tuple]:
    """
    Generate new particles from lost particle data, validating positions.

    Args:
        loss_data: List of tuples (birth_time, idx, species, I, J, lambda_val, pressure).
        metadata: Dictionary with grid metadata (i_max, j_max).
        chunk_data_dict: Dictionary mapping chunk indices to chunk data.

    Returns:
        List of validated particle tuples.
    """
    start_time = libs.time.time()
    process = libs.psutil.Process()
    mem_before = process.memory_info().rss / 1024**2

    new_particles = []
    for birth_time, idx, species, I, J, lambda_val, pressure in loss_data:
        if not is_minute_aligned(birth_time):
            libs.logging.warning(
                f"Rank {rank}: birth_time {birth_time} not minute-aligned; coercing via seconds_to_chunk(round)."
            )
        chunk = seconds_to_chunk(birth_time, rounding="round")
        libs.logging.debug(f"Rank {rank}: Processing particle {idx} ({species}) - time: {birth_time}, chunk: {chunk}, I: {I}, J: {J}")

        if chunk < params.first_chunk or chunk > params.last_chunk:
            libs.logging.warning(f"Rank {rank}: Particle {idx} ({species}): Invalid birth time {birth_time}s (chunk {chunk}), skipping")
            continue
        if chunk not in chunk_data_dict:
            libs.logging.warning(f"Rank {rank}: Particle {idx} ({species}): No data for chunk {chunk}, skipping")
            continue

        chunk_data = chunk_data_dict[chunk]
        try:
            for dataset in ['XMIN', 'YMIN']:
                if dataset not in chunk_data:
                    libs.logging.warning(f"Rank {rank}: Particle {idx} ({species}): Dataset '{dataset}' missing in chunk {chunk}, skipping")
                    raise KeyError(f"Dataset '{dataset}' missing")

            I_vals = libs.np.arange(1, metadata["i_max"] + 1, dtype=libs.np.float64)
            J_vals = libs.np.arange(1, metadata["j_max"] + 1, dtype=libs.np.float64)
            interp_X = libs.RegularGridInterpolator((J_vals, I_vals), chunk_data['XMIN'], method='linear', bounds_error=False, fill_value=libs.np.nan)
            interp_Y = libs.RegularGridInterpolator((J_vals, I_vals), chunk_data['YMIN'], method='linear', bounds_error=False, fill_value=libs.np.nan)

            x0, y0, semimajor, semiminor, is_inside_ellipse = tools.plot_ellipse_boundary(
                a1=params.a1, a2=params.a2, semiminor=params.semiminor, plot=False, return_mask=True
            )

            # --- PERIODICITY FIRST (match initialize_particles) ---
            j_max = metadata["j_max"]
            j_period = j_max - 3
            if J < 1 or J > j_max:
                J = ((J - 1) % j_period) + 1
                libs.logging.debug(f"Rank {rank}: Adjusted J for particle {idx} to {J} (periodicity)")

            # --- THEN BOUNDS CHECK/CLAMP ---
            if not (1 <= I <= metadata["i_max"] and 1 <= J <= metadata[j_max]):
                libs.logging.warning(f"Rank {rank}: Particle {idx} ({species}): Invalid position (J={J}, I={I}) at chunk {chunk}, adjusting")
                I = max(1, min(I, metadata["i_max"]))
                J = max(1, min(J, metadata["j_max"]))

            x = float(interp_X([J, I])[0])
            y = float(interp_Y([J, I])[0])
            if libs.np.isnan(x) or libs.np.isnan(y):
                libs.logging.warning(f"Rank {rank}: Particle {idx} ({species}): Interpolation failed at (J={J}, I={I}), x={x}, y={y}, skipping")
                continue

            if not is_inside_ellipse(x, y):
                dist = ((x - x0)**2 / semimajor**2) + ((y - y0)**2 / semiminor**2)
                libs.logging.warning(
                    f"Rank {rank}: Particle {idx} ({species}): Position (X={x:.6f}, Y={y:.6f}) outside ellipse, distance={dist:.6f}, skipping"
                )
                continue

            new_particles.append((birth_time, idx, species, I, J, lambda_val, pressure))
            libs.logging.debug(f"Rank {rank}: Generated particle {idx} ({species}) at chunk {chunk}")

        except Exception as e:
            libs.logging.error(f"Rank {rank}: Failed to process particle {idx} ({species}) at chunk {chunk}: {str(e)}")
            continue

    if not new_particles:
        libs.logging.warning(f"Rank {rank}: No new particles generated from loss_data (length: {len(loss_data)})")

    elapsed_time = libs.time.time() - start_time
    mem_after = process.memory_info().rss / 1024**2
    libs.logging.debug(f"Rank {rank}: particle_generator took {elapsed_time:.2f}s, memory used: {mem_after - mem_before:.2f} MB")

    return new_particles

def initialize_particles(
    chunk_data_dict: Dict[int, Dict],
    metadata: Dict,
    particles_to_process: List[Tuple],
    start_chunk: int = params.last_chunk,
    verbose: bool = True,
    set_lambda_to_zero: bool = params.set_lambda_to_zero
) -> List[Dict]:
    """
    Initialize particles with validated positions and properties.

    Args:
        chunk_data_dict: Dictionary mapping chunk indices to chunk data.
        metadata: Dictionary with grid metadata (i_max, j_max).
        initial_particles: List of tuples (init_time, idx, species, I, J, lambda_val, pressure).
        start_chunk: Starting chunk index (default: last_chunk).
        verbose: Enable detailed logging (default: True).
        set_lambda_to_zero: Set lambda to 0.0 (default: set_lambda_to_zero).

    Returns:
        List of particle dictionaries for the current MPI rank.
    """
    start_time = libs.time.time()
    process = libs.psutil.Process()
    mem_before = process.memory_info().rss / 1024**2

    local_particles = particles_to_process
    libs.logging.debug(f"Rank {rank}: Initializing {len(local_particles)} particles: {[p[1] for p in local_particles]}")

    particles: List[Dict] = []
    processed_ids = set()  # Track processed particle IDs to avoid duplicates

    for init_time, idx, species, I, J, lambda_val, pressure, *maybe_group in local_particles:
        group = maybe_group[0] if maybe_group else 'initial'
        if idx in processed_ids:
            libs.logging.debug(f"Rank {rank}: Skipping duplicate particle {idx} ({species})")
            continue
        processed_ids.add(idx)

        if not is_minute_aligned(init_time):
            libs.logging.warning(
                f"Rank {rank}: init_time {init_time} not minute-aligned; coercing via seconds_to_chunk(round)."
            )
        chunk = seconds_to_chunk(init_time, rounding="round")
        libs.logging.debug(f"Rank {rank}: Processing particle {idx} ({species}) - time: {init_time}, chunk: {chunk}, I: {I}, J: {J}")

        if chunk < params.first_chunk or chunk > params.last_chunk:
            libs.logging.warning(f"Rank {rank}: Particle {idx} ({species}): Invalid initialization time {init_time}s (chunk {chunk}), skipping")
            continue
        if chunk not in chunk_data_dict:
            libs.logging.warning(f"Rank {rank}: Particle {idx} ({species}): No data for chunk {chunk}, skipping")
            continue

        try:
            chunk_data = chunk_data_dict[chunk]
            for dataset in ['XMIN', 'YMIN']:
                if dataset not in chunk_data:
                    libs.logging.warning(f"Rank {rank}: Particle {idx} ({species}): Dataset '{dataset}' missing in chunk {chunk}, skipping")
                    raise KeyError(f"Dataset '{dataset}' missing")

            I_vals = libs.np.arange(1, metadata["i_max"] + 1, dtype=libs.np.float64)
            J_vals = libs.np.arange(1, metadata["j_max"] + 1, dtype=libs.np.float64)
            interp_X = libs.RegularGridInterpolator((J_vals, I_vals), chunk_data['XMIN'], method='linear', bounds_error=False, fill_value=libs.np.nan)
            interp_Y = libs.RegularGridInterpolator((J_vals, I_vals), chunk_data['YMIN'], method='linear', bounds_error=False, fill_value=libs.np.nan)

            x0, y0, semimajor, semiminor, is_inside_ellipse = tools.plot_ellipse_boundary(
                a1=params.a1, a2=params.a2, semiminor=params.semiminor, plot=False, return_mask=True
            )

            if J < 1:
                J += (metadata["j_max"] - 3)
                libs.logging.debug(f"Rank {rank}: Adjusted J for particle {idx} to {J} (periodicity)")
            elif J > metadata["j_max"]:
                J -= (metadata["j_max"] - 3)
                libs.logging.debug(f"Rank {rank}: Adjusted J for particle {idx} to {J} (periodicity)")

            if not (1 <= I <= metadata["i_max"] and 1 <= J <= metadata["j_max"]):
                libs.logging.warning(f"Rank {rank}: Particle {idx} ({species}): Invalid position (J={J}, I={I}), skipping")
                continue

            x = float(interp_X([J, I])[0]); y = float(interp_Y([J, I])[0])
            rmin = getattr(params, "R_MIN", None)
            rmax = getattr(params, "R_MAX", None)
            if (rmin is not None or rmax is not None):
                _rmin = rmin if rmin is not None else 0.0
                _rmax = rmax if rmax is not None else float('inf')
                r_xy = (x*x + y*y) ** 0.5
                if not (_rmin <= r_xy <= _rmax):
                    libs.logging.debug(f"Rank {rank}: Particle {idx} rejected by radius cut r={r_xy:.3f} not in [{_rmin},{_rmax}]")
                    continue
            if libs.np.isnan(x) or libs.np.isnan(y):
                libs.logging.warning(f"Rank {rank}: Particle {idx} ({species}): Interpolation failed at (J={J}, I={I}), x={x}, y={y}, skipping")
                continue

            if not is_inside_ellipse(x, y):
                dist = ((x - x0)**2 / semimajor**2) + ((y - y0)**2 / semiminor**2)
                libs.logging.warning(
                    f"Rank {rank}: Particle {idx} ({species}): (X={x:.6f}, Y={y:.6f}) outside ellipse, distance={dist:.6f}, skipping"
                )
                continue

            effective_lambda = 0.0 if set_lambda_to_zero else lambda_val
            particle = {
                'time': init_time,
                'idx': idx,
                'species': species,
                'I': I,
                'J': J,
                'X': x,
                'Y': y,
                'lambda': effective_lambda,
                'pressure': pressure,
                'chunk': chunk,
                'active': chunk == params.last_chunk,
                'stopped': False,
                'final_X': None,
                'final_Y': None,
                'group': group
            }
            particles.append(particle)
            libs.logging.debug(f"Rank {rank}: Initialized particle {idx} ({species}) at chunk {chunk}")

        except Exception as e:
            libs.logging.error(f"Rank {rank}: Failed to initialize particle {idx} ({species}) at chunk {chunk}: {str(e)}")
            continue

    if not particles:
        libs.logging.warning(f"Rank {rank}: No particles initialized from {len(local_particles)} assigned")

    libs.logging.debug(f"Rank {rank}: Final initialized particle IDs: {sorted(processed_ids)}")

    elapsed_time = libs.time.time() - start_time
    mem_after = process.memory_info().rss / 1024**2
    libs.logging.debug(f"Rank {rank}: initialize_particles took {elapsed_time:.2f}s, memory used: {mem_after - mem_before:.2f} MB")

    return particles

