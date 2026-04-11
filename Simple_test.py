import params
import argparse
import tools
import libs
import data_init
import particle_init
import move_particles
import plotimate
import mpi4py
from mpi4py import MPI
import logging
from typing import List, Dict, Tuple, Set, Optional
import logging as _logging
import matplotlib
from scipy.interpolate import RegularGridInterpolator

# --- MPI ---------------------------------------------------------------
comm = libs.MPI.COMM_WORLD
rank: int = comm.Get_rank()
size: int = comm.Get_size()

# Have only Rank 0 print the version and setup information once.
if rank == 0:
    print(f"Python version: {libs.sys.version.split()[0]}")
    print(f"mpi4py: {mpi4py.__version__}")
    print(libs.MPI.Get_library_version())
    print("-" * 40) # Add a separator for clarity

# --- Define a log directory ---
log_dir = "logs"
if rank == 0:
    libs.os.makedirs(log_dir, exist_ok=True)
comm.Barrier() # Ensure directory is created before other ranks proceed

# Configure logging
log_file_path = libs.os.path.join(log_dir, f'backtrack_simulation.rank{rank}.log')
logging.basicConfig(
    filename=log_file_path,
    filemode='w',
    format=f'%(asctime)s [%(levelname)s] [rank {rank}] %(message)s',
    level=logging.DEBUG,  
    force=True
)

# ---- QUIET 3rd-party loggers ASAP (right after basicConfig) ----
for _name in ("matplotlib", "matplotlib.font_manager", "PIL", "fontTools"):
    logging.getLogger(_name).setLevel(logging.WARNING)
import matplotlib as _mpl
_mpl.set_loglevel("warning")  # Matplotlib's own switch

class RankFilter(logging.Filter):
    def filter(self, record):
        record.rank = rank
        return True

logging.getLogger().addFilter(RankFilter())

def main() -> None:

    run_start_time = libs.time.time()
    
    parser = argparse.ArgumentParser(description="Run a particle backtracking simulation or render an animation.")
    parser.add_argument('--render-only', action='store_true', help="Only render the animation from an existing HDF5 file.")
    parser.add_argument('--input', type=str, default=params.output_hdf5_path, help="Input HDF5 file for rendering.")
    parser.add_argument('--output', type=str, default=params.animation_file_path, help="Output MP4 file name.")
    args = parser.parse_args()
    
    if args.render_only:
        # --- RENDER-ONLY MODE ---
        if rank == 0:
            logging.info("--- Starting in Render-Only Mode ---")
            logging.info(f"Input HDF5: {args.input}")
            logging.info(f"Output MP4: {args.output}")

        try:
            plotimate.render_animation_from_h5_mpi(
                sim_h5_path=args.input,
                out_mp4=args.output,
                fps=getattr(params, "animation_fps", 4)
            )
        except Exception as e:
            logging.exception(f"Rank {rank}: animation failed: {e}")
        
    else:
        logging.info(f"Number of MPI processes: {size}")

        # --- CLEAN START STATUS FILE ---
        status_file_path = "run_status.json"
        if rank == 0:
            init_status = {
                "stage": "Initializing",
                "current_chunk": None,
                "start_chunk": None,
                "stop_chunk": None,
                "progress_percent": "0.00%",
                "elapsed_time_str": "00:00:00",
                "eta_str": "00:00:00"
            }
            with open(status_file_path, "w") as f:
                libs.json.dump(init_status, f, indent=4)
        comm.Barrier()
        # --- END CLEAN ---
        
        if rank == 0 and not libs.os.path.exists(params.hdf5_file_path):
            logging.error(f"FATAL: HDF5 file does not exist: {params.hdf5_file_path}")
            comm.Abort(1)
        comm.Barrier()

        t0 = libs.time.time()
        try:
            # --- Open field HDF5 in parallel READ mode ------------------------
            logging.info(f"[rank {rank}] About to open field HDF5 file: {params.hdf5_file_path}")
            logging.info(f"[rank {rank}] Using driver='mpio', locking=False")
            logging.info(f"[rank {rank}] Awaiting HDF5 open...")

            with libs.h5py.File(params.hdf5_file_path, "r", driver="mpio", comm=comm, locking=False) as f:
                # Metadata (0 loads + bcast; others just receive)
                metadata = data_init.load_metadata(f)

                max_init = getattr(params, "MAX_INITIAL_PARTICLES", None)
                max_lost = 0 if not getattr(params, "include_lost_particles", True) else getattr(params, "MAX_LOST_PARTICLES", None)

                # Distribute particles (already includes the +10000 lost-id offset if enabled in particle_init.py)
                local_particles, required_chunks = particle_init.distribute_particles(
                    particle_file_path=params.particle_file_path,
                    loss_file_path=params.loss_data_path,
                    metadata=metadata,
                    max_initial_particles = max_init,
                    max_lost_particles = max_lost,
                )

                # --- SANITY CHECK: no particles loaded means something is wrong ---
                total_local = len(local_particles)
                total_global = comm.allreduce(total_local, op=libs.MPI.SUM)

                if rank == 0:
                    logging.info(f"Total particles loaded across all ranks: {total_global}")

                if total_global == 0:
                    if rank == 0:
                        logging.error("FATAL: No particles were loaded! Check file paths or HDF5 I/O.")
                    comm.Abort(1)


                logging.info(f"[rank {rank}] received {len(local_particles)} local particles")

                # Expand each particle’s minute range: from its start chunk down to first_chunk (inclusive)
                # expanded = set()
                # for (t_birth, _idx, _sp, _I, _J, _lam, _pres) in local_particles:
                #    start_ch = data_init.seconds_to_chunk(t_birth, rounding="round")
                #    for ch in data_init.iter_minute_chunks(start_ch, params.first_chunk, descending=True):
                #        expanded.add(int(ch))
                # union across all ranks so everyone loads the same minute slices
                # all_required_sets = comm.allgather(expanded)
                # required_chunks = sorted(set().union(*all_required_sets), reverse=True)
                # logging.info(f"[rank {rank}] required minute-chunks = {required_chunks}")

                # --- Load chunks exactly once per rank -------------------------

                # Determine the full time range for the simulation
                start_chunk = max(required_chunks) if required_chunks else params.last_chunk
                stop_chunk  = params.first_chunk
                # stop_chunk  = min(required_chunks) if required_chunks else params.first_chunk
                logging.info(f"[rank {rank}] Simulation will run from chunk {start_chunk} down to {stop_chunk}")

            # --- Initialize Particles
            # NOTE: We need one chunk of data just to initialize particles at their start times.

            initial_data_to_load = {ch for ch in required_chunks}
            chunk_data_for_init = {}
            with libs.h5py.File(params.hdf5_file_path, "r", driver="mpio", comm=comm) as f:
                for ch in initial_data_to_load:
                    chunk_data_for_init[int(ch)] = data_init.load_chunk(f, int(ch), metadata, verbose=False)

            # Split local particles into initial-at-last-chunk and born-elsewhere (“lost”)
            last_chunk_sec = data_init.chunk_to_seconds(params.last_chunk)
            initial_particles = [p for p in local_particles if data_init.seconds_to_chunk(p[0], rounding="round") == params.last_chunk]
            lost_particles    = [p for p in local_particles if data_init.seconds_to_chunk(p[0], rounding="round") != params.last_chunk]

            logging.debug(f"Rank {rank}: initial={len(initial_particles)} (ids={[p[1] for p in initial_particles]})")
            logging.debug(f"Rank {rank}: lost   ={len(lost_particles)} (ids={[p[1] for p in lost_particles]})")

            # Turn “lost” tuples into spawnable particles (the function already validates ellipse & grid)
            # Create a unified list that includes the group label
            initial_particles_with_group = [p + ('initial',) for p in initial_particles]
            lost_particles_with_group = [p + ('lost',) for p in lost_particles]
            all_inits = initial_particles_with_group + lost_particles_with_group
            # generated_from_lost = particle_init.particle_generator(lost_particles, metadata, chunk_data_for_init)
            # all_inits = initial_particles + generated_from_lost

            # Initialize to phase space (I,J)->(X,Y) with minute alignment checks
            particles = particle_init.initialize_particles(chunk_data_dict = chunk_data_for_init, 
                                                           metadata = metadata, 
                                                           particles_to_process = all_inits
                                                           )

            particle_init.save_initialization_png_mpi(local_particles=particles, metadata=metadata, chunk_data_dict=chunk_data_for_init)

            # We can now clear this initial data to save memory
            del chunk_data_for_init

            # Create the ellipse checker once
            _, _, _, _, is_inside_ellipse = tools.plot_ellipse_boundary(
                a1=params.a1, a2=params.a2, semiminor=params.semiminor, plot=False, return_mask=True
            )
            
            # Each rank creates its own temporary output file
            temp_output_path = f"{params.output_hdf5_path}.rank{rank}.tmp.npy"
            # Ensure the temp file is fresh for each run
            if libs.os.path.exists(temp_output_path):
                libs.os.remove(temp_output_path)

            # --- NEW MAIN LOOP: Backtrack one chunk at a time ---
            if rank == 0:
                # This calculation now happens only one time.
                total_chunks = start_chunk - stop_chunk

            logging.info(f"[rank {rank}] Initialization complete. Entering backtracking loop.")
            logging.info(f"Rank {rank}: Starting backtracking from chunk {start_chunk} down to {stop_chunk}")
            for current_chunk in range(start_chunk, stop_chunk -1, -1):
                    logging.debug(f"Processing chunk {current_chunk}")

                    # 1. Load data for ONLY the current chunk
                    with libs.h5py.File(params.hdf5_file_path, "r", driver="mpio", comm=comm, locking=False) as f:
                        chunk_data = data_init.load_chunk(f, current_chunk, metadata, verbose=False)
        
                    # 2. Create interpolators for ONLY the current chunk
                    interpolators = data_init.precompute_interpolation_grids(None, {current_chunk: chunk_data}, metadata)
        
                    # Activate particles whose birth time matches this chunk
                    # OLD way 11/9 #
                    for p in particles:
                        if p['chunk'] == current_chunk:
                            p['active'] = True
                            logging.debug(f"Activating particle {p['idx']} at chunk {current_chunk}")
                    # OLD way 11/9 #

                    # NEW way 11/9 #
                    # Activate particles whose birth time matches this chunk
                    # for p in particles:
                    #    if p['chunk'] == current_chunk:
                    #        p['active'] = True
                    #        # NEW: mark that we still owe a birth snapshot for this particle
                    #        if not p.get('emitted_birth', False):
                    #            p['emitted_birth'] = False
                    #        logging.debug(f"Activating particle {p['idx']} at chunk {current_chunk}")
                    # NEW way 11/9 #

                    # OLD 11/9 #
                    #Create a list of only the currently active particles to move
                    active_particles = [p for p in particles if p.get('active', False)]
                    # OLD 11/9 #

                    # NEW 11/9 #
                    # NEW: Emit birth snapshot(s) BEFORE any motion occurs
                    # birth_output_rows = []
                    # for p in particles:
                    #    if p.get('active', False) and (p['chunk'] == current_chunk) and (p.get('emitted_birth', False) is False):
                    #        species_num = 1 if p['species'] == 'e' else 2
                    #        group_num   = 1 if p['group']   == 'initial' else 2
                    #        active_num  = 1
                    #        stopped_num = 0
                    #        # IMPORTANT: write exactly the same schema as your step rows
                    #        birth_output_rows.append([
                    #            current_chunk, p['idx'], species_num, p['X'], p['Y'],
                    #            p['lambda'], p['pressure'], group_num, active_num, stopped_num
                    #        ])
                    #        p['emitted_birth'] = True  # ensure one-time write per particle
                    #
                    # Write the birth rows in one append
                    #if birth_output_rows:
                    #    with open(temp_output_path, 'ab') as f_out:
                    #        libs.np.save(f_out, libs.np.array(birth_output_rows, dtype=libs.np.float64))
                    #
                    # Create a list of only the currently active particles to move
                    #active_particles = [p for p in particles if p.get('active', False)]
                    # NEW 11/9 #

                    # FIX 11/9 #
                    output_data = []
                    for p in active_particles:
                        species_num = 1 if p['species'] == 'e' else 2
                        group_num = 1 if p['group'] == 'initial' else 2
                        active_num = 1 if p.get('active', False) else 0
                        stopped_num = 1 if p.get('stopped', False) else 0

                        output_data.append([current_chunk, p['idx'], species_num, p['X'], p['Y'], p['lambda'], p['pressure'], group_num, active_num, stopped_num])

                    if output_data:
                        with open(temp_output_path, 'ab') as f_out:
                            libs.np.save(f_out, libs.np.array(output_data, dtype=libs.np.float64))
                    # END FIX 11/9 #

                    # 3. Move active particles by one step
                    updated_active_particles = move_particles.move_particles_one_step(
                        local_particles = active_particles,
                        current_chunk = current_chunk,
                        metadata = metadata,
                        interpolators = interpolators[current_chunk],
                        chunk_interpolators = interpolators[current_chunk],
                        is_inside_ellipse=is_inside_ellipse)

                    # Update the main particles list with the new states
                    # This is a more robust way to handle the updates
                    updated_map = {p['idx']: p for p in updated_active_particles}
                    for i, p in enumerate(particles):
                        if p['idx'] in updated_map:
                            particles[i] = updated_map[p['idx']]


                    # --- NEW BLOCK: Write particles that JUST stopped ---
                    # This runs AFTER the move to catch the final "stopped" state.
                    stopped_data = []
                    for p in updated_active_particles:
                        # Check if this particle was flagged as stopped *in this step*
                        # move_particles_one_step sets 'stop_chunk' to 'current_chunk' when it stops a particle.
                        if p.get('stopped', False) and p.get('stop_chunk') == current_chunk:

                            species_num = 1 if p['species'] == 'e' else 2
                            group_num = 1 if p['group'] == 'initial' else 2
                            active_num = 0  # It is no longer active
                            stopped_num = 1 # It is stopped

                            # Use the 'final_X' and 'final_Y' keys, which were set inside move_particles_one_step
                            stopped_data.append([
                                    current_chunk, p['idx'], species_num, p.get('final_X'), p.get('final_Y'),
                                    p['lambda'], p['pressure'], group_num, active_num, stopped_num
                            ])

                    # Write the collected "stopped" data for this timestep
                    if stopped_data:
                        with open(temp_output_path, 'ab') as f_out: # 'ab' = append binary
                            libs.np.save(f_out, libs.np.array(stopped_data, dtype=libs.np.float64))
                    # --- END NEW BLOCK ---

                    #output_data = []
                    # 4. Write current state of local particles to temp file
                    #for p in updated_active_particles:
                        # Convert categorical data to numbers for efficient storage
                    #    species_num = 1 if p['species'] == 'e' else 2
                    #    group_num = 1 if p['group'] == 'initial' else 2
                    #    active_num = 1 if p.get('active', False) else 0
                    #    stopped_num = 1 if p.get('stopped', False) else 0
                        
                    #    output_data.append([
                    #            current_chunk, p['idx'], species_num, p['X'], p['Y'], 
                    #             p['lambda'], p['pressure'], group_num, active_num, stopped_num
                    #    ])

                    # Write the collected data for this timestep as a single NumPy array
                    #if output_data:
                    #    with open(temp_output_path, 'ab') as f_out: # 'ab' = append binary
                    #        libs.np.save(f_out, libs.np.array(output_data, dtype=libs.np.float64))

                    # --- REPORT INSIDE THE LOOP ---
                    # In Simple_test.py, inside the for loop (around line 257)

                    # --- REPORT INSIDE THE LOOP ---
                    if rank == 0 and current_chunk % 1 == 0:
                        elapsed_time = libs.time.time() - run_start_time
                        frames_done = start_chunk - current_chunk
                        
                        # The 'total_chunks' variable is now used here, not redefined.
                        if total_chunks > 0:
                            progress = frames_done / total_chunks
                            progress_percent = progress * 100
                            eta_seconds = (elapsed_time / progress) * (1 - progress) if progress > 0 else 0
                            eta_str = f"{int(eta_seconds // 3600):02d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"
                        else:
                            progress_percent = 100.0
                            eta_str = "00:00:00"

                        status = {
                            "stage": "Running Simulation",
                            "current_chunk": current_chunk,
                            "start_chunk": start_chunk,
                            "stop_chunk": stop_chunk,
                            "progress_percent": f"{progress_percent:.2f}%",
                            "elapsed_time_str": f"{int(elapsed_time // 3600):02d}:{int((elapsed_time % 3600) // 60):02d}:{int(elapsed_time % 60):02d}",
                            "eta_str": eta_str
                        }
                        with open("run_status.json", 'w') as f:
                            libs.json.dump(status, f, indent=4)
                    # --- END ---

            # GATHER all initial data to rank 0 to get correct birth chunks for HDF5 attributes
            all_inits_gathered = comm.gather(all_inits, root=0)
            
            if len(output_data) > 0:
                with open(temp_output_path, 'ab') as f_out:
                    libs.np.save(f_out, libs.np.array(output_data, dtype=libs.np.float64))
                output_data.clear()

            # --- Assemble final HDF5 on Rank 0 from temporary files ---
            # AFTER: This is the new, memory-efficient code you will paste in its place.

            # --- NEW: Save initial particle data for birth chunk lookup ---
            # Each rank saves its list of initial tuples to a temporary JSON file.
            # This gives the assembly script the necessary global context.
            init_data_path = f"{params.output_hdf5_path}.rank{rank}.init.json"
            try:
                with open(init_data_path, 'w') as f_init:
                    # The all_inits list contains tuples like (time, idx, species, I, J, lambda, pressure, group)
                    libs.json.dump(all_inits, f_init)
            except Exception as e:
                logging.error(f"Failed to write initial data to {init_data_path}: {e}")
                
            comm.Barrier() # Ensure all ranks have finished writing their temp files

            # --- Assemble final HDF5 on Rank 0 from temporary files (Streaming Method) ---
            if rank == 0:
                # Final status update to signal completion for the monitor
                with open("run_status.json", 'r') as f:
                    final_status = libs.json.load(f)
                final_status["stage"] = "Complete"
                final_status["progress_percent"] = "100.00%"
                with open("run_status.json", 'w') as f:
                    libs.json.dump(final_status, f, indent=4)
        
        except Exception as e:
            # Fail fast to avoid hanging peers at a barrier
            logging.exception(f"Rank {rank}: fatal error in main(): {e}")
            try:
                comm.Abort(1)
            finally:
                raise
        finally:
            dt = libs.time.time() - t0
            if rank == 0:
                logging.info(f"Total wall time: {dt/60.0:.2f} min")

if __name__ == '__main__':
    main()
