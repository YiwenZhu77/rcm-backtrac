import os
import logging
import time
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import json
import numpy as np
import h5py
import params 
import data_init

# --- Configuration ---
OUT_H5 = params.output_hdf5_path
TEMP_PREFIX = f"{OUT_H5}.rank"
SECONDS_PER_CHUNK = abs(params.time_step)

# --- Argument Parsing ---
# This allows the PBS script to pass the number of ranks
parser = argparse.ArgumentParser(description="Assemble HDF5 file from temporary .npy files.")
parser.add_argument('n_ranks', type=int, help='Total number of MPI ranks that produced .npy files.')
args = parser.parse_args()
N_RANKS = args.n_ranks

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("assemble.log", mode='w'),
        logging.StreamHandler()
    ]
)

def build_birth_chunk_lookup(n_ranks, temp_prefix):
    """
    Reads all .init.json files to build a global mapping from particle ID
    to its true birth chunk.
    """
    pid_to_birth_chunk = {}
    all_initial_tuples = []

    for i in range(n_ranks):
        json_path = f"{temp_prefix}{i}.init.json"
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    all_initial_tuples.extend(json.load(f))
            except Exception as e:
                logging.warning(f"Could not read or parse {json_path}: {e}")

    for p_tuple in all_initial_tuples:
        t_birth, pid = p_tuple[0], int(p_tuple[1])
        birth_chunk = data_init.seconds_to_chunk(t_birth, rounding="round")
        pid_to_birth_chunk[pid] = birth_chunk

    logging.info(f"Built birth chunk lookup for {len(pid_to_birth_chunk)} unique particles.")
    return pid_to_birth_chunk

# --- HDF5 Helper Functions ---

# --- OCT16 version (1) ---
#def ensure_ds(group, name, dtype):
#    if name in group:
#        return group[name]
#    return group.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=True, compression="gzip")
# --- OCT16 version (1) ---

# --- OCT16 NEW version (1) ---
def ensure_ds(group, name, dtype):
    if name in group:
        return group[name]
    # Let h5py choose chunk dims automatically; no compression while assembling.
    return group.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=True)
# --- OCT16 NEW version (1) ---

def load_rank_file(rank):
    """
    Worker: Reads the entire .npy file for a rank.
    Since the simulation appends multiple np.save() calls, we must loop
    until EOF to capture all time steps.
    """
    fpath = f"{TEMP_PREFIX}{rank}.tmp.npy"
    if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
        return None

    data_list = []
    try:
        with open(fpath, "rb") as f:
            while True:
                try:
                    # Attempt to read the next array in the stream
                    chunk = np.load(f, allow_pickle=False)

                    # Ensure 2D
                    if chunk.ndim == 1:
                        chunk = chunk.reshape(1, -1)

                    data_list.append(chunk)
                except (EOFError, ValueError):
                    # EOF reached (normal) or empty read
                    break

        if not data_list:
            return None

        # Combine all time steps (T=360, 359, 358...) into one array
        full_data = np.vstack(data_list)
        return full_data.astype(np.float64, copy=False)

    except Exception as e:
        logging.warning(f"Error processing {fpath}: {e}")
        return None
    
def flush_buffer_to_hdf5(hdf5_group, buffer_list, stop_info, seconds_per_chunk):
    """
    Concatenates, sorts by PID, and writes data to HDF5.
    """
    if not buffer_list:
        return

    # 1. Stack all arrays (Fast in-memory operation)
    full_batch = np.vstack(buffer_list)
    
    # 2. Sort by PID (Column 1)
    # This aligns writes to be sequential on disk
    sort_indices = np.argsort(full_batch[:, 1])
    sorted_batch = full_batch[sort_indices]

    # 3. Find unique PIDs and their boundaries
    pids = sorted_batch[:, 1].astype(np.int64)
    unique_pids, start_indices = np.unique(pids, return_index=True)
    
    # 4. Iterate over unique PIDs and write block
    # skipping the first index 0, we zip starts to get slices
    end_indices = np.roll(start_indices, -1)
    end_indices[-1] = sorted_batch.shape[0]

    for pid, start, end in zip(unique_pids, start_indices, end_indices):
        chunk_data = sorted_batch[start:end]

        # Extract columns
        chunks = chunk_data[:, 0]
        # pids = chunk_data[:, 1] (we already have this)
        s_nums = chunk_data[:, 2]
        Xs     = chunk_data[:, 3]
        Ys     = chunk_data[:, 4]
        lams   = chunk_data[:, 5]
        press  = chunk_data[:, 6]
        g_nums = chunk_data[:, 7]
        # active = chunk_data[:, 8]
        stopped= chunk_data[:, 9]

        # Create/Get Group
        group_name = f"particle_{pid}"
        if group_name in hdf5_group:
            gid = hdf5_group[group_name]
        else:
            gid = hdf5_group.create_group(group_name)
            # Set static attributes only on creation
            species_str = 'e' if s_nums[0] == 1 else 'p'
            group_str   = 'initial' if g_nums[0] == 1 else 'lost'
            gid.attrs["idx"]      = pid
            gid.attrs["species"]  = species_str
            gid.attrs["lambda"]   = float(lams[0])
            gid.attrs["pressure"] = float(press[0])
            gid.attrs["group"]    = group_str

        # Handle Stopped Status
        if pid not in stop_info and np.any(stopped == 1):
            # Find the index where it stopped
            stop_idx = np.argmax(stopped == 1)
            stop_info[pid] = (int(chunks[stop_idx]), float(Xs[stop_idx]), float(Ys[stop_idx]))

        # WRITE DATA
        # Convert chunks to time
        times = chunks * seconds_per_chunk

        # Helper to extend dataset
        def extend_and_write(dset_name, data_arr):
            if dset_name in gid:
                dset = gid[dset_name]
                old_len = dset.shape[0]
                dset.resize((old_len + len(data_arr),))
                dset[old_len:] = data_arr
            else:
                # Create with chunking, but sized exactly to start
                gid.create_dataset(dset_name, data=data_arr, maxshape=(None,), chunks=True)

        extend_and_write("time", times)
        extend_and_write("X", Xs)
        extend_and_write("Y", Ys)

# --- Main Assembly Logic ---
def main():
    start_time = time.time()
    logging.info(f"Starting HDF5 assembly from {N_RANKS} rank files.")
    logging.info(f"Input prefix: {TEMP_PREFIX}")
    logging.info(f"Output file: {OUT_H5}")

    # --- Consolidated Manager and Queue setup ---
    status_file = "assemble_status.json"

    # Load PID lookup first
    pid_to_birth_chunk = build_birth_chunk_lookup(N_RANKS, TEMP_PREFIX)

    with h5py.File(OUT_H5, "w", libver='latest') as h5:
        meta = h5.create_group("meta")
        meta.attrs["assembled_from_ranks"] = N_RANKS
        meta.attrs["seconds_per_chunk"] = SECONDS_PER_CHUNK
        part_root = h5.create_group("particles")
        stop_info = {}

        # MEMORY BUFFER SETTINGS
        MAX_BUFFER_ROWS = 5_000_000
        row_buffer = []
        current_rows = 0

        # Simple local counter
        files_processed_count = 0

        num_workers = min(N_RANKS, cpu_count())
        logging.info(f"Scanning {N_RANKS} files with {num_workers} workers...")

        with Pool(processes=num_workers) as pool:
            # imap_unordered yields results one by one as workers finish
            for data_array in pool.imap_unordered(load_rank_file, range(N_RANKS)):

                # 1. Update Progress
                files_processed_count += 1

                # Update status file every 100 files (or whatever frequency you prefer)
                if files_processed_count % 50 == 0 or files_processed_count == N_RANKS:
                    progress_percent = (files_processed_count / N_RANKS) * 100
                    elapsed = time.time() - start_time
                    status = {
                        "stage": "Assembling HDF5",
                        "files_processed": files_processed_count,
                        "total_files": N_RANKS,
                        "progress_percent": progress_percent,
                        "elapsed_time_str": f"{int(elapsed // 3600):02d}:{int((elapsed % 3600) // 60):02d}:{int(elapsed % 60):02d}"
                    }
                    try:
                        with open(status_file, 'w') as f:
                            json.dump(status, f, indent=4)
                    except Exception:
                        pass # Don't crash assembly if status file is locked

                # 2. Accumulate Data
                if data_array is None:
                    continue

                row_buffer.append(data_array)
                current_rows += data_array.shape[0]

                # 3. Flush Buffer if full
                if current_rows >= MAX_BUFFER_ROWS:
                    logging.info(f"Buffer full ({current_rows} rows). Flushing to HDF5...")
                    flush_buffer_to_hdf5(part_root, row_buffer, stop_info, SECONDS_PER_CHUNK)
                    row_buffer = []
                    current_rows = 0

            # 4. Flush remaining data after loop ends
            if row_buffer:
                logging.info(f"Flushing final {current_rows} rows...")
                flush_buffer_to_hdf5(part_root, row_buffer, stop_info, SECONDS_PER_CHUNK)

        logging.info("Setting final birth chunk and stop info attributes...")

        for name in part_root.keys():
            gid = part_root[name]
            pid = int(gid.attrs["idx"])

            if pid in pid_to_birth_chunk:
                gid.attrs["chunk"] = int(pid_to_birth_chunk[pid])
            else:
                gid.attrs["chunk"] = int(params.last_chunk)
                logging.warning(f"PID {pid} not found in birth chunk lookup, falling back.")

            if pid in stop_info:
                stop_chunk, final_X, final_Y = stop_info[pid]
                gid.attrs["stop_chunk"] = int(stop_chunk)
                gid.attrs["final_X"] = float(final_X)
                gid.attrs["final_Y"] = float(final_Y)

    logging.info("Cleaning up temporary .init.json files...")
    for i in range(N_RANKS):
        json_path = f"{TEMP_PREFIX}{i}.init.json"
        if os.path.exists(json_path):
            # os.remove(json_path)
            pass

    # --- Add a final 100% progress update before completion ---
    final_elapsed = time.time() - start_time
    final_progress_status = {
        "stage": "Assembling HDF5",
        "files_processed": N_RANKS,
        "total_files": N_RANKS,
        "progress_percent": 100.0,
        "elapsed_time_str": f"{int(final_elapsed // 3600):02d}:{int((final_elapsed % 3600) // 60):02d}:{int(final_elapsed % 60):02d}"
    }
    with open(status_file, 'w') as f:
        json.dump(final_progress_status, f, indent=4)
    
    # Give the monitor a moment to catch the 100% update
    time.sleep(2) 

    # Now, write the final "Complete" status
    final_progress_status["stage"] = "Complete"
    with open(status_file, 'w') as f:
        json.dump(final_progress_status, f, indent=4)

    elapsed = time.time() - start_time
    logging.info(f"Assembly complete in {elapsed:.2f} seconds.")
    logging.info(f"Final HDF5 file created: {OUT_H5}")

if __name__ == "__main__":
    main()
