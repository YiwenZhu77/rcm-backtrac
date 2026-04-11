# data_init.py

import libs
import params
import tools

# Initialize MPI
comm = libs.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --- Minute-cadence helpers (explicitly encode 1 update = 1 minute) ---
# Prefer params.seconds_per_chunk when present; default to 60s/min.
SECONDS_PER_CHUNK = getattr(params, "seconds_per_chunk", 60.0)
_MINUTE_ALIGN_TOL = 1e-6  # seconds

def chunk_to_seconds(chunk: int) -> float:
    """Map chunk index -> elapsed seconds (0-based chunking)."""
    return float(chunk) * SECONDS_PER_CHUNK

def seconds_to_chunk(t: float, rounding: str = "floor") -> int:
    """
    Map seconds -> chunk index using specified rounding:
      - "floor" (default): previous minute
      - "round": nearest minute (robust to tiny FP noise)
      - "ceil": next minute
    """
    x = t / SECONDS_PER_CHUNK
    if rounding == "ceil":
        return int(libs.np.ceil(x))
    if rounding == "floor":
        return int(libs.np.floor(x))
    return int(libs.np.round(x))

def is_minute_aligned(t: float, tol: float = _MINUTE_ALIGN_TOL) -> bool:
    """True if t is aligned to minute grid within ±tol."""
    m = libs.np.round(t / SECONDS_PER_CHUNK)
    return abs(t - m * SECONDS_PER_CHUNK) <= tol

def assert_minute_aligned(t: float, where: str = "time"):
    """Raise if t is not on minute boundaries (helps catch drift early)."""
    if not is_minute_aligned(t):
        raise ValueError(f"{where}: {t} not aligned to minute (±{_MINUTE_ALIGN_TOL}s)")

def iter_minute_chunks(start_chunk: int, stop_chunk: int, descending: bool = True):
    """
    Iterate chunk indices one-by-one (== one minute per step), inclusive of endpoints.
    descending=True yields start_chunk, start_chunk-1, ..., stop_chunk.
    """
    if descending:
        rng = range(start_chunk, stop_chunk - 1, -1)
    else:
        rng = range(start_chunk, stop_chunk + 1, 1)
    for c in rng:
        yield c

def load_metadata(f):
    """
    Load metadata from HDF5 file on rank 0 and broadcast to all MPI ranks.

    Parameters:
    - f: h5py.File object (opened in parallel mode).

    Returns:
    - metadata: Dictionary with grid metadata (total_size, chunk_size, num_chunks, i_max, j_max).
    """
    start_time = libs.time.time()
    process = libs.psutil.Process()
    mem_before = process.memory_info().rss / 1024**2

    metadata = None
    if rank == 0:
        try:
            if not libs.os.path.exists(f.filename) or not libs.os.access(f.filename, libs.os.R_OK):
                raise IOError(f"HDF5 file {f.filename} is not accessible")

            required_datasets = ['I', 'J', 'XMIN']
            for dataset in required_datasets:
                if dataset not in f:
                    raise KeyError(f"Dataset '{dataset}' not found in HDF5 file {f.filename}.")

            i_max = int(libs.np.max(f['I']))
            j_max = int(libs.np.max(f['J']))
            chunk_size = i_max * j_max
            total_size = len(f['XMIN'])
            num_chunks = total_size // chunk_size

            if total_size % chunk_size != 0:
                raise ValueError(f"Total size ({total_size}) is not evenly divisible by chunk size ({chunk_size}).")

            metadata = {
                "total_size": total_size,
                "chunk_size": chunk_size,
                "num_chunks": num_chunks,
                "i_max": i_max,
                "j_max": j_max,
                "seconds_per_chunk" : SECONDS_PER_CHUNK,
            }
            libs.logging.info(f"Loaded metadata: {metadata}")
        except Exception as e:
            libs.logging.error(f"Failed to load metadata: {str(e)}")
            raise

    metadata = comm.bcast(metadata, root=0)
    libs.logging.info(f"Rank {rank}: Received metadata: {metadata}")

    if metadata is None:
        raise ValueError("Metadata broadcast failed")

    elapsed_time = libs.time.time() - start_time
    mem_after = process.memory_info().rss / 1024**2
    libs.logging.debug(f"Rank {rank}: load_metadata took {elapsed_time:.2f}s, memory used: {mem_after - mem_before:.2f} MB")

    return metadata

def load_chunk(f, chunk_index, metadata, buffer_distance=params.buffer_distance, verbose=True):
    """
    Load field data for a specific chunk from HDF5 file, applying elliptical mask.

    Parameters:
    - f: h5py.File object (opened in parallel mode).
    - chunk_index: Integer index of the chunk to load.
    - metadata: Dictionary with grid metadata.
    - buffer_distance: Distance for buffer zone around ellipse boundary.
    - verbose: Enable detailed logging (default: True).

    Returns:
    - chunk_data: Dictionary of field arrays reshaped to (j_max, i_max).
    """
    start_time = libs.time.time()
    process = libs.psutil.Process()
    mem_before = process.memory_info().rss / 1024**2

    if chunk_index < 0 or chunk_index >= metadata["num_chunks"]:
        libs.logging.error(f"Rank {rank}: Invalid chunk index {chunk_index} (valid range: 0 to {metadata['num_chunks']-1})")
        raise ValueError(f"Invalid chunk index {chunk_index} (valid range: 0 to {metadata['num_chunks']-1})")

    chunk_data = {}
    try:
        start_idx = chunk_index * metadata["chunk_size"]
        end_idx = (chunk_index + 1) * metadata["chunk_size"]
        for key in f.keys():
            dset = f[key]
            chunk_data[key] = dset[start_idx:end_idx]
            chunk_data[key] = chunk_data[key].reshape(metadata["j_max"], metadata["i_max"])

        required_datasets = ['XMIN', 'YMIN']
        for dataset in required_datasets:
            if dataset not in chunk_data:
                raise KeyError(f"Dataset '{dataset}' not found in chunk {chunk_index}")

        if verbose:
            sample_size = min(5, chunk_data['XMIN'].shape[1])
            libs.logging.debug(f"Rank {rank}: Chunk {chunk_index} XMIN sample (before mask): {chunk_data['XMIN'][0, :sample_size]}")
            libs.logging.debug(f"Rank {rank}: Chunk {chunk_index} YMIN sample (before mask): {chunk_data['YMIN'][0, :sample_size]}")

        # Define X and Y for all ranks
        X = chunk_data['XMIN']
        Y = chunk_data['YMIN']

        # Compute elliptical mask on Rank 0 and broadcast
        if rank == 0:
            _, _, _, _, is_inside_ellipse = tools.plot_ellipse_boundary(
                a1=params.a1, a2=params.a2, semiminor=params.semiminor, plot=False, return_mask=True
            )
            mask = is_inside_ellipse(X, Y)
            libs.logging.debug(f"Rank {rank}: Chunk {chunk_index} mask shape: {mask.shape}, non-zero count: {libs.np.sum(mask)}")
        else:
            mask = None
        mask = comm.bcast(mask, root=0)
        libs.logging.debug(f"Rank {rank}: Received mask for chunk {chunk_index}, non-zero count: {libs.np.sum(mask)}")

        # BEFORE per-key loop:
        inside_points = libs.np.column_stack((X[mask], Y[mask]))
        outside_points = libs.np.column_stack((X[~mask], Y[~mask]))

        if inside_points.size == 0:
            # All points outside ellipse (degenerate grid vs ellipse) — just zero fields
            for key in chunk_data:
                if key not in ["I", "J", "XMIN", "YMIN"]:
                    chunk_data[key][...] = 0.0
        else:
            tree = libs.cKDTree(inside_points)
            distances, indices = tree.query(outside_points, k=1)
            buffer_mask = distances <= buffer_distance
            outer_rows, outer_cols = libs.np.where(~mask)
            # Row/col indices for buffer-zone subset in grid space
            buf_rows = outer_rows[buffer_mask]
            buf_cols = outer_cols[buffer_mask]

            for key, values in chunk_data.items():
                if key in ["I", "J", "XMIN", "YMIN"]:
                    continue

                grid_values = values.copy()
                grid_values[~mask] = 0.0
                # Restore buffer zone from nearest inside values
                nearest_inside_values = values[mask][indices[buffer_mask]]
                grid_values[buf_rows, buf_cols] = nearest_inside_values

                chunk_data[key] = grid_values

        if verbose:
            sample_size = min(5, chunk_data['XMIN'].shape[1])
            libs.logging.debug(f"Rank {rank}: Chunk {chunk_index} XMIN sample (after mask): {chunk_data['XMIN'][0, :sample_size]}")
            libs.logging.debug(f"Rank {rank}: Chunk {chunk_index} YMIN sample (after mask): {chunk_data['YMIN'][0, :sample_size]}")

        libs.logging.info(f"Rank {rank}: Loaded chunk {chunk_index}")

    except Exception as e:
        libs.logging.error(f"Rank {rank}: Failed to load chunk {chunk_index}: {str(e)}")
        raise

    elapsed_time = libs.time.time() - start_time
    mem_after = process.memory_info().rss / 1024**2
    libs.logging.debug(f"Rank {rank}: load_chunk {chunk_index} took {elapsed_time:.2f}s, memory used: {mem_after - mem_before:.2f} MB")

    return chunk_data

# data_init.py
import libs
import params
from scipy.interpolate import RegularGridInterpolator

def precompute_interpolation_grids(h5, chunk_data_dict, metadata):
    """
    Build per-chunk interpolators for all fields we have, on (J, I) ordering.

    Returns
    -------
    interpolators : dict[int, dict[str, RegularGridInterpolator]]
        Map: chunk -> { field_name -> interpolator }
    """
    if not chunk_data_dict:
        raise ValueError("chunk_data_dict is empty.")

    i_max, j_max = int(metadata["i_max"]), int(metadata["j_max"])
    I_vals = libs.np.arange(1.0, i_max + 1.0, dtype=float)
    J_vals = libs.np.arange(1.0, j_max + 1.0, dtype=float)

    interpolators: dict[int, dict[str, RegularGridInterpolator]] = {}

    for chunk_idx, chunk_data in chunk_data_dict.items():
        chunk_interps: dict[str, RegularGridInterpolator] = {}

        # Helper to create an interpolator with consistent options
        def _mk_interp(name, arr, *, fill_value):
            return RegularGridInterpolator(
                (J_vals, I_vals), arr, method='linear', bounds_error=False, fill_value=fill_value
            )

        # Always build geometry / angle bases
        if "XMIN" in chunk_data:
            chunk_interps["XMIN"] = _mk_interp("XMIN", chunk_data["XMIN"], fill_value=libs.np.nan)
        if "YMIN" in chunk_data:
            chunk_interps["YMIN"] = _mk_interp("YMIN", chunk_data["YMIN"], fill_value=libs.np.nan)
        if "COLAT" in chunk_data:
            # angle in degrees; allow extrap to nearest
            chunk_interps["COLAT"] = _mk_interp("COLAT", chunk_data["COLAT"], fill_value=None)
        if "ALOCT" in chunk_data:
            # use sin/cos to keep phase continuous (like your one-part script)
            aloct = chunk_data["ALOCT"]
            chunk_interps["ALOCT_sin"] = _mk_interp("ALOCT_sin", libs.np.sin(aloct), fill_value=0.0)
            chunk_interps["ALOCT_cos"] = _mk_interp("ALOCT_cos", libs.np.cos(aloct), fill_value=0.0)
            # (Optional straight ALOCT too, for consumers that expect it)
            chunk_interps["ALOCT"]     = _mk_interp("ALOCT", aloct, fill_value=None)

        # Scalars used in dynamics / plotting. If present, make them available.
        for fld, fv in [
            ("V",  0.0),
            ("VM", 0.0),
            ("PVG", 0.0),
            # add any additional scalar fields you have here
        ]:
            if fld in chunk_data:
                arr = chunk_data[fld]
                # sanitize NaNs -> 0 so downstream math/lognorm is predictable
                if libs.np.isnan(arr).any():
                    arr = libs.np.nan_to_num(arr, nan=0.0)
                chunk_interps[fld] = _mk_interp(fld, arr, fill_value=fv)

        interpolators[chunk_idx] = chunk_interps

    return interpolators
