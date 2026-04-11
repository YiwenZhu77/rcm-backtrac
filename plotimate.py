# plot & animate

import params
import glob
import libs
import tools
import data_init
import os
import math
import shutil
import subprocess
import matplotlib.image as mpimg  # <— for reading the static colorbar image
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import LogFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter, MaxNLocator, LogFormatterMathtext, LogLocator, FuncFormatter


# Static colorbar settings (tweak as you like)
USE_STATIC_CBAR = False           # set False to disable entirely
CBAR_POS = [0.86, 0.18, 0.03, 0.64]  # [left, bottom, width, height] in figure coords

comm = libs.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def _ellipse_mask(X, Y, a1, a2, semiminor, y0=0.0):
    # inside == True means “draw here”
    a = 0.5 * (a1 - a2)  # semimajor
    x0 = 0.5 * (a1 + a2)
    b = semiminor
    return ((X - x0) ** 2) / (a ** 2) + ((Y - y0) ** 2) / (b ** 2) <= 1.0

def _triangulate_structured(X, Y, vertex_mask=None):
    """
    X, Y: shape (Ny, Nx) of **cell centers** (curvilinear ok).
    vertex_mask: optional bool array of shape (Ny*Nx,), True->mask vertex.
                 We convert to a triangle mask (mask triangle if any vertex masked).
    """
    Ny, Nx = X.shape
    idx = np.arange(Ny * Nx).reshape(Ny, Nx)

    # two triangles per cell
    tri1 = np.stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[:-1, 1:].ravel()], axis=1)
    tri2 = np.stack([idx[1:, 1:].ravel(),  idx[:-1, 1:].ravel(), idx[1:, :-1].ravel()], axis=1)
    triangles = np.vstack([tri1, tri2])

    tri = mtri.Triangulation(X.ravel(), Y.ravel(), triangles)

    if vertex_mask is not None:
        # mask triangles that touch any masked vertex
        tri_mask = vertex_mask[triangles].any(axis=1)
        tri.set_mask(tri_mask)

    return tri

# ---------- helpers to read simulation_data.h5 ----------

def _species_to_char(val):
    """
    Robustly map HDF5-stored species to 'e' or 'p'.
    Accepts bytes/str ('e'/'p') or ints (1=e, 2=p). Also tolerates older 0/1 encodings.
    """
    # string/bytes cases
    if isinstance(val, (bytes, str)):
        s = val.decode() if isinstance(val, bytes) else val
        return 'e' if s.strip().lower().startswith('e') else 'p'
    # numeric cases
    try:
        iv = int(val)
        if iv in (1, 0):   # 1=e (current writer), 0=e (older encodings, if any)
            return 'e'
        if iv == 2:        # 2=p (current writer)
            return 'p'
    except Exception:
        pass
    # fallback (default to electron to make the bug obvious rather than silent)
    return 'e'

def load_simulation_data(h5_path):
    """
    Returns:
      particles   : list of dicts (idx, species, lambda, pressure, time, chunk, stop_chunk, final_X, final_Y, group)
      trajectories: list[list[{'chunk', 'X', 'Y'}]]
      first_chunk, last_chunk: ints
    """
    particles, trajectories = [], []
    first_chunk, last_chunk = None, None

    with libs.h5py.File(h5_path, "r") as f:
        root = f["particles"] if "particles" in f else f
        keys = sorted((k for k in root.keys() if k.startswith("particle_")),
                      key=lambda s: int(s.split("_")[1]))
        for k in keys:
            g = root[k]
            X = libs.np.asarray(g["X"][:], dtype=float)
            Y = libs.np.asarray(g["Y"][:], dtype=float)
            T = libs.np.asarray(g["time"][:], dtype=float)

            mask = libs.np.isfinite(X) & libs.np.isfinite(Y) & libs.np.isfinite(T)
            X, Y, T = X[mask], Y[mask], T[mask]

            steps = []
            for xx, yy, tt in zip(X, Y, T):
                ch = data_init.seconds_to_chunk(float(tt), rounding="round")
                steps.append({"chunk": int(ch), "X": float(xx), "Y": float(yy)})
                first_chunk = ch if first_chunk is None else min(first_chunk, ch)
                last_chunk  = ch if last_chunk  is None else max(last_chunk , ch)

            atr      = g.attrs
            species  = _species_to_char(atr.get("species", 0))
            lam      = float(atr.get("lambda", 0.0))
            press    = float(atr.get("pressure", 0.0))
            idx      = int(atr.get("idx", len(particles)))
            stop_ch  = atr.get("stop_chunk", None)
            stop_ch  = int(stop_ch) if stop_ch is not None else None
            final_X  = float(atr.get("final_X", libs.np.nan))
            final_Y  = float(atr.get("final_Y", libs.np.nan))
            grp      = atr.get("group", "initial")
            grp      = grp.decode() if isinstance(grp, bytes) else str(grp)
            init_t   = float(T[0]) if len(T) else 0.0

            # Prioritize the explicitly saved birth chunk attribute
            if "chunk" in atr:
                init_ch = int(atr["chunk"])
            else: # Fallback for old files
                init_ch = data_init.seconds_to_chunk(init_t, rounding="round") if len(T) else (first_chunk or 0)

            particles.append({
                "idx": idx,
                "species": species,
                "lambda": lam,
                "pressure": press,
                "time": init_t,
                "chunk": int(init_ch),
                "stop_chunk": stop_ch,
                "final_X": final_X,
                "final_Y": final_Y,
                "group": grp if grp in ("initial", "lost") else "initial",
            })
            trajectories.append(steps)

    if first_chunk is None:
        first_chunk, last_chunk = 0, 0
    return particles, trajectories, int(first_chunk), int(last_chunk)


# ----------------- PVG field plotting -----------------

# plotimate.py
import libs
import params
import tools
import matplotlib.pyplot as plt

def plot_pvg_field(ax, chunk_data, chunk_interpolators, metadata, chunk_idx,
                   levels=None, resolution_factor=params.resolution_factor):
    """
    Draw PVG (log-scaled) for `chunk_idx` using the same approach as the one-part script.
    Returns the QuadContourSet (or None if PVG missing).
    """

    if 'PVG' not in chunk_interpolators:
        # keep previous PVG visible rather than wiping the frame
        return None, None, None

    interp_PVG = chunk_interpolators['PVG']
    interp_X   = chunk_interpolators['XMIN']
    interp_Y   = chunk_interpolators['YMIN']

    i_max = int(metadata["i_max"])
    j_max = int(metadata["j_max"])

    # resolution control
    i_step = max(1, int(1.0 / float(resolution_factor)))
    j_step = max(1, int(1.0 / float(resolution_factor)))

    I_vals = libs.np.arange(1.0, i_max + 1.0, i_step, dtype=float)
    J_vals = libs.np.arange(1.0, j_max + 1.0, j_step, dtype=float)
    if I_vals.size < 2:  # safety
        I_vals = libs.np.linspace(1.0, i_max, 2)
    if J_vals.size < 2:
        J_vals = libs.np.linspace(1.0, j_max, 2)

    I_grid, J_grid = libs.np.meshgrid(I_vals, J_vals)  # shape (len(J_vals), len(I_vals))
    pts = libs.np.column_stack((J_grid.ravel(), I_grid.ravel()))

    # Interpolate PVG + coords to the (I,J) sample grid
    PVG = interp_PVG(pts).reshape(J_grid.shape)
    Xv  = interp_X(pts).reshape(J_grid.shape)
    Yv  = interp_Y(pts).reshape(J_grid.shape)

    # Apply mask and log-friendly clamping (this is the bit that prevents vanish)
    PVG = PVG.copy()
    PVG[PVG <= 0] = libs.np.nan

    # Check params.py for hardcoded values first.
    pvg_vmin_hc = getattr(params, "pvg_vmin_hardcoded", None)
    pvg_vmax_hc = getattr(params, "pvg_vmax_hardcoded", None)

    if pvg_vmin_hc is not None and pvg_vmax_hc is not None:
        # --- CASE 1: Use hardcoded values from params.py ---
        vmin = pvg_vmin_hc
        vmax = pvg_vmax_hc
        # Clip data to the hardcoded range for accurate color mapping
        libs.np.clip(PVG, vmin, vmax, out=PVG)
    else:
        # --- CASE 2: Fallback to automatic calculation ---
        if not libs.np.any(libs.np.isfinite(PVG)):
            return None, None, None  # No valid data to plot

        vmin = float(libs.np.nanmin(PVG))
        vmax = float(libs.np.nanmax(PVG))

        if not np.isfinite(vmin) or not np.isfinite(vmax):
             return None, None, None

    n_levels = getattr(params, "pvg_levels", 21)
    levels = libs.np.logspace(np.log10(vmin), np.log10(vmax), n_levels)

    # IMPORTANT: use contourf (not pcolormesh) to avoid the "non-monotonic centers" edge warnings
    cs = ax.contourf(Xv, Yv, PVG, levels=levels, norm=LogNorm(vmin=vmin, vmax=vmax), 
                     cmap=getattr(params, "pvg_cmap", "binary"), alpha=1.0, zorder=1)

    return cs, vmin, vmax

# ----------------- build field data / interpolators -----------------

def build_interpolators_for_chunks(chunks_needed):
    """
    Open params.hdf5_file_path and construct:
      - metadata: dict with i_max, j_max, etc.
      - chunk_data_dict: {chunk -> raw arrays for that chunk}
      - chunk_interpolators: {chunk -> {field -> RegularGridInterpolator}}
    Only chunks in 'chunks_needed' are loaded to save memory/time.
    """
    # Normalize and sort the requested chunks
    if isinstance(chunks_needed, (set, list, tuple)):
        chunks_list = sorted(int(c) for c in chunks_needed)
    else:
        chunks_list = [int(chunks_needed)]

    with libs.h5py.File(params.hdf5_file_path, 'r') as f_data:
        # 1) metadata from the file
        metadata = data_init.load_metadata(f_data)

        # 2) load the raw arrays for the requested chunks
        chunk_data_dict = {}
        for ch in chunks_list:
            try:
                chunk_data_dict[ch] = data_init.load_chunk(
                    f_data, ch, metadata,
                    buffer_distance=getattr(params, "buffer_distance", 1.0),
                    verbose=False
                )
            except Exception as e:
                libs.logging.warning(f"[rank {rank}] Failed to load chunk {ch}: {e}")

        # 3) build per-chunk interpolators (V, VM, PVG, XMIN, YMIN, COLAT, ALOCT_sin/cos, etc.)
        chunk_interpolators = data_init.precompute_interpolation_grids(
            f_data, chunk_data_dict, metadata
        )

    return metadata, chunk_data_dict, chunk_interpolators

# ----------------- MPI frame renderer -----------------
def render_animation_from_h5_mpi(sim_h5_path=None, out_mp4=None, fps=4,
                                 metadata=None, chunk_data_dict=None, chunk_interpolators=None):
    """
    MPI frame renderer:
      - 1 PNG per minute-chunk (first_chunk..last_chunk)
      - PVG is redrawn at every frame with log-scaled colorbar
      - particles plotted at that minute
      - rank 0 assembles MP4 at the end
    """
    # --- Monitoring BLOCK ---
    status_file_path = "render_status.json"
    if rank == 0:
        render_start_time = libs.time.time()
    # --- END BLOCK ---

    if sim_h5_path is None:
        sim_h5_path = params.simulation_output_file_path
    if out_mp4 is None:
        out_mp4 = params.animation_file_path

    data_to_bcast = None
    if rank == 0:
        libs.logging.info(f"[rank {rank}] Reading simulation data from: {sim_h5_path}")
        # Only Rank 0 reads the file
        # Load trajectories/particles from the simulation HDF5
        p, t, fc, lc = load_simulation_data(sim_h5_path)
        data_to_bcast = {
                'particles': p,
                'trajectories': t,
                'first_chunk': fc,
                'last_chunk': lc
        }
    
    # Broadcast the single data dictionary to all ranks
    data = comm.bcast(data_to_bcast, root=0)

    # Broadcast to all ranks so everyone can plot their subset
    particles    = data['particles']
    trajectories = data['trajectories']
    first_chunk = params.first_chunk
    last_chunk  = params.last_chunk
    # first_chunk  = data['first_chunk']
    # last_chunk   = data['last_chunk']

    # Now that all ranks have the correct chunk boundaries, this section can run safely.
    if hasattr(params, "first_chunk") and params.first_chunk is not None:
        first_chunk = max(int(params.first_chunk), int(first_chunk))
    if hasattr(params, "last_chunk") and params.last_chunk is not None:
        last_chunk = min(int(params.last_chunk), int(last_chunk))

    # This line will now work correctly on all ranks.
    frame_minutes = list(range(first_chunk*60, (last_chunk + 1)*60, 1))

    # Build a dict for O(1) chunk -> (X,Y) lookup per particle
    traj_by_chunk = []
    for traj in trajectories:
        d = {}
        for st in traj:
            try:
                d[int(st['chunk'])] = (float(st['X']), float(st['Y']))
            except Exception:
                pass
        traj_by_chunk.append(d)

    # Global lambda range for consistent color mapping
    lam_arr_local = libs.np.array([p['lambda'] for p in particles], dtype=float)
    lam_min_local = float(libs.np.nanmin(lam_arr_local)) if lam_arr_local.size else 0.0
    lam_max_local = float(libs.np.nanmax(lam_arr_local)) if lam_arr_local.size else 1.0
    try:
        from mpi4py import MPI
        lam_min = comm.allreduce(lam_min_local, op=MPI.MIN) if size > 1 else lam_min_local
        lam_max = comm.allreduce(lam_max_local, op=MPI.MAX) if size > 1 else lam_max_local
    except Exception:
        lam_min, lam_max = lam_min_local, lam_max_local
    LAM_NORM = libs.mcolors.Normalize(vmin=lam_min, vmax=lam_max)
    LAM_CMAP = 'viridis'

    # Build the full list of frame chunks and split work across ranks
    frame_chunks = list(range(first_chunk, last_chunk + 1))
    num_frames = len(frame_chunks)
    per   = int(math.ceil(num_frames / float(size)))
    start = rank * per
    end   = min(start + per, num_frames)
    my_frame_ids = list(range(start, end))          # indices into frame_chunks

    # If data wasn't passed, load it as a fallback. Otherwise, use the provided data.
    if metadata is None or chunk_data_dict is None or chunk_interpolators is None:
        libs.logging.info(f"[rank {rank}] Field data not provided, loading from disk...")
        my_chunks = set(frame_chunks)
        # my_chunks = { frame_chunks[i] for i in my_frame_ids}
        # This correctly loads all necessary data and interpolators
        metadata, chunk_data_dict, chunk_interps = build_interpolators_for_chunks(my_chunks)
    else:
        # This block executes when data is passed in, but it was flawed.
        libs.logging.info(f"[rank {rank}] Using pre-loaded field data for animation.")
        chunk_interps = chunk_interpolators # Use the passed interpolators

    missing = [c for c, d in chunk_data_dict.items() if 'PVG' not in d]
    if missing:
        libs.logging.warning(f"PVG not present in chunks: {missing[:10]}{' ...' if len(missing)>10 else ''}")

    # Output frames directory
    frames_dir = libs.os.path.join(libs.os.path.dirname(out_mp4) or ".", "frames_original")
    if rank == 0:
        libs.os.makedirs(frames_dir, exist_ok=True)
    comm.Barrier()

    # Draw each assigned frame
    for fidx in my_frame_ids:
        ch = frame_chunks[fidx]

        fig, ax = libs.plt.subplots(figsize=(8, 8))
        ax.set_xlabel('XMIN'); ax.set_ylabel('YMIN')
        ax.set_xlim([10, -25]); ax.set_ylim([15, -15])
        ax.set_aspect('equal'); ax.grid(False)

        # Title + overlays (ellipse + GEO)
        tsec = float(data_init.chunk_to_seconds(ch))
        hh = int(tsec // 3600); mm = int((tsec % 3600) // 60); ss = tsec % 60
        ax.set_title(f'T={hh:02d}:{mm:02d}:{ss:05.2f}')
        tools.plot_ellipse_boundary(a1=params.a1, a2=params.a2, semiminor=params.semiminor,
                                    color='red', linestyle='--', plot=True, ax=ax)
        tools.add_circle(ax, fill_black=True, zorder=10)

        # --- PVG for this minute (fresh draw every frame) ---
        pvg_cs = None
        pvg_vmin = pvg_vmax = None
        has_chunk_data = ch in chunk_data_dict
        has_interps    = ch in chunk_interps
        has_pvg_array  = has_chunk_data and ('PVG' in chunk_data_dict[ch])

        if has_chunk_data and has_interps and has_pvg_array:
            pvg_cs, pvg_vmin, pvg_vmax = plot_pvg_field(ax, chunk_data_dict[ch], chunk_interps[ch], metadata, ch)
        else:
            if not has_chunk_data:
                libs.logging.warning(f"[rank {rank}] Missing chunk arrays for chunk {ch}; skipping PVG.")
            elif not has_interps:
                libs.logging.warning(f"[rank {rank}] Missing interpolators for chunk {ch}; skipping PVG.")
            else:  # has_chunk_data but no PVG in it
                libs.logging.warning(f"[rank {rank}] Chunk {ch} has no PVG dataset; skipping PVG.")

        # --- Colorbars (PVG + lambda) ---
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax_lam = divider.append_axes("right", size="4%",  pad=0.2)
        sm_lam = libs.cm.ScalarMappable(norm=LAM_NORM, cmap=libs.cm.get_cmap(LAM_CMAP))
        sm_lam.set_array([])
        cb_lam = fig.colorbar(sm_lam, cax=cax_lam)

        fmt_lam = ScalarFormatter(useMathText=True)
        fmt_lam.set_scientific(True)          # always allow sci notation
        fmt_lam.set_powerlimits((-3, 3))      # switch to sci outside [1e-3, 1e3]
        cb_lam.formatter = fmt_lam
        cb_lam.update_ticks()

        cb_lam.set_label("Lambda Value")

        # Create the PVG colorbar ONLY if pvg_cs was successfully created
        if pvg_cs is not None:
            cax_pvg = divider.append_axes("bottom", size="4%", pad=0.55)
            cb_pvg = fig.colorbar(pvg_cs, cax=cax_pvg, orientation='horizontal')

            cb_pvg.set_label("PVG", labelpad=10)
            cb_pvg.locator = LogLocator(base=10, subs=(0.05, 0.1))

            # Plain decimal formatter (no 1e-2):
            def _plain_decimal(x, pos):
                # Render as a trimmed fixed-point decimal, e.g. 0.01, 0.05, 0.1, 0.2
                s = f"{x:.6f}"             # enough precision
                s = s.rstrip('0').rstrip('.')  # trim
                return s if s else "0"

            cb_pvg.formatter = FuncFormatter(_plain_decimal)
            cb_pvg.update_ticks()

        # Particles at THIS minute
        # [plotimate.py]
        # Particles at THIS minute
        for i, p in enumerate(particles):
            birth_chunk = int(p['chunk'])
            stop_chunk = p.get('stop_chunk')

            # 1. APPEARANCE CHECK: Don't draw particle if animation time is after its birth time.
            if ch > birth_chunk:
                continue

            # 2. STOPPING CHECK: If animation time is at or before the particle's stop time,
            #    it should be drawn as a static 'x' at its final position.
            if stop_chunk is not None and ch <= int(stop_chunk):
                final_X, final_Y = p.get('final_X'), p.get('final_Y')
                if not (libs.np.isnan(final_X) or libs.np.isnan(final_Y)):
                    ax.plot(final_X, final_Y, marker='x', color='gray', markersize=6, zorder=15)
                # This particle's drawing for this frame is done, so we continue to the next particle.
                continue

            # 3. NORMAL TRAJECTORY DRAWING: If the particle is active and hasn't stopped,
            #    draw it at its position for the current chunk.
            xy = traj_by_chunk[i].get(ch) 
            if xy is not None:
                x, y = xy
                marker = '^' if p['species'] == 'e' else 'o'
                color = libs.cm.get_cmap(LAM_CMAP)(LAM_NORM(p['lambda']))
                ax.plot(x, y, marker=marker, color=color, markersize=4, zorder=15)

        # --- MONITORING BLOCK ---
        if rank == 0:
            # The current frame index `fidx` (since rank 0 starts at index 0)
            # directly corresponds to the number of frames completed.
            frames_done_total = fidx + 1
            progress_percent = (frames_done_total / num_frames) * 100 if num_frames > 0 else 100.0

            elapsed_time = libs.time.time() - render_start_time

            # ROBUST FIX: Calculate time per frame using the correct variable (frames_done_total),
            # which is guaranteed to exist and be > 0 in this loop.
            time_per_frame = elapsed_time / frames_done_total

            frames_remaining = num_frames - frames_done_total
            eta_seconds = frames_remaining * time_per_frame

            status = {
                "stage": "Rendering Animation",
                "current_frame": frames_done_total,
                "total_frames": num_frames,
                "progress_percent": f"{progress_percent:.2f}%",
                "elapsed_time_str": f"{int(elapsed_time // 3600):02d}:{int((elapsed_time % 3600) // 60):02d}:{int(elapsed_time % 60):02d}",
                "eta_str": f"{int(eta_seconds // 3600):02d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"
            }

            with open(status_file_path, 'w') as f:
                libs.json.dump(status, f, indent=4)
        # --- END MONITORING ---
        
        # Save frame and close
        png_path = libs.os.path.join(frames_dir, f"frame_{fidx:06d}.png")
        fig.savefig(png_path, dpi=50, bbox_inches='tight')
        libs.plt.close(fig)

    comm.Barrier()

    # Assemble MP4 on rank 0
    if rank == 0:
        import glob as _glob, subprocess, shutil
        frame_list = sorted(_glob.glob(os.path.join(frames_dir, "frame_*.png")))
        if not frame_list:
            libs.logging.error(f"[rank 0] No frames found in {frames_dir}.")
            return

        codecs_to_try = [["libx264", "yuv420p"], ["libopenh264", "yuv420p"], ["mpeg4", "yuv420p"]]
        ffmpeg_err = None
        for codec, pixfmt in codecs_to_try:
            cmd = [
                "ffmpeg", "-y", "-framerate", str(fps),
                "-pattern_type", "glob", "-i", os.path.join(frames_dir, "frame_*.png"),
                "-c:v", codec, "-pix_fmt", pixfmt, out_mp4
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                libs.logging.info(f"[rank 0] Wrote animation with ffmpeg ({codec}): {out_mp4}")
                break
            except subprocess.CalledProcessError as e:
                ffmpeg_err = e
                libs.logging.warning(f"[rank 0] ffmpeg failed with {codec}: {e.returncode}\n"
                                     f"stderr:\n{e.stderr.decode(errors='ignore')[:800]}...")
        else:
            try:
                import imageio, imageio.v3 as iio
                try: import imageio_ffmpeg  # noqa: F401
                except Exception: pass
                with imageio.get_writer(out_mp4, fps=fps) as w:
                    for fp in frame_list:
                        w.append_data(iio.imread(fp))
                libs.logging.info(f"[rank 0] Wrote animation via imageio: {out_mp4}")
            except Exception as e2:
                libs.logging.error(f"[rank 0] Failed to write MP4 with imageio after ffmpeg error ({ffmpeg_err}). "
                                   f"Last error: {e2}")

        if getattr(params, "cleanup_animation_frames", True):
            shutil.rmtree(frames_dir, ignore_errors=True)

    comm.Barrier()
