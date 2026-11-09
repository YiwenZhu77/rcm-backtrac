# params — adapted for MAGE sp13_075 output

# Constants
RE = 6.37e6  # Earth radius in meters
r = RE + 1e5  # Radial distance from Earth's center (ionosphere)
B0 = 30570e-9  # Earth's equatorial magnetic field strength in Tesla (30570 nT)
buffer_distance = 1.0
FIELD_CADENCE = 1  # Minutes per field dump in HDF5
last_chunk, first_chunk, time_step = 240, 0, -60.0  # 241 chunks (t=21600 to t=36000)
epsilon = 1e-10
set_lambda_to_zero = False
int_method = 'rk4'
RK4_sub_step = 10

# MAGE-specific: COLAT is in radians (not degrees)
colat_in_radians = True

# Set to None to use the automatic, per-frame limits.
pvg_vmin_hardcoded = 5e-3
pvg_vmax_hardcoded = 7e-2
resolution_factor = 1.0

# Ellipse (RCM domain boundary):
a1 = 8.5
a2 = -10.0
semiminor = 10.0
include_lost_particles = False
offset_lost_idx: bool = True
lost_idx_offset: int = 1000000
MAX_INITIAL_PARTICLES = None
shuffle_initial = False
R_MIN: float | None = 4.0
R_MAX: float | None = 6.6
MAX_LOST_PARTICLES = 2

# File paths — MAGE sp13_075
hdf5_file_path = "/path/to/pilot-data/sp13_075_rcm_mage.h5"
particle_file_path = "/path/to/pilot-data/sina_1k_particles.h5"
loss_data_path = ""

save_ani_file_path = "/path/to/pilot-data/sina_out/"
output_hdf5_path = save_ani_file_path + "simulation_data.h5"
animation_file_path = save_ani_file_path + "particle_trajectories.mp4"
initial_setup_path = save_ani_file_path + "particle_initialization.png"

code_path = "/path/to/rcm-backtrac"
