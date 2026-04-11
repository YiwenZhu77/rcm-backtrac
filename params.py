# params


# Constants #here
RE = 6.37e6  # Earth radius in meters (updated to match RE = 6370e3 m)
r = RE + 1e5  # Radial distance from Earth's center (6470e3 m)  
B0 = 30570e-9  # Earth’s equatorial magnetic field strength in Tesla (30570 nT)
buffer_distance = 1.0
FIELD_CADENCE = 1  # Minutes per field dump in HDF5 (e.g., 5 if sparse)
last_chunk, first_chunk, time_step = 480, 0, -60.0  # Backtracking with negative time step 1021
epsilon = 1e-10  # Small value to avoid division by zero in fac
set_lambda_to_zero = False
int_method = 'rk4'  # 'euler' and 'rk4'
RK4_sub_step = 10

# Set to None to use the automatic, per-frame limits.
pvg_vmin_hardcoded = 5e-3
pvg_vmax_hardcoded = 7e-2
resolution_factor = 1.0  # you already have this; keep as-is unless you want to thin sampling

# Ellipse:
a1 = 8.5
a2 = -10.0
semiminor = 10.0
include_lost_particles = False
offset_lost_idx: bool = True
lost_idx_offset: int = 1000000
MAX_INITIAL_PARTICLES = None #10000
shuffle_initial = False
R_MIN: float | None = None  # Example: Only select particles outside 2 Earth radii
R_MAX: float | None = None  # Example: Only select particles inside 6 Earth radii
MAX_LOST_PARTICLES = 2

# Define file paths
#hdf5_file_path = "/glade/work/sinas/Nov2023/data_ion3h.h5"
#hdf5_file_path = "/glade/work/sinas/Nov2023/data_ion120min.h5"
#hdf5_file_path = "/glade/work/sinas/4_idealized_tests/RUN0/BT_in/RCM_data_RUN0.h5"
hdf5_file_path = "/home/ss323/Python1/BT_input/RCM_data_extreme_RCME.h5"
#particle_file_path = "/glade/work/sinas/Simple_test/initial_distribution_3h.h5"
particle_file_path = "/home/ss323/Python1/BT_input/id_extreme_rcme_initial_distribution_stratified_100000.h5"
#particle_file_path = "/glade/work/sinas/4_idealized_tests/RUN3/BT_in/simple_initial_distribution_particles.h5"
loss_data_path = "/home/ss323/Python1/BT_input/simple_lost_electrons_2.h5"

#save_ani_file_path= "/glade/work/sinas/Nov2023/BT_out/"
#save_ani_file_path= "/glade/work/sinas/4_idealized_tests/RUN3/BT_out/"
#save_ani_file_path= "/glade/work/sinas/4_idealized_tests/RUN0/BT_cas_out100k/"
save_ani_file_path= "/home/ss323/Python1/BT_output/"

output_hdf5_path = save_ani_file_path + "simulation_data.h5"
animation_file_path = save_ani_file_path + "particle_trajectories_v3.mp4"
initial_setup_path = save_ani_file_path + "particle_initialization.png"

code_path = "/home/ss323/Python1/BT_main_RUN3_extreme_rcme"
