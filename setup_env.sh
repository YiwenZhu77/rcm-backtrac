#!/bin/bash

# 1. Activate the Virtual Environment
echo "🔹 Activating venv: venv310"
source "$HOME/Python1/venv310/bin/activate"

# 2. Export all required library paths for your custom-compiled tools
echo "🔹 Setting library paths for Python, MPI, and HDF5"

# Export PATH for MPI executables (like mpirun)
export PATH="$HOME/backtrack_env/install/openmpi-4.1.5/bin:$PATH"

# Export LD_LIBRARY_PATH for all .so files (Python, MPI, HDF5)
export LD_LIBRARY_PATH="$HOME/backtrack_env/install/python-3.10/lib:$HOME/backtrack_env/install/openmpi-4.1.5/lib:$HOME/backtrack_env/install/hdf5-1.12.1/lib:$LD_LIBRARY_PATH"

# 3. Set HDF5/MPI environment flags
export HDF5_USE_FILE_LOCKING=FALSE
export HDF5_MPI_OPT_TYPES=TRUE
export OMPI_MCA_btl_vader_single_copy_mechanism=none

echo "✅ Environment is ready."
