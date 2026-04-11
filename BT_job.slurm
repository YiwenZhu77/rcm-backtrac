#!/bin/bash
#SBATCH --job-name=backtrack
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --time=06:00:00
#SBATCH --output=backtrack_%j.out
#SBATCH --error=backtrack_%j.err

# Activate your environment
source $HOME/Python1/venv/bin/activate

# Set libraries and paths
export PATH=$HOME/backtrack_env/install/openmpi-4.1.5/bin:$PATH
export LD_LIBRARY_PATH=$HOME/backtrack_env/install/openmpi-4.1.5/lib:$HOME/backtrack_env/install/hdf5-1.12.1/lib:$LD_LIBRARY_PATH
export HDF5_USE_FILE_LOCKING=FALSE
export OMPI_MCA_btl_vader_single_copy_mechanism=none

# Run your code
cd $SLURM_SUBMIT_DIR
mpirun -np 18 python $HOME/BT_main_RUN0/backtrack_particles.py
