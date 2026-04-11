#!/bin/bash
JOB_NAME="Simple_test"
NPROCS=32
CODE_DIR="$HOME/Python1/BT_main_RUN3_extreme_rcme"
SCRIPT="$CODE_DIR/Simple_test.py"
LOGDIR="$CODE_DIR/logs"
SESSION_NAME="backtrack_run"
mkdir -p "$LOGDIR"

LOGFILE="$LOGDIR/${JOB_NAME}_$(date +%Y%m%d_%H%M%S).log"

# ----------------------------------------
# CREATE INNER SCRIPT FOR SCREEN SESSION
# ----------------------------------------
INNER_SCRIPT=$(mktemp /tmp/backtrack_inner_XXXX.sh)
cat > "$INNER_SCRIPT" <<EOF
#!/bin/bash

# === ENVIRONMENT MUST BE SET INSIDE SCREEN SESSION ===

# MPI + Python + HDF5 library paths
export PATH="$HOME/backtrack_env/install/openmpi-4.1.5/bin:\$PATH"
export LD_LIBRARY_PATH="$HOME/backtrack_env/install/python-3.10/lib:$HOME/backtrack_env/install/openmpi-4.1.5/lib:$HOME/backtrack_env/install/hdf5-1.12.1/lib:$LD_LIBRARY_PATH"
export HDF5_USE_FILE_LOCKING=FALSE
export HDF5_MPI_OPT_TYPES=TRUE
export OMPI_MCA_btl_vader_single_copy_mechanism=none

cd "$CODE_DIR"

echo "==============================================================="
echo "🔹 Starting job: $JOB_NAME"
echo "🔹 Host: \$(hostname)"
echo "🔹 Using LD_LIBRARY_PATH: \$LD_LIBRARY_PATH"
echo "🔹 Using PATH: \$PATH"
echo "🔹 Virtualenv: \$VIRTUAL_ENV"
echo "🔹 Running with $NPROCS MPI ranks"
echo "🔹 Logfile: $LOGFILE"
echo "Start time: \$(date)"
echo "==============================================================="

mpirun -np $NPROCS python "$SCRIPT" > "$LOGFILE" 2>&1

echo "---------------------------------------------------------------"
echo "Job completed at \$(date)"
echo "---------------------------------------------------------------"
EOF

chmod +x "$INNER_SCRIPT"

# ----------------------------------------
# START SCREEN SESSION
# ----------------------------------------
if screen -list | grep -q "$SESSION_NAME"; then
    echo "⚠️  A session named $SESSION_NAME already exists."
    exit 1
fi

echo "🚀 Launching run inside detached screen session: $SESSION_NAME"
screen -dmS "$SESSION_NAME" bash "$INNER_SCRIPT"

echo "✅ Backtrack run started in background."
echo "   Attach:  screen -r $SESSION_NAME"
echo "   Logs:    $LOGFILE"
