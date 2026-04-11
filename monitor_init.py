import time
import os
import sys
from datetime import datetime
import json

LOG_DIR = "logs"
RANK0_LOG = os.path.join(LOG_DIR, "backtrack_simulation.rank0.log")
RUN_STATUS = "run_status.json"


def get_recent_lines(file_path, n=5):
    try:
        with open(file_path, "rb") as f:
            try:
                f.seek(-4096, os.SEEK_END)
            except OSError:
                f.seek(0)
            text = f.read().decode(errors="ignore")
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            return lines[-n:]
    except Exception:
        return []


def interpret_stage(lines):
    if not lines:
        return "Waiting for log file creation..."

    text = " ".join(lines).lower()

    if "reading initial particles" in text:
        return "Reading Particle HDF5 File (Rank 0)"
    if "reading lost particles" in text:
        return "Reading Lost Particles File (Rank 0)"
    if "distribute_particles" in text or "received" in text:
        return "MPI Distribution (Scattering particles to ranks)"
    if "load_chunk" in text:
        return "Loading Field Data (Reading chunks for start time)"
    if "initialize_particles" in text and "took" not in text:
        return "Initializing Particles (Interpolating Positions)"
    if "saved initialization plot" in text:
        return "Generating Initialization PNG"
    if "starting backtracking" in text:
        return "INITIALIZATION COMPLETE"

    return "Working..."


def monitor():
    print(f"--- Monitoring Initialization via '{RANK0_LOG}' ---")
    print("Press Ctrl+C to stop.\n")

    job_start = time.time()

    try:
        while True:

            # Early exit ONLY if run_status.json is fresh (written after job start)
            if os.path.exists(RUN_STATUS):
                s_stat = os.stat(RUN_STATUS)
                if s_stat.st_mtime >= job_start:
                    try:
                        import json
                        with open(RUN_STATUS, "r") as f:
                            status = json.load(f)
                            
                        stage = (status.get("stage") or "").lower()
                        if "running" in stage or "complete" in stage:
                            print("\nrun_status.json from THIS run indicates initialization finished!")
                            print("Switch to: python monitor_run.py")
                            return
                        
                    except Exception:
                        pass  # ignore partial writes

            if not os.path.exists(LOG_DIR):
                sys.stdout.write("\rWaiting for 'logs/' directory...")
                sys.stdout.flush()
                time.sleep(2)
                continue

            if not os.path.exists(RANK0_LOG):
                sys.stdout.write("\rWaiting for Rank 0 log file...")
                sys.stdout.flush()
                time.sleep(2)
                continue

            stats = os.stat(RANK0_LOG)
            last_mod = stats.st_mtime
            age = time.time() - last_mod
            size_mb = stats.st_size / (1024 * 1024)

            lines = get_recent_lines(RANK0_LOG)
            stage = interpret_stage(lines)

            # Health logic
            if size_mb == 0 and age > 60:
                health = "NO OUTPUT (Program stuck before logging?)"
            elif age > 300:
                health = "POTENTIALLY HUNG (>5 min no update)"
            elif age > 60:
                health = "SLOW / HEAVY COMPUTATION"
            else:
                health = "RUNNING"

            sys.stdout.write("\033[K")
            sys.stdout.write(
                f"\rStatus: [{health}] | Stage: {stage} | "
                f"Log Size: {size_mb:.2f} MB | Last Update: {int(age)}s ago"
            )
            sys.stdout.flush()

            if "INITIALIZATION COMPLETE" in stage:
                print("\nInitialization finished! Switch to monitor_run.py")
                return

            time.sleep(2)

    except KeyboardInterrupt:
        print("\nStopped monitoring.")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    monitor()
