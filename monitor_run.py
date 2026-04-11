# monitor_run.py
import json
import time
import sys
import os
from datetime import datetime

STATUS_FILE = "run_status.json"

def display_progress(status):
    """Prints a formatted progress bar for the simulation."""
    percent = float(status['progress_percent'][:-1])
    progress_bar_length = 40
    filled_length = int(progress_bar_length * percent / 100)
    bar = '█' * filled_length + '-' * (progress_bar_length - filled_length)
    
    current_time_str = datetime.now().strftime("%H:%M:%S")

    output_str = (
        f"\rSimulating: [{bar}] {status['progress_percent']} | "
        #f"Chunk: {status['current_chunk']}/{status['stop_chunk']} | "
        f"Chunk: {status['current_chunk']} (Range: {status['start_chunk']} -> {status['stop_chunk']}) | "
        f"Elapsed: {status['elapsed_time_str']} | "
        f"ETA: {status['eta_str']} | "
        f"Now: {current_time_str}   "
    )
    sys.stdout.write(output_str)
    sys.stdout.flush()

def monitor():
    print(f"--- Monitoring simulation progress from '{STATUS_FILE}' ---")
    print("Press Ctrl+C to stop monitoring.")
    try:
        while True:
            try:
                if not os.path.exists(STATUS_FILE):
                    sys.stdout.write("\rWaiting for simulation to start...")
                    sys.stdout.flush()
                else:
                    with open(STATUS_FILE, 'r') as f:
                        status = json.load(f)
                    
                    if status.get("stage") == "Complete":
                        display_progress(status)
                        print("\nSimulation has completed.")
                        break
                    display_progress(status)

            except (json.JSONDecodeError, FileNotFoundError):
                sys.stdout.write("\rReading simulation status file...")
                sys.stdout.flush()
            
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    monitor()
