# monitor_assemble.py
import json
import time
import sys
import os
from datetime import datetime

STATUS_FILE = "assemble_status.json"

def display_progress(status):
    """Prints a formatted progress bar for the assembly."""
    percent = status.get('progress_percent', 0.0)
    progress_bar_length = 40
    filled_length = int(progress_bar_length * percent / 100)
    bar = '█' * filled_length + '-' * (progress_bar_length - filled_length)
    
    current_time_str = datetime.now().strftime("%H:%M:%S")

    output_str = (
        f"\rAssembling: [{bar}] {percent:.2f}% | "
        f"Files: {status.get('files_processed', 0)}/{status.get('total_files', 0)} | "
        f"Elapsed: {status.get('elapsed_time_str', '00:00:00')} | "
        f"Now: {current_time_str}   "
    )
    sys.stdout.write(output_str)
    sys.stdout.flush()

def monitor():
    print(f"--- Monitoring assembly progress from '{STATUS_FILE}' ---")
    print("Press Ctrl+C to stop monitoring.")
    try:
        while True:
            try:
                if not os.path.exists(STATUS_FILE):
                    sys.stdout.write("\rWaiting for assembly to start...")
                    sys.stdout.flush()
                else:
                    with open(STATUS_FILE, 'r') as f:
                        status = json.load(f)
                    
                    if status.get("stage") == "Complete":
                        display_progress(status)
                        print("\nAssembly has completed.")
                        break
                    display_progress(status)

            except (json.JSONDecodeError, FileNotFoundError):
                sys.stdout.write("\rReading assembly status file...")
                sys.stdout.flush()
            
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    monitor()
