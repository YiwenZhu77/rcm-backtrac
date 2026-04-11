# monitor_render.py
import json
import time
import sys
import os
from datetime import datetime

STATUS_FILE = "render_status.json"

def display_progress():
    """Reads the render status file and prints a formatted progress bar."""
    try:
        if not os.path.exists(STATUS_FILE):
            sys.stdout.write("\rWaiting for rendering to start...")
            sys.stdout.flush()
            return

        with open(STATUS_FILE, 'r') as f:
            status = json.load(f)

        percent = float(status['progress_percent'][:-1])
        progress_bar_length = 40
        filled_length = int(progress_bar_length * percent / 100)
        bar = '█' * filled_length + '-' * (progress_bar_length - filled_length)
        
        current_time_str = datetime.now().strftime("%H:%M:%S")

        output_str = (
            f"\rRendering: [{bar}] {status['progress_percent']} | "
            f"Frame: {status['current_frame']}/{status['total_frames']} | "
            f"Elapsed: {status['elapsed_time_str']} | "
            f"ETA: {status['eta_str']} | "
            f"Now: {current_time_str}   "
        )
        sys.stdout.write(output_str)
        sys.stdout.flush()

        if percent >= 100:
            print("\nRendering has likely completed.")
            return True # Signal to exit

    except (json.JSONDecodeError, FileNotFoundError):
        sys.stdout.write("\rReading render status file...")
        sys.stdout.flush()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        return True

    return False

if __name__ == "__main__":
    print(f"--- Monitoring rendering progress from '{STATUS_FILE}' ---")
    print("Press Ctrl+C to stop monitoring.")
    try:
        while True:
            done = display_progress()
            if done:
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
