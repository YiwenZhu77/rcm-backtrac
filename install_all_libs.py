#!/usr/bin/env python3
"""
install_all_libs.py
-------------------
Installs all Python packages required for your Backtrack project.
Safe to re-run anytime. Automatically upgrades pip first.
"""

import subprocess
import sys

# List of required packages (from libs.py)
packages = [
    "numpy",
    "mpi4py",
    "psutil",
    "h5py",
    "scipy",
    "matplotlib",
    "tqdm"
]

def main():
    print("🔹 Upgrading pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)

    print("🔹 Installing required packages:")
    for pkg in packages:
        print(f"   ➤ {pkg}")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)

    print("\n✅ All libraries installed successfully!")

if __name__ == "__main__":
    main()
