#!/usr/bin/env python3
import subprocess
import sys

def install_requirements():
    """Install Python packages from requirements.txt"""
    try:
        print("Installing dependencies from requirements.txt...")
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "-r", 
            "requirements.txt"
        ])
        print("✓ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_requirements()
