#!/usr/bin/env python3
import subprocess
import sys
import os

def install_requirements():
    """Install Python packages from requirements.txt"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(script_dir, "requirements.txt")
    
    # Check if requirements.txt exists
    if not os.path.exists(requirements_path):
        print(f"✗ requirements.txt not found at: {requirements_path}")
        print(f"   Script directory: {script_dir}")
        print(f"   Current working directory: {os.getcwd()}")
        sys.exit(1)
    
    try:
        print(f"Installing dependencies from {requirements_path}...")
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "-r", 
            requirements_path
        ])
        print("✓ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_requirements()
