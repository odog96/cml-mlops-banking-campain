#!/usr/bin/env python3
"""
Drift Detection Reporting App Launcher for Cloudera AI
Simple script to launch the Streamlit reporting application with proper configuration
"""

import subprocess
import sys
import os


def launch_app(app_script):
    """Launch the Streamlit application"""

    # Get the port assigned by Cloudera AI
    port = os.environ.get("CDSW_APP_PORT", "8100")

    print(f"Starting Drift Detection Reporting App on port {port}")

    # Launch Streamlit with Cloudera-compatible settings
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", app_script,
        "--server.port", port,
        "--server.address", "127.0.0.1"  # Cloudera uses 127.0.0.1, not 0.0.0.0
    ])


if __name__ == "__main__":
    print("🔧 Drift Detection Reporting App - Cloudera AI Launcher")
    print("=" * 60)

    # Get the script directory - use __file__ if available (normal Python scripts)
    # Otherwise search for the module3 directory from the project root
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # In Jupyter notebooks, __file__ is not defined
        # Search upward from cwd to find module3/reporting_main_app.py
        search_dir = os.getcwd()
        script_dir = None

        # First try: module3 subdirectory from current directory
        candidate = os.path.join(search_dir, "module3")
        if os.path.exists(os.path.join(candidate, "reporting_main_app.py")):
            script_dir = candidate
        # Second try: current directory itself
        elif os.path.exists(os.path.join(search_dir, "reporting_main_app.py")):
            script_dir = search_dir
        # Third try: hardcoded path as last resort
        else:
            script_dir = "/home/cdsw/module3"

    # Check if main app file exists
    app_script = os.path.join(script_dir, "reporting_main_app.py")
    if not os.path.exists(app_script):
        print("❌ Error: reporting_main_app.py not found")
        print(f"📁 Searched in: {script_dir}")
        print(f"📁 Please run this from: /home/cdsw or /home/cdsw/module3")
        sys.exit(1)

    print(f"✅ Found app at: {app_script}")

    # Launch application
    launch_app(app_script)
