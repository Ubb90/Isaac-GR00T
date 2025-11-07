#!/usr/bin/env python3
"""
Direct launcher script for Gr00T ROS2 evaluation that ensures correct Python environment.
This script bypasses the ROS2 entry point system to avoid Python version conflicts.
"""

import sys
import os

# Ensure we're using the correct Python environment
if 'CONDA_DEFAULT_ENV' in os.environ:
    print(f"Using conda environment: {os.environ['CONDA_DEFAULT_ENV']}")
else:
    print("Warning: No conda environment detected")

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Add the Isaac-GR00T path to sys.path
groot_path = os.path.expanduser("~/Documents/Isaac-GR00T")
if groot_path not in sys.path:
    sys.path.insert(0, groot_path)

# Import and run the main function
try:
    from so_100_track_groot.eval_lerobot_ros2 import main
    main()
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you've built the ROS2 package and sourced the workspace")
    sys.exit(1)
except Exception as e:
    print(f"Error running evaluation: {e}")
    sys.exit(1)