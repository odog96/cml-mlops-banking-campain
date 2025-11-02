#!/usr/bin/env python3
"""
Module 1 Test Runner - Convenient Entry Point

Run all numbered scripts sequentially to test the pipeline in your environment.

Usage:
    python run_tests.py                          # Run all scripts (exits on first failure)
    python run_tests.py --skip 01,04             # Skip specific scripts
    python run_tests.py --only 01,02,03          # Run only specific scripts
    python run_tests.py --verbose                # Show full output
    python run_tests.py --timeout 300            # Set timeout per script
    python run_tests.py --no-exit-on-failure     # Continue running all tests even if one fails

Quick examples:
    python run_tests.py                          # Full pipeline test (exits on first failure)
    python run_tests.py --skip 04                # Skip deployment
    python run_tests.py --only 01,02             # Test ingestion and EDA only
    python run_tests.py --skip 05_1,05_2         # Skip inference pipeline
    python run_tests.py --no-exit-on-failure     # Run all tests regardless of failures

Output:
    - Console: Execution summary and status
    - test_results.log: Detailed execution log
    - test_results.json: Machine-readable results
"""

import sys
from pathlib import Path

# Add module1 folder and parent directory to path
module1_path = Path(__file__).parent.parent
cdsw_path = module1_path.parent  # /home/cdsw
sys.path.insert(0, str(module1_path))
sys.path.insert(0, str(cdsw_path))

from helpers.test_runner import TestRunner, main

if __name__ == '__main__':
    main()
