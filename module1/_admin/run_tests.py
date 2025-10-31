#!/usr/bin/env python3
"""
Module 1 Test Runner - Convenient Entry Point

Run all numbered scripts sequentially to test the pipeline in your environment.

Usage:
    python run_tests.py                          # Run all scripts
    python run_tests.py --skip 01,04             # Skip specific scripts
    python run_tests.py --only 01,02,03          # Run only specific scripts
    python run_tests.py --verbose                # Show full output
    python run_tests.py --timeout 300            # Set timeout per script

Quick examples:
    python run_tests.py                          # Full pipeline test
    python run_tests.py --skip 04                # Skip deployment
    python run_tests.py --only 01,02             # Test ingestion and EDA only
    python run_tests.py --skip 05_1,05_2         # Skip inference pipeline

Output:
    - Console: Execution summary and status
    - test_results.log: Detailed execution log
    - test_results.json: Machine-readable results
"""

import sys
from pathlib import Path

# Add module1 folder to path (parent of _admin)
module1_path = Path(__file__).parent.parent
sys.path.insert(0, str(module1_path))

from helpers.test_runner import TestRunner, main

if __name__ == '__main__':
    main()
