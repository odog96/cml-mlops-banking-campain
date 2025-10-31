"""
Module 1 Test Runner
====================

Automated test runner that executes all numbered scripts sequentially
and tracks their execution status and any errors.

Usage:
    python test_runner.py                          # Run all scripts
    python test_runner.py --skip 01,04             # Skip specific scripts
    python test_runner.py --only 01,02,03          # Run only specific scripts
    python test_runner.py --verbose                # Show full output
    python test_runner.py --timeout 300            # Set timeout per script (seconds)

Output:
    - Execution summary to console
    - Detailed log file: test_results.log
    - Results JSON: test_results.json
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class TestRunner:
    """Runs all numbered scripts in the module1 directory sequentially."""

    def __init__(self, module_path: str = None, verbose: bool = False,
                 timeout: int = 600, skip: List[str] = None, only: List[str] = None):
        """
        Initialize the test runner.

        Args:
            module_path: Path to module1 directory (auto-detected if not provided)
            verbose: Print full output from each script
            timeout: Timeout for each script in seconds (default: 600)
            skip: List of script numbers to skip (e.g., ['04', '06'])
            only: List of script numbers to run (e.g., ['01', '02', '03'])
        """
        if module_path is None:
            # Get the parent of helpers folder (module1 directory)
            module_path = str(Path(__file__).parent.parent)

        self.module_path = Path(module_path)
        self.verbose = verbose
        self.timeout = timeout
        self.skip_scripts = set(skip or [])
        self.only_scripts = set(only or [])

        self.results = {}
        self.start_time = None
        self.end_time = None

        # Define the pipeline - all numbered scripts
        self.pipeline = [
            ('01_ingest.py', 'Data Ingestion'),
            ('02_eda_notebook.ipynb', 'EDA Analysis (Jupyter)'),
            ('03_train_quick.py', 'Quick Model Training'),
            ('04_deploy.py', 'Model Deployment'),
            ('05_1_inference_data_prep.py', 'Inference Data Preparation'),
            ('05_2_inference_predict.py', 'Inference Predictions'),
        ]

    def should_run(self, script_name: str) -> bool:
        """Check if a script should be run based on skip/only filters."""
        # Extract script number (e.g., '01' from '01_ingest.py')
        script_num = script_name.split('_')[0]

        if self.only_scripts and script_num not in self.only_scripts:
            return False

        if script_num in self.skip_scripts:
            return False

        return True

    def run_python_script(self, script_path: Path) -> Tuple[int, str]:
        """
        Run a Python script and capture output.

        Args:
            script_path: Path to the script

        Returns:
            Tuple of (return_code, output)
        """
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.module_path),
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return result.returncode, result.stdout + result.stderr

        except subprocess.TimeoutExpired:
            return -1, f"Script timed out after {self.timeout} seconds"
        except Exception as e:
            return -1, f"Error running script: {str(e)}"

    def run_jupyter_notebook(self, notebook_path: Path) -> Tuple[int, str]:
        """
        Run a Jupyter notebook and capture output.

        Args:
            notebook_path: Path to the notebook

        Returns:
            Tuple of (return_code, output)
        """
        try:
            # Try using nbconvert to run notebook
            output_notebook = notebook_path.parent / f"{notebook_path.stem}_output.ipynb"

            result = subprocess.run(
                [
                    'jupyter', 'nbconvert',
                    '--to', 'notebook',
                    '--execute',
                    '--output', str(output_notebook),
                    str(notebook_path)
                ],
                cwd=str(self.module_path),
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            return result.returncode, result.stdout + result.stderr

        except FileNotFoundError:
            return -1, "Jupyter nbconvert not found. Install with: pip install nbconvert"
        except subprocess.TimeoutExpired:
            return -1, f"Notebook timed out after {self.timeout} seconds"
        except Exception as e:
            return -1, f"Error running notebook: {str(e)}"

    def run_script(self, script_name: str, description: str) -> Dict:
        """
        Run a single script and record results.

        Args:
            script_name: Name of the script
            description: Human-readable description

        Returns:
            Dictionary with execution results
        """
        script_path = self.module_path / script_name
        script_num = script_name.split('_')[0]

        result = {
            'script': script_name,
            'description': description,
            'number': script_num,
            'status': 'PENDING',
            'return_code': None,
            'output': '',
            'error': None,
            'duration': 0,
            'timestamp': datetime.now().isoformat()
        }

        # Check if file exists
        if not script_path.exists():
            result['status'] = 'SKIPPED'
            result['error'] = f"File not found: {script_path}"
            return result

        # Skip if necessary
        if not self.should_run(script_name):
            result['status'] = 'SKIPPED'
            result['error'] = 'Filtered by --skip or --only flags'
            return result

        print(f"\n{'='*80}")
        print(f"Running: {script_num} - {description}")
        print(f"File: {script_name}")
        print(f"{'='*80}")

        start = datetime.now()

        # Run based on file type
        if script_name.endswith('.ipynb'):
            return_code, output = self.run_jupyter_notebook(script_path)
        else:
            return_code, output = self.run_python_script(script_path)

        duration = (datetime.now() - start).total_seconds()

        # Update result
        result['return_code'] = return_code
        result['output'] = output
        result['duration'] = duration

        if return_code == 0:
            result['status'] = 'PASSED'
            print(f"✅ PASSED in {duration:.2f}s")
        else:
            result['status'] = 'FAILED'
            result['error'] = f"Exit code: {return_code}"
            print(f"❌ FAILED (exit code: {return_code})")

        # Print output if verbose
        if self.verbose:
            print("\n--- Output ---")
            print(output[:2000])  # Limit to first 2000 chars
            if len(output) > 2000:
                print(f"\n... (output truncated, {len(output)} total chars)")

        return result

    def run_all(self) -> Dict:
        """
        Run all scripts in the pipeline.

        Returns:
            Dictionary with all results
        """
        self.start_time = datetime.now()
        print(f"\n{'='*80}")
        print("MODULE 1 TEST RUNNER - SEQUENTIAL EXECUTION")
        print(f"{'='*80}")
        print(f"Module path: {self.module_path}")
        print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Timeout per script: {self.timeout}s")
        print(f"Verbose: {self.verbose}")

        for script_name, description in self.pipeline:
            result = self.run_script(script_name, description)
            self.results[script_name] = result

        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        # Print summary
        self._print_summary(duration)

        # Save results
        self._save_results()

        return self.results

    def _print_summary(self, total_duration: float):
        """Print execution summary."""
        print(f"\n{'='*80}")
        print("TEST EXECUTION SUMMARY")
        print(f"{'='*80}")

        passed = sum(1 for r in self.results.values() if r['status'] == 'PASSED')
        failed = sum(1 for r in self.results.values() if r['status'] == 'FAILED')
        skipped = sum(1 for r in self.results.values() if r['status'] == 'SKIPPED')
        total = len(self.results)

        print(f"\nResults:")
        print(f"  Passed:  {passed}/{total} ✅")
        print(f"  Failed:  {failed}/{total} ❌")
        print(f"  Skipped: {skipped}/{total} ⏭️")

        print(f"\nTiming:")
        print(f"  Total time:  {total_duration:.2f}s ({total_duration/60:.2f}m)")
        print(f"  Start time:  {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  End time:    {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"\nDetails:")
        for script_name, result in self.results.items():
            status_symbol = {
                'PASSED': '✅',
                'FAILED': '❌',
                'SKIPPED': '⏭️'
            }.get(result['status'], '❓')

            print(f"  {status_symbol} {result['number']} - {result['description']}")
            if result['status'] == 'PASSED':
                print(f"      Duration: {result['duration']:.2f}s")
            else:
                print(f"      Error: {result['error']}")

        print(f"\n{'='*80}")

        if failed == 0:
            print("✅ ALL TESTS PASSED!")
        else:
            print(f"❌ {failed} TEST(S) FAILED")
            print("\nFailed scripts:")
            for script_name, result in self.results.items():
                if result['status'] == 'FAILED':
                    print(f"  • {script_name}: {result['error']}")

        print(f"{'='*80}\n")

    def _save_results(self):
        """Save detailed results to files."""
        log_path = self.module_path / 'test_results.log'
        json_path = self.module_path / 'test_results.json'

        # Save as JSON
        results_json = {
            'summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat(),
                'total_duration': (self.end_time - self.start_time).total_seconds(),
                'passed': sum(1 for r in self.results.values() if r['status'] == 'PASSED'),
                'failed': sum(1 for r in self.results.values() if r['status'] == 'FAILED'),
                'skipped': sum(1 for r in self.results.values() if r['status'] == 'SKIPPED'),
            },
            'results': self.results
        }

        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2, default=str)

        print(f"Results saved to: {json_path}")

        # Save as detailed log
        with open(log_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODULE 1 TEST EXECUTION LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Start time: {self.start_time.isoformat()}\n")
            f.write(f"End time: {self.end_time.isoformat()}\n")
            f.write(f"Total duration: {(self.end_time - self.start_time).total_seconds():.2f}s\n\n")

            for script_name, result in self.results.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"Script: {result['number']} - {result['description']}\n")
                f.write(f"File: {script_name}\n")
                f.write(f"Status: {result['status']}\n")
                f.write(f"Duration: {result['duration']:.2f}s\n")
                f.write(f"Timestamp: {result['timestamp']}\n")

                if result['error']:
                    f.write(f"Error: {result['error']}\n")

                if result['output']:
                    f.write(f"\nOutput:\n{result['output']}\n")

        print(f"Log saved to: {log_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Module 1 Test Runner - Execute all scripts sequentially',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_runner.py                    # Run all scripts
  python test_runner.py --skip 04          # Skip script 04
  python test_runner.py --only 01,02,03    # Run only scripts 01, 02, 03
  python test_runner.py --verbose          # Show full output
  python test_runner.py --timeout 300      # 5 minute timeout per script
        """
    )

    parser.add_argument(
        '--skip',
        type=str,
        default=None,
        help='Comma-separated list of script numbers to skip (e.g., 04,06)'
    )

    parser.add_argument(
        '--only',
        type=str,
        default=None,
        help='Comma-separated list of script numbers to run (e.g., 01,02,03)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print full output from each script'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=600,
        help='Timeout for each script in seconds (default: 600)'
    )

    parser.add_argument(
        '--module-path',
        type=str,
        default=None,
        help='Path to module1 directory (auto-detected if not provided)'
    )

    args = parser.parse_args()

    # Parse skip and only lists
    skip_list = [s.strip() for s in args.skip.split(',')] if args.skip else []
    only_list = [s.strip() for s in args.only.split(',')] if args.only else []

    # Create and run test runner
    runner = TestRunner(
        module_path=args.module_path,
        verbose=args.verbose,
        timeout=args.timeout,
        skip=skip_list,
        only=only_list
    )

    results = runner.run_all()

    # Exit with appropriate code
    failed = sum(1 for r in results.values() if r['status'] == 'FAILED')
    sys.exit(1 if failed > 0 else 0)


if __name__ == '__main__':
    main()
