#!/usr/bin/env python3
"""
EcoZone AI Smart Energy Management System - Standalone CLI Test Runner
Supports --tier, --feature, --json-output, -v/--verbose, and exits 0 on test pass.
"""

import sys
import os
import time
import json
import argparse
import platform
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ANSI Color Codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


class JsonReportPlugin:
    """Pytest plugin to collect test execution metrics and write JSON report."""
    def __init__(self, output_path=None):
        self.output_path = output_path
        self.results = []
        self.start_time = time.time()

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            nodeid = report.nodeid
            outcome = report.outcome  # 'passed', 'failed', 'skipped'
            duration = report.duration

            # Identify tier from nodeid or markers
            tier = "unknown"
            if "tier1" in nodeid or "test_tier1" in nodeid:
                tier = "tier1"
            elif "tier2" in nodeid or "test_tier2" in nodeid:
                tier = "tier2"
            elif "tier3" in nodeid or "test_tier3" in nodeid:
                tier = "tier3"
            elif "tier4" in nodeid or "test_tier4" in nodeid:
                tier = "tier4"
            elif "hardware" in nodeid:
                tier = "hardware"
            elif "challenger" in nodeid:
                tier = "challenger"

            # Identify feature
            feature = "unknown"
            for i in range(1, 12):
                if f"feature({i})" in nodeid or f"test_f{i}_" in nodeid or f"test_f{i:02d}_" in nodeid or f"Feature {i}" in str(report.keywords):
                    feature = f"feature_{i}"
                    break

            self.results.append({
                "nodeid": nodeid,
                "outcome": outcome,
                "duration": duration,
                "tier": tier,
                "feature": feature,
                "error": str(report.longrepr) if report.failed else None
            })

    def pytest_sessionfinish(self, session, exitstatus):
        total_duration = time.time() - self.start_time
        total = len(self.results)
        passed = sum(1 for r in self.results if r["outcome"] == "passed")
        failed = sum(1 for r in self.results if r["outcome"] == "failed")
        skipped = sum(1 for r in self.results if r["outcome"] == "skipped")

        # Tier breakdown
        tier_breakdown = {}
        for t in ["tier1", "tier2", "tier3", "tier4", "hardware", "challenger"]:
            t_results = [r for r in self.results if r["tier"] == t]
            if t_results:
                tier_breakdown[t] = {
                    "total": len(t_results),
                    "passed": sum(1 for r in t_results if r["outcome"] == "passed"),
                    "failed": sum(1 for r in t_results if r["outcome"] == "failed"),
                    "skipped": sum(1 for r in t_results if r["outcome"] == "skipped")
                }

        # Feature breakdown
        feature_breakdown = {}
        for i in range(1, 12):
            feat_key = f"feature_{i}"
            f_results = [r for r in self.results if r["feature"] == feat_key]
            if f_results:
                feature_breakdown[feat_key] = {
                    "total": len(f_results),
                    "passed": sum(1 for r in f_results if r["outcome"] == "passed"),
                    "failed": sum(1 for r in f_results if r["outcome"] == "failed")
                }

        report_data = {
            "title": "EcoZone AI E2E Test Execution Summary",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "is_simulation": True,
                "grid_layout": "3 rows x 2 columns (6 zones)",
                "canvas_dimensions": "600x400 (cell 300x133)"
            },
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "duration_seconds": round(total_duration, 3)
            },
            "tier_breakdown": tier_breakdown,
            "feature_breakdown": feature_breakdown,
            "failures": [r for r in self.results if r["outcome"] == "failed"]
        }

        if self.output_path:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)
            print(f"\n{CYAN}[INFO]{RESET} Test execution JSON report saved to: {self.output_path}")


def print_banner(tier, feature, verbose):
    print(f"\n{BOLD}{CYAN}{'=' * 75}{RESET}")
    print(f"{BOLD}{CYAN}  EcoZone AI Smart Energy Management System - Modernized Test Runner{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 75}{RESET}")
    print(f"  {BOLD}Platform:{RESET} {platform.platform()} | Python {platform.python_version()}")
    print(f"  {BOLD}Layout:{RESET}   3 rows x 2 columns (6 zones: 300x133px cells in 600x400 canvas)")
    print(f"  {BOLD}Target:{RESET}   Tier: {tier.upper()} | Feature Filter: {feature or 'All (1-11)'}")
    print(f"  {BOLD}Mode:{RESET}     Simulation / Offline Mode")
    print(f"{BOLD}{CYAN}{'-' * 75}{RESET}\n")


def print_summary(exit_code, plugin):
    total = len(plugin.results)
    passed = sum(1 for r in plugin.results if r["outcome"] == "passed")
    failed = sum(1 for r in plugin.results if r["outcome"] == "failed")
    skipped = sum(1 for r in plugin.results if r["outcome"] == "skipped")
    elapsed = time.time() - plugin.start_time

    print(f"\n{BOLD}{CYAN}{'-' * 75}{RESET}")
    print(f"{BOLD}  TEST EXECUTION SUMMARY{RESET}")
    print(f"{BOLD}{CYAN}{'-' * 75}{RESET}")
    print(f"  Total Tests Executed: {BOLD}{total}{RESET}")
    print(f"  Passed:               {GREEN}{BOLD}{passed}{RESET}")
    if failed > 0:
        print(f"  Failed:               {RED}{BOLD}{failed}{RESET}")
    else:
        print(f"  Failed:               0")
    if skipped > 0:
        print(f"  Skipped:              {YELLOW}{skipped}{RESET}")
    print(f"  Elapsed Time:         {elapsed:.2f}s")
    print(f"{BOLD}{CYAN}{'=' * 75}{RESET}")

    if exit_code == 0:
        print(f"  {GREEN}{BOLD}RESULT: SUCCESS — 100% of all executed tests passed cleanly! (Exit 0){RESET}")
    else:
        print(f"  {RED}{BOLD}RESULT: FAILURE — Test run finished with exit code {exit_code}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 75}{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="EcoZone AI Test Runner CLI")
    parser.add_argument("--tier", type=str, choices=["1", "2", "3", "4", "all"], default="all", help="Target test tier to execute (1, 2, 3, 4, all)")
    parser.add_argument("--feature", type=int, choices=list(range(1, 12)), help="Target feature ID (1 to 11)")
    parser.add_argument("--json-output", type=str, help="Output file path for JSON test report summary")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose test output")
    args = parser.parse_args()

    print_banner(args.tier, args.feature, args.verbose)

    # Build pytest arguments
    tests_dir = os.path.join(PROJECT_ROOT, "tests")
    pytest_args = [tests_dir]

    if args.verbose:
        pytest_args.append("-v")

    markers = []
    if args.tier != "all":
        markers.append(f"tier{args.tier}")

    if args.feature:
        markers.append(f"feature({args.feature})")

    if markers:
        pytest_args.extend(["-m", " and ".join(markers)])

    plugin = JsonReportPlugin(output_path=args.json_output)
    exit_code = pytest.main(pytest_args, plugins=[plugin])

    print_summary(exit_code, plugin)

    # Clean up perspective_matrix.npy if created during test run
    matrix_path = os.path.join(PROJECT_ROOT, "perspective_matrix.npy")
    if os.path.exists(matrix_path):
        try:
            os.remove(matrix_path)
        except Exception:
            pass

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
