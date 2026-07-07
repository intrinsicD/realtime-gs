#!/usr/bin/env python3
"""Benchmark harness entry point — thin wrapper around rtgs.bench (see docs/BENCHMARKS.md)."""

import sys
from pathlib import Path

from rtgs.bench import main

if __name__ == "__main__":
    sys.exit(main(["--repo-root", str(Path(__file__).resolve().parent.parent), *sys.argv[1:]]))
