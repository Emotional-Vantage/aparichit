import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from unlearn_ai.benchmark.compare_all import run_benchmark

if __name__ == "__main__":
    run_benchmark()
