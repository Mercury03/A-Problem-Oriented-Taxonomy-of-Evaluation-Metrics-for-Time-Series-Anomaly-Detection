#!/bin/bash
# One-click script to run the complete experiment

set -e  # Exit immediately on error

echo "=========================================="
echo "One-click Complete Experiment Runner"
echo "=========================================="
echo ""

# Check parameters
MODE=${1:-"test"}
GPU_DEVICES=${2:-"0"}

# Configuration parameters
if [ "$MODE" = "test" ]; then
    echo "Mode: Test mode (quick validation)"
    N_DATASETS=4
    N_RUNS=2
    PARALLEL="False"
    OUTPUT_DIR="./test_results"
elif [ "$MODE" = "single" ]; then
    echo "Mode: Single run mode (quick experiment)"
    N_DATASETS=9
    N_RUNS=1
    PARALLEL="False"
    OUTPUT_DIR="./experiment_results"
elif [ "$MODE" = "multi" ]; then
    echo "Mode: Multi-run mode (robust experiment)"
    N_DATASETS=9
    N_RUNS=3
    PARALLEL="False"
    OUTPUT_DIR="./experiment_results_multi"
elif [ "$MODE" = "parallel" ]; then
    echo "Mode: Parallel training mode (multi-GPU acceleration)"
    N_DATASETS=9
    N_RUNS=2
    PARALLEL="True"
    OUTPUT_DIR="./experiment_results_parallel"
elif [ "$MODE" = "full" ]; then
    echo "Mode: Full mode (multi-run + parallel training)"
    N_DATASETS=12
    N_RUNS=5
    PARALLEL="True"
    OUTPUT_DIR="./experiment_results_full"
else
    echo "Error: Unknown mode '$MODE'"
    echo ""
    echo "Usage: $0 <mode> [gpu_devices]"
    echo ""
    echo "Modes:"
    echo "  test      - Test mode (2 datasets, 2 runs, no parallelism)"
    echo "  single    - Single run (9 datasets, 1 run, no parallelism) ~30 min"
    echo "  multi     - Multi-run (9 datasets, 3 runs, no parallelism) ~90 min"
    echo "  parallel  - Parallel training (9 datasets, 1 run, multi-GPU) ~12 min (4 GPU)"
    echo "  full      - Full mode (9 datasets, 5 runs, multi-GPU) ~60 min (4 GPU)"
    echo ""
    echo "Examples:"
    echo "  $0 test          # Quick test"
    echo "  $0 single        # Daily single experiment"
    echo "  $0 parallel 0,1,2,3  # Parallel training with 4 GPUs"
    echo "  $0 full 0,1,2,3  # Complete experiment (recommended for papers)"
    exit 1
fi

echo "Configuration:"
echo "  - Number of datasets: $N_DATASETS"
echo "  - Number of runs: $N_RUNS"
echo "  - Parallel training: $PARALLEL"
echo "  - GPU devices: $GPU_DEVICES"
echo "  - Output directory: $OUTPUT_DIR"
echo ""

# Get script directory (absolute path)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TSAD_EVAL_DIR="$SCRIPT_DIR/TSAD_eval_main"

# Create Python execution script
cat > /tmp/run_experiment_$$.py << EOF
import sys
import os
from pathlib import Path

# Use absolute path
tsad_eval_dir = Path('$TSAD_EVAL_DIR')

# Add to Python path
sys.path.insert(0, str(tsad_eval_dir))

# Change to TSAD_eval-main directory
os.chdir(str(tsad_eval_dir))

from exp import MetricRobustnessExperiment, ExperimentConfig

def main():
    # Configuration
    gpu_devices = [int(x) for x in "$GPU_DEVICES".split(',')]

    config = ExperimentConfig(
        n_datasets=$N_DATASETS,
        n_random_trials=9,
        n_experiment_runs=$N_RUNS,
        parallel_model_training=$PARALLEL,
        gpu_devices=gpu_devices,
        aggregate_runs="mean",
        output_dir="$OUTPUT_DIR",
        save_plots=True,
        save_data=True,
        enable_parallel=True,
        skip_slow_metrics=False,  # Do not skip slow metrics, including VUS
        disable_metrics=["pate_auc_small","pate_auc_default"]  # Only disable PATE metrics
    )

    print("\nStarting experiment...")
    experiment = MetricRobustnessExperiment(config)
    results = experiment.run_experiment()

    print("\nExperiment completed!")
    print(f"Results saved in: {config.output_dir}")
    return results

if __name__ == '__main__':
    main()
EOF

# Run experiment
echo "=========================================="
echo "Running experiment..."
echo "=========================================="
cd "$SCRIPT_DIR"
python /tmp/run_experiment_$$.py

# Clean up temporary files
rm /tmp/run_experiment_$$.py

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "=========================================="
echo ""
echo "View results:"
echo "  cd $OUTPUT_DIR/reports"
echo "  ls -lh *.png *.csv"
echo ""
echo "Regenerate charts:"
echo "  python TSAD_eval-main/plot_results.py $OUTPUT_DIR/experiment_data_*.pkl"
echo ""
