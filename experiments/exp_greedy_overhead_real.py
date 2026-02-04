#!/usr/bin/env python3
"""
Experiment 4 (Real Benchmark): Overhead Analysis of Algorithms on Real Benchmark Tasks
File: experiments/exp_greedy_overhead_real.py

Goal:
Directly execute Tasks 1-28 as defined in run_benchmark.py to measure the actual execution 
time of the Greedy Partitioning algorithm. This evaluates the scalability of the algorithm 
across different models, datasets, and power constraints.

Implementation:
- Directly invoke GreedyPartitioner (avoiding subprocess overhead).
- Run both 'standard' and 'vectorized' versions once for each task.
- Automatically iterate through tasks for both 3080Ti and A100 GPUs.
- Utilize the raw partitioning time returned by the run() method.

Task Distribution:
- 3080Ti: Tasks 1-14 (7 models × 2 datasets).
- A100: Tasks 15-28 (7 models × 2 datasets).

CSV Output Fields:
- task_id, gpu, model, dataset, power_cap
- standard_time_ms, vectorized_time_ms, speedup
"""

import os
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import sys
import csv
from pathlib import Path
from io import StringIO

current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from greedy_partitioning import GreedyPartitioner

# === Task Configuration (Extracted from run_benchmark.py) ===
BENCHMARK_TASKS = {
    "3080Ti": [
        # ID, DNN, Dataset, Power Cap (W)
        (1, "mnasnet", "imagenet", 120),
        (2, "densenet201", "imagenet", 180),
        (3, "efficientnet_v2_m", "imagenet", 140),
        (4, "maxvit", "imagenet", 170),
        (5, "mobilenet_v2", "imagenet", 120),
        (6, "resnet50", "imagenet", 130),
        (7, "vgg19", "imagenet", 155),
        (8, "mnasnet", "caltech256", 140),
        (9, "densenet201", "caltech256", 190),
        (10, "efficientnet_v2_m", "caltech256", 190),
        (11, "maxvit", "caltech256", 200),
        (12, "mobilenet_v2", "caltech256", 125),
        (13, "resnet50", "caltech256", 170),
        (14, "vgg19", "caltech256", 180),
    ],
    "A100": [
        # ID, DNN, Dataset, Power Cap (W)
        (15, "mnasnet", "imagenet", 70),
        (16, "densenet201", "imagenet", 200),
        (17, "efficientnet_v2_m", "imagenet", 180),
        (18, "maxvit", "imagenet", 200),
        (19, "mobilenet_v2", "imagenet", 100),
        (20, "resnet50", "imagenet", 150),
        (21, "vgg19", "imagenet", 130),
        (22, "mnasnet", "caltech256", 100),
        (23, "densenet201", "caltech256", 280),
        (24, "efficientnet_v2_m", "caltech256", 240),
        (25, "maxvit", "caltech256", 250),
        (26, "mobilenet_v2", "caltech256", 120),
        (27, "resnet50", "caltech256", 180),
        (28, "vgg19", "caltech256", 180),
    ],
}

# CSV 文件路径
CSV_FILE = current_dir / "overhead_benchmark.csv"

# CSV 字段定义
CSV_FIELDS = [
    "task_id", "gpu", "model", "dataset", "power_cap",
    "method",  # 'standard' or 'vectorized'
    "partition_time_ms", "configs_searched",
    "final_freq", "final_mem_freq", "num_tasks",
    "p_list", "b_list",
    "predicted_throughput", "actual_throughput", "throughput_error_percent"
]


def run_partition_once(model, dataset, power_cap, platform, vectorized=False):
    """
    Executes a single partitioning task and returns detailed results.

    Args:
        model: Model name
        dataset: Dataset name
        power_cap: Power limit constraint
        platform: Platform name (3080ti / a100)
        vectorized: Whether to use vectorized search

    Returns:
        dict: A dictionary containing partition results, or None if the task fails.
    """
    try:

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            partitioner = GreedyPartitioner(
                model_name=model,
                p_cap=power_cap,
                mps_step=5,
                platform=platform,
                dataset=dataset,
                vectorized=vectorized
            )

            final_tasks, final_freq, final_mem_freq, partition_time_ms, configs_searched = partitioner.run()

            p_list = [t.mps for t in final_tasks]
            b_list = [t.batch for t in final_tasks]

            predictor = partitioner.predictor
            if final_mem_freq != predictor.mem_freq:
                mem_key = float(final_mem_freq)
                if mem_key in partitioner._predictors:
                    predictor = partitioner._predictors[mem_key]

            _, _, predicted_throughput, _ = predictor.predict_all(p_list, b_list, final_freq, mem_freq=final_mem_freq)

            return {
                'partition_time_ms': partition_time_ms,
                'configs_searched': configs_searched,
                'final_freq': final_freq,
                'final_mem_freq': final_mem_freq,
                'num_tasks': len(final_tasks),
                'p_list': ','.join(map(str, p_list)),
                'b_list': ','.join(map(str, b_list)),
                'predicted_throughput': predicted_throughput,
            }
        finally:
            sys.stdout = old_stdout

    except Exception as e:
        sys.stdout = old_stdout
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_benchmark():
    """
    Runs all benchmark tasks, executing both 'standard' and 'vectorized' 
    methods once for each task. 
    
    Results for each method are recorded on a separate line and appended 
    to the CSV file in real-time.
    """
    print(f"\n{'='*120}")
    print(f"Real Benchmark Overhead Analysis (Tasks 1-28)")
    print(f"Running standard and vectorized once per task")
    print(f"Results will be saved incrementally to {CSV_FILE}")
    print(f"{'='*120}\n")

    # Initialize CSV file (write headers)
    csv_exists = CSV_FILE.exists()
    if not csv_exists:
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()

    total_tasks = sum(len(tasks) for tasks in BENCHMARK_TASKS.values())
    current_task = 0
    successful_tasks = 0

    for gpu in ["3080Ti", "A100"]:
        platform = "a100" if gpu == "A100" else "3080ti"
        tasks = BENCHMARK_TASKS[gpu]

        print(f"\n{'#'*120}")
        print(f"### GPU: {gpu} (Platform: {platform})")
        print(f"{'#'*120}")
        print(f"{'Task':<6} | {'Model':<18} | {'Dataset':<12} | {'Power(W)':<10} | {'Standard':<15} | {'Vectorized':<15} | {'Speedup':<10}")
        print(f"{'-'*120}")

        for task_id, model, dataset, power_cap in tasks:
            current_task += 1

            #  Standard Mode
            print(f"Task {task_id}: Running standard mode...", end=' ', flush=True)
            std_result = run_partition_once(model, dataset, power_cap, platform, vectorized=False)

            #  Vectorized Mode
            print(f"Running vectorized mode...", end=' ', flush=True)
            vec_result = run_partition_once(model, dataset, power_cap, platform, vectorized=True)

            if std_result is not None and vec_result is not None:
                std_time = std_result['partition_time_ms']
                vec_time = vec_result['partition_time_ms']
                speedup = std_time / vec_time if vec_time > 0 else 0

                # Writing standard results
                std_row = {
                    "task_id": task_id,
                    "gpu": gpu,
                    "model": model,
                    "dataset": dataset,
                    "power_cap": power_cap,
                    "method": "standard",
                    "partition_time_ms": std_time,
                    "configs_searched": std_result['configs_searched'],
                    "final_freq": std_result['final_freq'],
                    "final_mem_freq": std_result['final_mem_freq'],
                    "num_tasks": std_result['num_tasks'],
                    "p_list": std_result['p_list'],
                    "b_list": std_result['b_list'],
                    "predicted_throughput": std_result['predicted_throughput'],
                    "actual_throughput": "",
                    "throughput_error_percent": "",
                }

                # Writing vectorized results
                vec_row = {
                    "task_id": task_id,
                    "gpu": gpu,
                    "model": model,
                    "dataset": dataset,
                    "power_cap": power_cap,
                    "method": "vectorized",
                    "partition_time_ms": vec_time,
                    "configs_searched": vec_result['configs_searched'],
                    "final_freq": vec_result['final_freq'],
                    "final_mem_freq": vec_result['final_mem_freq'],
                    "num_tasks": vec_result['num_tasks'],
                    "p_list": vec_result['p_list'],
                    "b_list": vec_result['b_list'],
                    "predicted_throughput": vec_result['predicted_throughput'],
                    "actual_throughput": "",
                    "throughput_error_percent": "",
                }

                with open(CSV_FILE, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                    writer.writerow(std_row)
                    writer.writerow(vec_row)

                successful_tasks += 1

                std_str = f"{std_time:.2f} ms" if std_time < 1000 else f"{std_time/1000:.3f} s"
                vec_str = f"{vec_time:.2f} ms" if vec_time < 1000 else f"{vec_time/1000:.3f} s"
                speedup_str = f"{speedup:.2f}x"

                print(f"Done!")
                print(f"{task_id:<6} | {model:<18} | {dataset:<12} | {power_cap:<10.0f} | {std_str:<15} | {vec_str:<15} | {speedup_str:<10}")
                print(f"  Standard: Freq={std_result['final_freq']:.0f} MHz, Mem={std_result['final_mem_freq']:.0f} MHz, Tasks={std_result['num_tasks']}, Tput={std_result['predicted_throughput']:.2f} img/s")
                print(f"  Vectorized: Freq={vec_result['final_freq']:.0f} MHz, Mem={vec_result['final_mem_freq']:.0f} MHz, Tasks={vec_result['num_tasks']}, Tput={vec_result['predicted_throughput']:.2f} img/s")
            else:
                print(f"FAILED!")
                std_str = "FAILED" if std_result is None else f"{std_result['partition_time_ms']:.2f} ms"
                vec_str = "FAILED" if vec_result is None else f"{vec_result['partition_time_ms']:.2f} ms"
                speedup_str = "N/A"

                print(f"{task_id:<6} | {model:<18} | {dataset:<12} | {power_cap:<10.0f} | {std_str:<15} | {vec_str:<15} | {speedup_str:<10}")

            print(f"  [{current_task}/{total_tasks}] {current_task/total_tasks*100:.1f}% done\n")

    print(f"\n{'='*120}")
    print(f"Results saved to {CSV_FILE}")
    print(f"Successful tasks: {successful_tasks}/{total_tasks}")

    if successful_tasks > 0:
        import pandas as pd
        df = pd.read_csv(CSV_FILE)

        std_df = df[df['method'] == 'standard'].set_index('task_id')
        vec_df = df[df['method'] == 'vectorized'].set_index('task_id')

        if len(std_df) > 0 and len(vec_df) > 0:
            speedups = std_df['partition_time_ms'] / vec_df['partition_time_ms']
            avg_speedup = speedups.mean()
            max_speedup = speedups.max()
            min_speedup = speedups.min()
            avg_configs = std_df['configs_searched'].mean()

            print(f"\nSpeedup Statistics:")
            print(f"  - Average: {avg_speedup:.2f}x")
            print(f"  - Max: {max_speedup:.2f}x")
            print(f"  - Min: {min_speedup:.2f}x")
            print(f"\nConfigs Searched (per task avg): {avg_configs:.0f}")
    else:
        print(f"\nNo successful tasks to analyze")


if __name__ == "__main__":
    run_benchmark()
