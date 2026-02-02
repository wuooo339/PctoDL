#!/usr/bin/env python3
"""
Automated benchmark runner - batch tests across models, datasets, and power caps.

Benchmark tasks (from the paper table):
- 3080Ti: 14 tasks
- A100: 14 tasks
- Models: MNASNet, DenseNet201, EfficientNetV2, MaxVit, MobileNetV2, ResNet50, VGG19
- Datasets: ImageNet, Caltech
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import pynvml

# Benchmark task matrix
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
        # Power scheduling tasks (no fixed power_cap, use dynamic schedule)
        (29, "resnet50", "imagenet", None),  # Downshift: 200W → 160W → 120W
        (30, "efficientnet_v2_m", "imagenet", None),  # Upshift: 130W → 180W → 230W
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
        # Power scheduling tasks (no fixed power_cap, use dynamic schedule)
        (29, "resnet50", "imagenet", None),  # Downshift: 200W → 160W → 120W
        (30, "efficientnet_v2_m", "imagenet", None),  # Upshift: 130W → 180W → 230W
    ],
}

# GPU name mapping (for auto-detection)
GPU_NAME_MAPPING = {
    # 3080Ti variants
    "RTX 3080 Ti": "3080Ti",
    "RTX3080Ti": "3080Ti",
    "3080 Ti": "3080Ti",
    "3080Ti": "3080Ti",
    "GeForce RTX 3080 Ti": "3080Ti",
    # A100 variants
    "A100": "A100",
    "A100-SXM": "A100",
    "A100-PCIE": "A100",
    "NVIDIA A100": "A100",
}


def detect_gpu():
    """
    Auto-detect the GPU model on the current system.

    Returns:
        str: Detected GPU model ("3080Ti", "A100", or None)
    """
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        if device_count == 0:
            return None

        # Get the name of the first GPU
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)

        # Convert bytes to str if needed
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode('utf-8')

        print(f"Detected GPU: {gpu_name}")

        # Match known GPU names
        for known_name, standard_name in GPU_NAME_MAPPING.items():
            if known_name.lower() in gpu_name.lower():
                print(f"✓ Identified as: {standard_name}")
                return standard_name

        # Unknown GPU
        print(f"⚠ Unknown GPU model. Add a mapping in the script: {gpu_name}")
        print("   Use --gpu to specify the model manually")
        return None

    except Exception as e:
        print(f"GPU detection failed: {e}")
        print("   Use --gpu to specify the model manually")
        return None


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run PctoDL benchmarks in batch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect GPU and run the matching task set (recommended)
  python run_benchmark.py

  # Run all 3080Ti tasks (single algorithm)
  python run_benchmark.py --gpu 3080Ti --algorithm pctodl

  # Run all algorithms sequentially
  python run_benchmark.py --gpu 3080Ti --algorithm all

  # Run two algorithms (comma-separated)
  python run_benchmark.py --gpu 3080Ti --algorithm pctodl,morak

  # Run specific A100 tasks
  python run_benchmark.py --gpu A100 --task-id 20 21 22

  # Resume from a given task ID
  python run_benchmark.py --gpu 3080Ti --start-from 5

  # Run Morak
  python run_benchmark.py --gpu A100 --algorithm morak

  # Run powercap (uses nvidia-smi -pl to control power)
  python run_benchmark.py --gpu A100 --algorithm powercap

  # Detect GPU only (no benchmark)
  python run_benchmark.py --detect-gpu

Supported GPUs:
  - 3080Ti: task IDs 1-14 (fixed power), 29-30 (dynamic power schedule)
  - A100: task IDs 15-28 (power range: 70-280W)
        """
    )
    parser.add_argument(
        "--gpu",
        type=str,
        required=False,
        choices=["3080Ti", "A100"],
        help="Target GPU model (auto-detect if omitted)"
    )
    parser.add_argument(
        "--detect-gpu",
        action="store_true",
        help="Detect GPU model only, do not run benchmarks"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Scheduling algorithm: pctodl, batchdvfs, morak, powercap. Use comma-separated values or 'all'"
    )
    parser.add_argument(
        "--task-id",
        type=int,
        nargs="+",
        help="Task IDs to run (omit to run all)"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        help="Start from a specific task ID (resume)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Iterations per task (default: 5)"
    )
    parser.add_argument(
        "--power-margin",
        type=float,
        default=0.0,
        help="Power margin (W) for greedy partitioning. Greedy uses (power_cap - margin) (default: 0)"
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        help="Platform name (for predictor)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Dataset root path (default: ./data)"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="./benchmark_results",
        help="Output directory (default: ./benchmark_results)"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel tasks (default: 1)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default="--disable-shadow",
        help="Extra args forwarded to main.py (default: '--disable-shadow')"
    )
    parser.add_argument(
        "--enable-shadow",
        action="store_true",
        help="Enable shadow optimizer (overrides default --disable-shadow)"
    )
    parser.add_argument(
        "--no-greedy-vectorized",
        action="store_true",
        help="Disable vectorized greedy partitioning (override default on)"
    )
    parser.add_argument(
        "--save-summary",
        action="store_true",
        help="Save a summary report to file (default: off)"
    )
    parser.add_argument(
        "--interleave",
        action="store_true",
        help="Interleaved mode: run all algorithms per task, instead of all tasks per algorithm"
    )
    return parser.parse_args()


def get_tasks_to_run(args):
    """Return the task list to run based on CLI arguments."""
    all_tasks = BENCHMARK_TASKS[args.gpu]

    if args.task_id:
        # Filter by task ID
        task_map = {t[0]: t for t in all_tasks}
        selected = []
        for tid in args.task_id:
            if tid in task_map:
                selected.append(task_map[tid])
            else:
                print(f"Warning: task ID {tid} is not defined for {args.gpu}")
        return selected

    if args.start_from:
        # Start from a given task ID
        for task in all_tasks:
            if task[0] >= args.start_from:
                return all_tasks[all_tasks.index(task):]

    # Return all tasks
    return all_tasks


def build_command(args, task, log_dir):
    """Build the command to run a single task."""
    task_id, model, dataset, power_cap = task

    # Build CSV path (no timestamp)
    # For power scheduling tasks (power_cap=None), use a special suffix
    if power_cap is None:
        # Determine schedule type by task ID
        if task_id == 29:  # ResNet50 downshift
            schedule_type = "down"
            csv_path = Path(log_dir) / f"{task_id}_{model}_{dataset}_power_down.csv"
        elif task_id == 30:  # EfficientNetV2 upshift
            schedule_type = "up"
            csv_path = Path(log_dir) / f"{task_id}_{model}_{dataset}_power_up.csv"
        else:
            schedule_type = None
            csv_path = Path(log_dir) / f"{task_id}_{model}_{dataset}_dynamic.csv"
    else:
        schedule_type = None
        csv_path = Path(log_dir) / f"{task_id}_{model}_{dataset}_{power_cap}W.csv"

    # Power scheduling tasks use 150 iterations automatically
    iterations = args.iterations
    if power_cap is None:
        iterations = 150
        print(f"[PowerScheduling] Task #{task_id} uses {iterations} iterations (needed for power transitions)")

    cmd = [
        sys.executable,
        "src/main.py",
        "--algorithm", args.algorithm,
        "--model", model,
        "--dataset", dataset,
        "--iterations", str(iterations),
        "--log-file", str(csv_path),
    ]

    # Only fixed-power tasks add --power-cap
    if power_cap is not None:
        cmd.extend(["--power-cap", str(power_cap)])

    # Dynamic power scheduling tasks add --power-schedule
    if schedule_type is not None:
        cmd.extend(["--power-schedule", schedule_type])

    if args.platform:
        cmd.extend(["--platform", args.platform])

    # Add power margin (if specified)
    if args.power_margin > 0:
        cmd.extend(["--power-margin", str(args.power_margin)])
    if not args.no_greedy_vectorized:
        cmd.append("--greedy-vectorized")

    # DirectCap: for A100, try loading configs from pctodl results
    if args.algorithm == "directcap" and args.gpu.upper() == "A100":
        pctodl_config = _load_pctodl_config(args.result_dir, args.gpu, task_id, model, dataset, power_cap)
        if pctodl_config:
            mps_resources, batch_sizes = pctodl_config
            cmd.extend(["--mps-resources", mps_resources])
            cmd.extend(["--batch-sizes", batch_sizes])
            print(f"[DirectCap] Loaded config from pctodl results: mps={mps_resources}, batch={batch_sizes}")

    extra_args = args.extra_args
    if args.enable_shadow and extra_args:
        tokens = [t for t in extra_args.split() if t != "--disable-shadow"]
        extra_args = " ".join(tokens)
    if extra_args:
        cmd.extend(extra_args.split())

    return cmd, task_id, model, dataset, power_cap, csv_path


def _load_pctodl_config(result_dir, gpu, task_id, model, dataset, power_cap):
    """
    Read mps_resources and batch_sizes from the first row of a pctodl result CSV.

    Returns: (mps_resources_str, batch_sizes_str) or None
    """
    # Build pctodl result file path
    pctodl_dir = Path(result_dir) / f"{gpu}_pctodl"

    # Determine filename based on power_cap
    if power_cap is None:
        if task_id == 29:
            csv_filename = f"{task_id}_{model}_{dataset}_power_down.csv"
        elif task_id == 30:
            csv_filename = f"{task_id}_{model}_{dataset}_power_up.csv"
        else:
            csv_filename = f"{task_id}_{model}_{dataset}_dynamic.csv"
    else:
        csv_filename = f"{task_id}_{model}_{dataset}_{power_cap}W.csv"

    pctodl_csv_path = pctodl_dir / csv_filename

    if not pctodl_csv_path.exists():
        print(f"[DirectCap] pctodl result file not found: {pctodl_csv_path}")
        return None

    try:
        with open(pctodl_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            first_row = next(reader, None)
            if first_row and 'mps_resources' in first_row and 'batch_sizes' in first_row:
                mps_resources = first_row['mps_resources']
                batch_sizes = first_row['batch_sizes']
                return mps_resources, batch_sizes
            else:
                print(f"[DirectCap] pctodl result file is invalid or empty: {pctodl_csv_path}")
                return None
    except Exception as e:
        print(f"[DirectCap] Failed to read pctodl result file: {e}")
        return None


def run_single_task(cmd, task_id, model, dataset, power_cap, csv_path, dry_run=False):
    """Run a single benchmark task."""
    print("\n" + "=" * 80)
    print(f"Task #{task_id}: {model} + {dataset} @ {power_cap}W")
    print(f"Result file: {csv_path}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)

    if dry_run:
        print(f"[DRY RUN] {' '.join(cmd)}")
        return True, 0.0

    start_time = time.time()
    success = False

    try:
        # Run directly; let main.py handle logging
        process = subprocess.Popen(cmd)

        # Stream output in real time
        while True:
            try:
                process.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                continue

        success = (process.returncode == 0)

    except KeyboardInterrupt:
        print(f"\n\n[Interrupted] Task #{task_id} interrupted by user")
        process.terminate()
        raise  # Re-raise so the main loop can stop all tasks
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    elapsed = time.time() - start_time

    if success:
        print(f"\n✓ Task #{task_id} completed (elapsed: {elapsed:.1f}s)")
    else:
        print(f"\n✗ Task #{task_id} failed (elapsed: {elapsed:.1f}s)")

    return success, elapsed


def generate_summary(results, args):
    """Generate a simple summary of task results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = Path(args.result_dir) / f"{args.gpu}_{args.algorithm}" / f"summary_{timestamp}.txt"

    # Ensure directory exists
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"PctoDL Benchmark Summary\n")
        f.write(f"GPU: {args.gpu}\n")
        f.write(f"Algorithm: {args.algorithm}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")

        # Stats
        total = len(results)
        success_count = sum(1 for r in results if r["success"])
        total_time = sum(r["elapsed"] for r in results)

        f.write(f"Total Tasks: {total}\n")
        f.write(f"Success: {success_count}\n")
        f.write(f"Failed: {total - success_count}\n")
        f.write(f"Total Time: {total_time/3600:.2f} hours\n\n")

        # Task list (CSV path for each task)
        f.write("Task Results:\n")
        f.write("-" * 80 + "\n")
        for r in results:
            status = "SUCCESS" if r["success"] else "FAILED"
            if r["power_cap"] is None:
                if r["task_id"] == 29:
                    csv_name = f"{r['task_id']}_{r['model']}_{r['dataset']}_power_down.csv"
                elif r["task_id"] == 30:
                    csv_name = f"{r['task_id']}_{r['model']}_{r['dataset']}_power_up.csv"
                else:
                    csv_name = f"{r['task_id']}_{r['model']}_{r['dataset']}_dynamic.csv"
                power_str = "Dynamic"
            else:
                csv_name = f"{r['task_id']}_{r['model']}_{r['dataset']}_{r['power_cap']}W.csv"
                power_str = f"{r['power_cap']}W"
            f.write(f"Task {r['task_id']}: {r['model']} + {r['dataset']} @ {power_str} - {status}\n")
            f.write(f"  CSV: {csv_name}\n")
            f.write(f"  Time: {r['elapsed']:.1f}s\n\n")

    print(f"\nSummary saved to: {summary_file}")
    return summary_file


def main():
    args = parse_arguments()

    # Detect-only mode
    if args.detect_gpu:
        print("\nDetecting GPU model...")
        detected_gpu = detect_gpu()
        if detected_gpu:
            print(f"\n✓ Detected: {detected_gpu}")
            print(f"   Task ID range: {BENCHMARK_TASKS[detected_gpu][0][0]} - {BENCHMARK_TASKS[detected_gpu][-1][0]}")
            return 0
        else:
            print("\n✗ Detection failed: unsupported GPU")
            print("   Supported GPUs: 3080Ti, A100")
            return 1

    # Auto-detect GPU if not specified
    if args.gpu is None:
        print("\nDetecting GPU model...")
        detected_gpu = detect_gpu()
        if detected_gpu is None:
            print("\n✗ Error: no supported GPU detected")
            print("   This script supports only 3080Ti and A100")
            print("   If your GPU matches, use --gpu to specify it")
            print("   Or run --detect-gpu for details")
            return 1
        args.gpu = detected_gpu
        print(f"✓ Auto-selected GPU: {args.gpu}\n")
    if args.platform is None:
        if args.gpu == "A100":
            args.platform = "a100"
        elif args.gpu == "3080Ti":
            args.platform = "3080ti"

    # Choose algorithm (if not specified)
    algorithms_to_run = []
    if args.algorithm is None:
        print("\nSelect scheduling algorithm:")
        print("  1) pctodl")
        print("  2) batchdvfs")
        print("  3) morak")
        print("  4) directcap (NVIDIA direct power control + greedy partitioning)")
        print("  5) pctodl,directcap (dual run: PctoDL + DirectCap)")
        print("  6) morak,batchdvfs (dual run: Morak + Batchdvfs)")
        print("  7) all (run all algorithms sequentially)")
        while True:
            choice = input("Enter choice (1/2/3/4/5/6/7): ").strip()
            if choice == "1":
                args.algorithm = "pctodl"
                algorithms_to_run = ["pctodl"]
                break
            elif choice == "2":
                args.algorithm = "batchdvfs"
                algorithms_to_run = ["batchdvfs"]
                break
            elif choice == "3":
                args.algorithm = "morak"
                algorithms_to_run = ["morak"]
                break
            elif choice == "4":
                args.algorithm = "directcap"
                algorithms_to_run = ["directcap"]
                break
            elif choice == "5":
                args.algorithm = "pctodl,directcap"
                algorithms_to_run = ["pctodl", "directcap"]
                break
            elif choice == "6":
                args.algorithm = "morak,batchdvfs"
                algorithms_to_run = ["morak", "batchdvfs"]
                break
            elif choice == "7":
                args.algorithm = "all"
                algorithms_to_run = ["pctodl", "batchdvfs", "morak", "directcap"]
                break
            else:
                print("Invalid choice. Enter 1, 2, 3, 4, 5, 6, or 7")

        print(f"✓ Selected algorithm: {args.algorithm}\n")
    elif args.algorithm == "all":
        algorithms_to_run = ["pctodl", "batchdvfs", "morak", "directcap"]
    elif "," in args.algorithm:
        # Allow comma-separated algorithms like "pctodl,morak"
        algorithms_to_run = [algo.strip() for algo in args.algorithm.split(",")]
        # Validate algorithms
        valid_algorithms = {"pctodl", "batchdvfs", "morak", "directcap"}
        invalid = [a for a in algorithms_to_run if a not in valid_algorithms]
        if invalid:
            print(f"Error: invalid algorithm name(s): {invalid}")
            print(f"Supported algorithms: {list(valid_algorithms)}")
            return 1
    else:
        algorithms_to_run = [args.algorithm]

    # Get tasks to run
    tasks = get_tasks_to_run(args)

    if not tasks:
        print("Error: no tasks found to run")
        return 1

    print(f"\n{'='*80}")
    print("PctoDL Benchmark Runner")
    print(f"{'='*80}")
    print(f"GPU: {args.gpu}")
    print(f"Algorithms: {algorithms_to_run}")
    print(f"Task count: {len(tasks)}")
    print(f"Iterations per task: {args.iterations}")
    print(f"Parallel tasks: {args.parallel}")
    print(f"{'='*80}\n")

    # Show tasks to run
    print("Tasks to run:")
    print(f"{'ID':<6} {'Model':<20} {'Dataset':<15} {'Power(W)':<12}")
    print("-" * 60)
    for task_id, model, dataset, power_cap in tasks:
        power_str = f"{power_cap}" if power_cap is not None else "Dynamic"
        print(f"{task_id:<6} {model:<20} {dataset:<15} {power_str:<12}")

    if not args.dry_run:
        print("\nChoose run mode:")
        print("  [Enter/Y] Run all tasks")
        print("  [N] Cancel")
        print("  [1,3,5] Run specific task IDs (comma-separated)")
        print("  [1-5] Run a range of task IDs")
        print("  [1-7,29] Mix ranges and single IDs (comma-separated)")

        response = input("\nEnter selection: ").strip().lower()

        if response == "n":
            print("Cancelled")
            return 0

        # Parse selected task IDs
        selected_task_ids = set()
        original_tasks = tasks  # Keep the original list

        if response == "" or response == "y":
            # Run all tasks
            selected_task_ids = {task_id for task_id, *_ in tasks}
        else:
            # Parse input IDs
            try:
                # Comma-separated parts: single ID or range (e.g., "1-7,29" or "1-5,8,10-12")
                parts = response.replace(",", " ").split()
                for part in parts:
                    if not part.strip():
                        continue
                    if "-" in part:
                        # Parse range (e.g., "1-7")
                        range_parts = part.split("-")
                        if len(range_parts) == 2:
                            start_id, end_id = int(range_parts[0]), int(range_parts[1])
                            selected_task_ids.update(range(start_id, end_id + 1))
                    else:
                        # Single ID
                        selected_task_ids.add(int(part))
            except ValueError:
                print("Invalid input. Running all tasks")
                selected_task_ids = {task_id for task_id, *_ in tasks}

        # Filter task list
        if selected_task_ids:
            tasks = [t for t in tasks if t[0] in selected_task_ids]
            if not tasks:
                print("Error: no matching task IDs")
                return 1
            print(f"\nSelected {len(tasks)} task(s): {sorted(selected_task_ids)}")
        else:
            print("Invalid input. Running all tasks")
            tasks = original_tasks

    # Store all results
    all_results = {}

    # Choose loop style based on interleaving
    if args.interleave:
        # Interleaved mode: run all algorithms per task
        print(f"\n{'='*80}")
        print("Interleaved mode: run all algorithms per task")
        print(f"{'='*80}\n")

        # Initialize results per algorithm
        for algo in algorithms_to_run:
            all_results[algo] = []

        try:
            for i, task in enumerate(tasks, 1):
                task_id, model, dataset, power_cap = task
                print(f"\n{'#'*80}")
                print(f"### [{i}/{len(tasks)}] Task ID: {task_id} - {model} on {dataset}")
                print(f"{'#'*80}")

                for algo in algorithms_to_run:
                    args.algorithm = algo
                    print(f"\n--- Algorithm: {algo} ---")

                    # Create the result directory for this algorithm
                    log_dir = Path(args.result_dir) / f"{args.gpu}_{algo}"
                    log_dir.mkdir(parents=True, exist_ok=True)

                    cmd, task_id, model, dataset, power_cap, csv_path = build_command(args, task, log_dir)

                    success, elapsed = run_single_task(
                        cmd, task_id, model, dataset, power_cap, csv_path,
                        args.dry_run
                    )

                    all_results[algo].append({
                        "task_id": task_id,
                        "model": model,
                        "dataset": dataset,
                        "power_cap": power_cap,
                        "success": success,
                        "elapsed": elapsed,
                    })

                # Save intermediate summaries every few tasks
                if not args.dry_run and args.save_summary and i % 5 == 0:
                    for algo in algorithms_to_run:
                        generate_summary(all_results[algo], args)

        except KeyboardInterrupt:
            print("\n\n[Interrupted] Benchmark interrupted by user")
            # Save completed summaries
            if not args.dry_run and args.save_summary:
                for algo in algorithms_to_run:
                    if all_results[algo]:
                        print(f"Saving summary for {algo}...")
                        generate_summary(all_results[algo], args)
            print("Use --start-from to resume from the next task")
            return 130  # 128 + SIGINT

        # Final summary per algorithm
        for algo in algorithms_to_run:
            args.algorithm = algo
            if not args.dry_run and args.save_summary:
                summary_file = generate_summary(all_results[algo], args)
                success_count = sum(1 for r in all_results[algo] if r["success"])
                print(f"\nAlgorithm {algo} complete! Success: {success_count}/{len(all_results[algo])}")
                print(f"Summary report: {summary_file}")
            elif not args.dry_run:
                success_count = sum(1 for r in all_results[algo] if r["success"])
                print(f"\nAlgorithm {algo} complete! Success: {success_count}/{len(all_results[algo])}")

    else:
        # Default mode: run each algorithm sequentially
        for algo in algorithms_to_run:
            args.algorithm = algo
            print(f"\n{'#'*80}")
            print(f"### Starting algorithm: {algo}")
            print(f"{'#'*80}")

            # Create the result directory for this algorithm
            log_dir = Path(args.result_dir) / f"{args.gpu}_{algo}"
            log_dir.mkdir(parents=True, exist_ok=True)

            # Run all tasks
            results = []

            try:
                for i, task in enumerate(tasks, 1):
                    cmd, task_id, model, dataset, power_cap, csv_path = build_command(args, task, log_dir)

                    print(f"\nProgress: [{i}/{len(tasks)}]", end="")

                    success, elapsed = run_single_task(
                        cmd, task_id, model, dataset, power_cap, csv_path,
                        args.dry_run
                    )

                    results.append({
                        "task_id": task_id,
                        "model": model,
                        "dataset": dataset,
                        "power_cap": power_cap,
                        "success": success,
                        "elapsed": elapsed,
                    })

                    # Save intermediate summaries (resume support)
                    if not args.dry_run and args.save_summary and i % 5 == 0:
                        generate_summary(results, args)
            except KeyboardInterrupt:
                print("\n\n[Interrupted] Benchmark interrupted by user")
                # Save completed summaries
                if results and not args.dry_run and args.save_summary:
                    print(f"Completed {len(results)} task(s). Saving summary...")
                    generate_summary(results, args)
                print("Use --start-from to resume from the next task")
                return 130  # 128 + SIGINT

            # Save results for this algorithm
            all_results[algo] = results

            # Final summary for this algorithm
            if not args.dry_run and args.save_summary:
                summary_file = generate_summary(results, args)

                # Print statistics for this algorithm
                success_count = sum(1 for r in results if r["success"])
                print(f"\nAlgorithm {algo} complete! Success: {success_count}/{len(results)}")
                print(f"Summary report: {summary_file}")
            elif not args.dry_run:
                success_count = sum(1 for r in results if r["success"])
                print(f"\nAlgorithm {algo} complete! Success: {success_count}/{len(results)}")

    # After all algorithms, print a cross-algorithm summary
    if not args.dry_run and len(algorithms_to_run) > 1:
        print(f"\n{'='*80}")
        print("All algorithms completed!")
        print(f"{'='*80}")

        # Comparison table
        print("\nAlgorithm Summary:")
        print(f"{'Algorithm':<15} {'Success/Total':<15} {'Total Time(h)':<12}")
        print("-" * 45)
        for algo in algorithms_to_run:
            if algo in all_results:
                results = all_results[algo]
                success_count = sum(1 for r in results if r["success"])
                total_time = sum(r["elapsed"] for r in results)
                print(f"{algo:<15} {success_count}/{len(results):<15} {total_time/3600:<12.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
