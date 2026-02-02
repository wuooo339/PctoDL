#!/usr/bin/env python3
"""
背景任务运行器 - 基于main.py的可靠实现
用于单独运行背景任务并准确测量GPU资源利用率
"""
import os
import sys
import time
import threading
import json
import subprocess
import numpy as np
import pynvml
from concurrent.futures import ThreadPoolExecutor

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataLoader
import doInference
import doProfile
import globalConfig


SUDO_PASSWORD = os.environ.get('SUDO_PASSWORD', 'wzk123456')

# 全局变量用于管理背景任务
_background_threads = []
_background_stop_events = []


def run_continuous_background_task(model_name, batch_size, stop_event, task_id=None, is_background=True):
    """
    持续运行的任务（可以是背景任务或当前任务）

    Args:
        model_name: 模型名称
        batch_size: 批次大小
        stop_event: 停止事件
        task_id: 任务ID（用于日志）
        is_background: 是否为背景任务（True=背景任务，False=当前测试任务）
    """
    if is_background:
        task_name = f"背景任务-{task_id}" if task_id else "背景任务"
    else:
        task_name = f"当前任务-{task_id}" if task_id else "当前任务"

    # 加载配置
    try:
        with open('./config/platform.json', 'r') as f:
            platform_raw = json.load(f)
        # 适配新格式: platform_raw["platforms"][0]
        if "platforms" in platform_raw:
            platform_data = platform_raw["platforms"][0]
        else:
            platform_data = platform_raw

        config_path = f'./config/{model_name}/{model_name}.json'
        if not os.path.exists(config_path):
            config_path = f'./config/{model_name}/{model_name.split("_")[0]}.json'

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"找不到配置文件: {config_path}")

        with open(config_path, 'r') as f:
            model_data = json.load(f)
            configs = globalConfig.GlobalConfig(platform_data, model_data)
    except Exception as e:
        print(f"[{task_name}] 配置加载失败: {e}")
        return

    # 准备数据
    try:
        data = dataLoader.load_images_to_batches(configs.model.data_path, batch_size)

        model_data = {
            "model_name": model_name,
            "data": data,
            "batchsize": batch_size,
            "resource": 100
        }

        # 预热
        for _ in range(2):
            doInference.inference(model_data)

        print(f"[{task_name}] 预热完成，开始持续推理 (Batch={batch_size})")

    except Exception as e:
        print(f"[{task_name}] 预热失败: {e}")
        return

    # 持续推理循环
    inference_count = 0
    while not stop_event.is_set():
        try:
            latency, throughput = doInference.inference(model_data)
            inference_count += 1

            # 每10次推理输出一次状态
            if inference_count % 10 == 0:
                print(f"[{task_name}] 持续运行中... 已完成{inference_count}次推理，吞吐量: {throughput:.1f} samples/s")

        except Exception as e:
            print(f"[{task_name}] 推理出错: {e}")
            time.sleep(0.1)  # 出错时短暂暂停

    print(f"[{task_name}] 收到停止信号，已完成{inference_count}次推理")


def start_current_task_continuous(model_name, batch_size, actual_mps):
    """
    启动当前任务的持续运行版本，用于显示实时性能

    Args:
        model_name: 模型名称
        batch_size: 批次大小
        actual_mps: 实际MPS百分比

    Returns:
        threading.Event: 停止事件
    """
    stop_event = threading.Event()

    # 创建当前任务的持续运行版本
    task_config = {
        'model': model_name,
        'batch': batch_size,
        'mps': actual_mps
    }

    print(f"\n[当前任务持续运行] 启动 {model_name} 的持续监控")
    print("-" * 50)

    # 创建并启动线程
    thread = threading.Thread(
        target=run_continuous_background_task,
        args=(model_name, batch_size, stop_event, len(_background_threads)+1, False),  # is_background=False
        daemon=True
    )

    thread.start()
    _background_threads.append(thread)
    _background_stop_events.append(stop_event)

    print(f"  ✓ 启动当前任务监控: {model_name}, batch={batch_size}, mps={actual_mps}%")
    print("-" * 50)
    print("当前任务持续监控已启动")

    return stop_event


def start_background_tasks(background_tasks, is_first_task=False):
    """
    启动持续运行的任务

    Args:
        background_tasks: 任务列表，每个是 {'model': str, 'batch': int, 'mps': int}
        is_first_task: 是否为第一个任务（False=背景任务，True=当前任务）

    Returns:
        list: stop_events 列表，用于后续停止任务
    """
    global _background_threads, _background_stop_events

    if not background_tasks:
        return []

    # 确保MPS运行
    if not is_mps_running():
        if not start_mps_simple():
            print("错误: 无法启动MPS")
            return []

    if is_first_task:
        print(f"\n[当前任务启动] 启动当前任务进行持续推理")
    else:
        print(f"\n[背景任务启动] 启动 {len(background_tasks)} 个持续运行的背景任务")
    print("-" * 60)

    stop_events = []

    for i, task in enumerate(background_tasks):
        stop_event = threading.Event()
        stop_events.append(stop_event)

        # 创建并启动线程
        thread = threading.Thread(
            target=run_continuous_background_task,
            args=(task['model'], task['batch'], stop_event, i+1, not is_first_task),  # 传递is_background参数
            daemon=True  # 设置为守护线程，主程序退出时自动结束
        )

        thread.start()
        _background_threads.append(thread)
        _background_stop_events.append(stop_event)

        if is_first_task:
            print(f"  ✓ 启动当前任务: {task['model']}, batch={task['batch']}, mps={task['mps']}%")
        else:
            print(f"  ✓ 启动背景任务 {i+1}: {task['model']}, batch={task['batch']}, mps={task['mps']}%")

    print("-" * 60)
    if is_first_task:
        print("当前任务已启动并持续运行")
    else:
        print("所有背景任务已启动并持续运行")

    return stop_events


def stop_background_tasks(stop_events=None):
    """
    停止所有持续运行的背景任务

    Args:
        stop_events: stop_events 列表，如果为None则停止全局背景任务
    """
    global _background_threads, _background_stop_events

    if stop_events is None:
        stop_events = _background_stop_events

    if not stop_events:
        return

    print(f"\n[背景任务停止] 停止 {len(stop_events)} 个背景任务...")

    # 发送停止信号
    for stop_event in stop_events:
        stop_event.set()

    # 等待线程结束
    for thread in _background_threads:
        if thread.is_alive():
            thread.join(timeout=2.0)  # 最多等待2秒

    # 清理全局列表
    if stop_events == _background_stop_events:
        _background_threads.clear()
        _background_stop_events.clear()

    print("所有背景任务已停止")


def test_current_task_with_background(current_model, current_batch, background_stop_events):
    """
    在背景任务持续运行时测试当前任务的性能

    Args:
        current_model: 当前任务模型名称
        current_batch: 当前任务批次大小
        background_stop_events: 背景任务的停止事件列表

    Returns:
        dict: 测试结果
    """
    print(f"\n[当前任务测试] 在背景任务运行时测试当前任务")
    print(f"当前任务: {current_model}, Batch={current_batch}")
    print("-" * 50)

    # 确保背景任务仍在运行
    if not background_stop_events:
        print("警告: 没有活跃的背景任务")
        return None

    # 检查背景任务状态
    active_bg_tasks = sum(1 for event in background_stop_events if not event.is_set())
    print(f"活跃背景任务数: {active_bg_tasks}")

    # 加载当前任务配置
    try:
        with open('./config/platform.json', 'r') as f:
            platform_raw = json.load(f)
        # 适配新格式: platform_raw["platforms"][0]
        if "platforms" in platform_raw:
            platform_data = platform_raw["platforms"][0]
        else:
            platform_data = platform_raw

        config_path = f'./config/{current_model}/{current_model}.json'
        if not os.path.exists(config_path):
            config_path = f'./config/{current_model}/{current_model.split("_")[0]}.json'

        with open(config_path, 'r') as f:
            model_data = json.load(f)
            configs = globalConfig.GlobalConfig(platform_data, model_data)
    except Exception as e:
        print(f"错误: 加载当前任务配置失败 - {e}")
        return None

    # 准备当前任务数据
    try:
        data = dataLoader.load_images_to_batches(configs.model.data_path, current_batch)
        current_model_data = {
            "model_name": current_model,
            "data": data,
            "batchsize": current_batch,
            "resource": 100
        }

        # 预热当前任务
        print("预热当前任务...")
        for _ in range(2):
            doInference.inference(current_model_data)

        # 快速测试：运行1次获得准确吞吐量
        print(f"  快速性能测试...", end=" ")
        _, throughput = doInference.inference(current_model_data)
        print(f"吞吐量: {throughput:.1f} samples/s")

        print(f"\n✓ 当前任务测试完成!")
        print(f"  吞吐量: {throughput:.1f} samples/s")

        return {
            'throughput': throughput,
            'model_name': current_model,
            'batch_size': current_batch,
            'background_tasks': active_bg_tasks
        }

    except Exception as e:
        print(f"当前任务测试失败: {e}")
        return None


def start_mps_simple():
    """简单的MPS启动函数 - 复制main.py的方法"""
    print("\n启动MPS服务...")
    try:
        # 复制main.py的start_mps.sh方法
        result = subprocess.run(['sudo', '-S', './start_mps.sh'],
                              input=SUDO_PASSWORD + '\n',
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              text=True)
        if result.returncode == 0:
            print("✓ MPS服务启动成功")
            time.sleep(2)  # 等待服务完全启动
            return True
        else:
            print(f"✗ MPS启动失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ MPS启动异常: {e}")
        return False

def is_mps_running():
    """检查MPS是否运行"""
    try:
        result1 = subprocess.run(['pgrep', '-x', 'nvidia-cuda-mps-control'],
                               capture_output=True)
        result2 = subprocess.run(['pgrep', '-x', 'nvidia-cuda-mps-server'],
                               capture_output=True)
        return result1.returncode == 0 and result2.returncode == 0
    except:
        return False

def run_single_background_task(model_name, batch_size):
    """
    运行单个背景任务 - 复制main.py的推理逻辑

    Args:
        model_name: 模型名称
        batch_size: 批次大小

    Returns:
        dict: 任务执行结果
    """
    print(f"\n[背景任务] 运行模型: {model_name}, batch_size={batch_size}")
    print("-" * 50)

    # 加载配置
    try:
        with open('./config/platform.json', 'r') as f:
            platform_raw = json.load(f)
        # 适配新格式: platform_raw["platforms"][0]
        if "platforms" in platform_raw:
            platform_data = platform_raw["platforms"][0]
        else:
            platform_data = platform_raw

        config_path = f'./config/{model_name}/{model_name}.json'
        if not os.path.exists(config_path):
            config_path = f'./config/{model_name}/{model_name.split("_")[0]}.json'

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"找不到配置文件: {config_path}")

        with open(config_path, 'r') as f:
            model_data = json.load(f)
            configs = globalConfig.GlobalConfig(platform_data, model_data)
    except Exception as e:
        print(f"错误: 加载配置失败 - {e}")
        return None

    data = dataLoader.load_images_to_batches(configs.model.data_path, batch_size)

    model_data = {
        "model_name": model_name,
        "client": None,
        "data": data,
        "resource": None,
        "batchsize": batch_size
    }

    print("  开始推理...")
    try:
        start_time = time.time()
        latency, throughput = doInference.inference(model_data)
        elapsed = time.time() - start_time
        print(f"  ✓ 完成! 延迟: {latency:.3f}ms, 吞吐量: {throughput:.2f} samples/s")
        return {
            'latency': latency,
            'throughput': throughput,
            'batch_size': batch_size,
            'model_name': model_name
        }

    except Exception as e:
        print(f"  推理失败: {e}")
        return None


def measure_background_tasks(background_tasks):
    """
    测量多个背景任务的GPU资源利用率
    Args:
        background_tasks: 背景任务列表，每个是 {'model': str, 'batch': int}

    Returns:
        tuple: (avg_mem_bw_util, avg_gpu_util) 平均资源利用率
    """
    if not background_tasks:
        return 0, 0

    print(f"\n[资源监控] 测量 {len(background_tasks)} 个背景任务的资源利用率")
    print("-" * 50)

    # 显示背景任务配置信息
    print("背景任务配置:")
    for i, task in enumerate(background_tasks, 1):
        print(f"  任务 {i}: 模型={task['model']}, 批次大小={task['batch']}")
    print("-" * 50)

    # 确保MPS运行
    if not is_mps_running():
        if not start_mps_simple():
            print("错误: 无法启动MPS")
            return 0, 0

    # 加载配置以获取GPU句柄
    try:
        with open('./config/platform.json', 'r') as f:
            platform_raw = json.load(f)
        # 适配新格式: platform_raw["platforms"][0]
        if "platforms" in platform_raw:
            platform_data = platform_raw["platforms"][0]
        else:
            platform_data = platform_raw

        # 使用第一个任务的模型加载配置
        first_task = background_tasks[0]
        config_path = f'./config/{first_task["model"]}/{first_task["model"]}.json'
        if not os.path.exists(config_path):
            config_path = f'./config/{first_task["model"]}/{first_task["model"].split("_")[0]}.json'

        with open(config_path, 'r') as f:
            model_data = json.load(f)
            configs = globalConfig.GlobalConfig(platform_data, model_data)
    except Exception as e:
        print(f"错误: 加载配置失败 - {e}")
        return 0, 0

    # 准备所有背景任务
    task_data_list = []
    for task in background_tasks:
        data = dataLoader.load_images_to_batches(configs.model.data_path, task['batch'])
        model_data = {
            "model_name": task['model'],
            "client": None,
            "data": data,
            "resource": None,
            "batchsize": task['batch']
        }
        task_data_list.append(model_data)

    # 初始化pynvml
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        print(f"错误: 无法初始化NVML - {e}")
        return 0, 0

    # 测量几轮取平均值
    utilization_samples = []

    for round_num in range(1):  # 只进行1轮快速测量
        print(f"\n快速资源测量...")

        # 启动所有背景任务
        stop_event = threading.Event()
        sampler_thread = ThreadPoolExecutor(max_workers=1).submit(doProfile.sample_gpu_info, stop_event, handle)

        # 并行运行所有背景任务
        with ThreadPoolExecutor(max_workers=len(background_tasks)) as executor:
            futures = [executor.submit(doInference.inference, task_data)
                      for task_data in task_data_list]

            # 等待所有任务完成
            for future in futures:
                try:
                    latency, throughput = future.result(timeout=30)
                except Exception as e:
                    print(f"  任务执行失败: {e}")

        # 停止采样
        stop_event.set()

        # 获取采样结果
        try:
            monitor_result = sampler_thread.result()
            # 解包10个值（与main.py一致）
            (max_power, avg_power,
             max_gpu_util, avg_gpu_util,
             max_mem_util, avg_mem_util,
             max_pcie, avg_pcie,
             max_mem_bw_util, avg_mem_bw_util, *_) = monitor_result
            # 记录内存带宽和GPU利用率
            utilization_samples.append({
                'mem_bw_util': avg_mem_bw_util,
                'gpu_util': avg_gpu_util
            })

        except Exception as e:
            print(f"  获取监控结果失败: {e}")

        time.sleep(1)  # 等待一秒再进行下一轮

    # 计算平均值
    if utilization_samples:
        avg_mem_bw_util = np.mean([u['mem_bw_util'] for u in utilization_samples])
        avg_gpu_util = np.mean([u['gpu_util'] for u in utilization_samples])

        print(f"\n[资源监控] 测量完成!")
        print(f"  平均GPU利用率: {avg_gpu_util:.1f}%")
        print(f"  平均内存带宽利用率: {avg_mem_bw_util:.1f}%")
        print("-" * 50)

        # 转换为0-1的比例
        return avg_mem_bw_util / 100.0, avg_gpu_util / 100.0
    else:
        print("错误: 没有获得有效的采样数据")
        return 0, 0


def test_with_current_task(background_tasks, current_model, current_batch):
    """
    测试在背景任务存在的情况下运行当前任务

    Args:
        background_tasks: 背景任务列表
        current_model: 当前任务模型名称
        current_batch: 当前任务批次大小

    Returns:
        dict: 测试结果
    """
    print(f"\n[共置测试] 背景任务: {len(background_tasks)}个, 当前任务: {current_model}, batch={current_batch}")
    print("-" * 70)

    # 确保MPS运行
    if not is_mps_running():
        if not start_mps_simple():
            print("错误: 无法启动MPS")
            return None

    # 加载配置
    try:
        with open('./config/platform.json', 'r') as f:
            platform_raw = json.load(f)
        # 适配新格式: platform_raw["platforms"][0]
        if "platforms" in platform_raw:
            platform_data = platform_raw["platforms"][0]
        else:
            platform_data = platform_raw

        config_path = f'./config/{current_model}/{current_model}.json'
        if not os.path.exists(config_path):
            config_path = f'./config/{current_model}/{current_model.split("_")[0]}.json'

        with open(config_path, 'r') as f:
            model_data = json.load(f)
            configs = globalConfig.GlobalConfig(platform_data, model_data)
    except Exception as e:
        print(f"错误: 加载配置失败 - {e}")
        return None

    # 准备所有任务数据
    all_task_data = []

    # 背景任务
    for task in background_tasks:
        data = dataLoader.load_images_to_batches(configs.model.data_path, task['batch'])
        model_data = {
            "model_name": task['model'],
            "client": None,
            "data": data,
            "resource": None,
            "batchsize": task['batch']
        }
        all_task_data.append(model_data)

    # 当前任务
    current_data = dataLoader.load_images_to_batches(configs.model.data_path, current_batch)
    current_task_data = {
        "model_name": current_model,
        "client": None,
        "data": current_data,
        "resource": None,
        "batchsize": current_batch
    }
    all_task_data.append(current_task_data)

    # 初始化NVML
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        print(f"错误: 无法初始化NVML - {e}")
        return None

    # 预热所有任务
    print("预热所有任务...")
    for task_data in all_task_data:
        try:
            doInference_pytorch.inference_pytorch(task_data)
        except Exception as e:
            print(f"预热失败: {e}")

    # 运行测试
    print("运行共置测试...")
    throughputs = []

    for test_round in range(3):
        print(f"  第 {test_round + 1}/3 轮...")

        # 启动监控
        stop_event = threading.Event()
        sampler_thread = ThreadPoolExecutor(max_workers=1).submit(
            doProfile.sample_gpu_info, stop_event, handle
        )

        # 并行运行所有任务
        with ThreadPoolExecutor(max_workers=len(all_task_data)) as executor:
            futures = [executor.submit(doInference.inference, task_data)
                      for task_data in all_task_data]

            # 收集结果
            results = []
            for i, future in enumerate(futures):
                try:
                    latency, throughput = future.result(timeout=30)
                    results.append((latency, throughput))
                except Exception as e:
                    print(f"    任务 {i} 失败: {e}")

        # 停止监控
        stop_event.set()

        # 当前任务的吞吐量是最后一个结果
        if results:
            current_latency, current_throughput = results[-1]
            throughputs.append(current_throughput)
            print(f"    当前任务吞吐量: {current_throughput:.2f} samples/s")

    # 计算平均吞吐量
    if throughputs:
        avg_throughput = np.mean(throughputs)
        print(f"\n[共置测试] 完成!")
        print(f"  平均吞吐量: {avg_throughput:.2f} samples/s")
        print("-" * 70)

        return {
            'throughput': avg_throughput,
            'background_tasks': len(background_tasks),
            'current_batch': current_batch,
            'current_model': current_model
        }

    return None


if __name__ == "__main__":
    """
    测试背景任务运行器
    """
    import argparse

    parser = argparse.ArgumentParser(description='背景任务运行器测试')
    parser.add_argument('--test', type=str, choices=['single', 'measure', 'colocate'],
                       default='single', help='测试类型')
    parser.add_argument('--bg-model', type=str, default='resnet50', help='背景任务模型')
    parser.add_argument('--bg-batch', type=int, default=32, help='背景任务批次大小')
    parser.add_argument('--bg-count', type=int, default=1, help='背景任务数量')
    parser.add_argument('--cur-model', type=str, default='resnet50', help='当前任务模型')
    parser.add_argument('--cur-batch', type=int, default=16, help='当前任务批次大小')

    args = parser.parse_args()

    # 创建背景任务配置
    background_tasks = []
    for i in range(args.bg_count):
        background_tasks.append({
            'model': args.bg_model,
            'batch': args.bg_batch
        })

    if args.test == 'single':
        # 测试单个背景任务
        result = run_single_background_task(args.bg_model, args.bg_batch)
        if result:
            print(f"\n✓ 单任务测试成功!")
        else:
            print(f"\n✗ 单任务测试失败!")

    elif args.test == 'measure':
        # 测量背景任务资源利用率
        bw_mem, util_sm = measure_background_tasks(background_tasks)
        print(f"\n结果: 内存带宽利用率={bw_mem:.2%}, GPU利用率={util_sm:.2%}")

    elif args.test == 'colocate':
        # 测试共置场景
        result = test_with_current_task(background_tasks, args.cur_model, args.cur_batch)
        if result:
            print(f"\n✓ 共置测试成功! 吞吐量: {result['throughput']:.2f}")
        else:
            print(f"\n✗ 共置测试失败!")
