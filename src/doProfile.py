import pynvml
import time
import warnings

def sample_gpu_info(stop_event, handle, warmup_seconds=0.0):
    total_power = 0
    max_power = 0
    max_pcie =0
    total_pcie_tx = 0
    max_pcie_tx = 0
    total_pcie_rx = 0
    max_pcie_rx = 0
    total_gpu_utilization = 0
    max_gpu_utilization = 0
    total_mem_utilization = 0
    max_mem_utilization = 0
    total_mem_bandwidth_utilization = 0  # Memory bandwidth utilization
    max_mem_bandwidth_utilization = 0     # Memory bandwidth utilization peak
    # Note: NVML utilization.memory is "memory controller busy time percentage (%)", not actual DRAM throughput.
    # Below we also provide an estimated value (GB/s) based on "current mem clock * bus width * utilization.memory".
    total_mem_clock_mhz = 0.0
    max_mem_clock_mhz = 0.0
    mem_bus_width_bits = None
    total_peak_dram_bw_gbs = 0.0
    max_peak_dram_bw_gbs = 0.0
    total_est_dram_bw_gbs = 0.0
    max_est_dram_bw_gbs = 0.0
    count = 0
    start_time = time.perf_counter()
    warmup_until = start_time + max(0.0, float(warmup_seconds))
    while not stop_event.is_set():
        if time.perf_counter() < warmup_until:
            time.sleep(0.01)
            continue
        #Power
        power_gpu=pynvml.nvmlDeviceGetPowerUsage(handle)/1000
        total_power+=power_gpu
        max_power=max(max_power,power_gpu)
        # 获取 PCIe 吞吐量（发送和接收）
        pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1024
        pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1024
        total_pcie_rx+=pcie_rx
        max_pcie_rx=max(max_pcie_rx,pcie_rx)
        total_pcie_tx+=pcie_tx
        max_pcie_tx=max(max_pcie_tx,pcie_tx)
        max_pcie = max(pcie_rx+pcie_tx,max_pcie)
        # 获取 GPU 利用率和内存带宽利用率
        utilization_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = utilization_rates.gpu      # GPU SM 利用率
        mem_bw_util = utilization_rates.memory  # 内存控制器利用率（带宽利用率）
        total_gpu_utilization+=gpu_util
        max_gpu_utilization=max(max_gpu_utilization,gpu_util)
        total_mem_bandwidth_utilization+=mem_bw_util
        max_mem_bandwidth_utilization=max(max_mem_bandwidth_utilization,mem_bw_util)
        # 估算DRAM带宽(GB/s)：utilization.memory(%) * 峰值带宽(由mem clock与bus width推导)
        try:
            mem_clock_mhz = float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
            if mem_bus_width_bits is None:
                mem_bus_width_bits = int(pynvml.nvmlDeviceGetMemoryBusWidth(handle))
            peak_dram_bw_gbs = (mem_clock_mhz * 1e6) * (mem_bus_width_bits / 8.0) * 2.0 / 1e9
            est_dram_bw_gbs = (float(mem_bw_util) / 100.0) * peak_dram_bw_gbs

            total_mem_clock_mhz += mem_clock_mhz
            max_mem_clock_mhz = max(max_mem_clock_mhz, mem_clock_mhz)
            total_peak_dram_bw_gbs += peak_dram_bw_gbs
            max_peak_dram_bw_gbs = max(max_peak_dram_bw_gbs, peak_dram_bw_gbs)
            total_est_dram_bw_gbs += est_dram_bw_gbs
            max_est_dram_bw_gbs = max(max_est_dram_bw_gbs, est_dram_bw_gbs)
        except Exception:
            # Some environments/drivers may not support these queries, ignore (keep original 10 return values usable)
            pass
        # 获取内存使用率（显存占用）
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_uti = memory_info.used / memory_info.total * 100
        total_mem_utilization+=mem_uti
        max_mem_utilization=max(max_mem_utilization,mem_uti)
        count = count + 1
        time.sleep(0.01)
    if count <= 0:
        warnings.warn("NVML sample window too short; no samples collected.", RuntimeWarning)
        return (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None, None)
    # 返回值：max_power, avg_power, max_gpu_util, avg_gpu_util, max_mem_usage, avg_mem_usage,
    #         max_pcie, avg_pcie, max_mem_bw_util, avg_mem_bw_util,
    #         max_mem_clock_mhz, avg_mem_clock_mhz, mem_bus_width_bits,
    #         max_peak_dram_bw_gbs, avg_peak_dram_bw_gbs,
    #         max_est_dram_bw_gbs, avg_est_dram_bw_gbs,
    #         max_pcie_tx, avg_pcie_tx, max_pcie_rx, avg_pcie_rx
    avg_mem_clock_mhz = (total_mem_clock_mhz / count) if total_mem_clock_mhz > 0 else None
    avg_peak_dram_bw_gbs = (total_peak_dram_bw_gbs / count) if total_peak_dram_bw_gbs > 0 else None
    avg_est_dram_bw_gbs = (total_est_dram_bw_gbs / count) if total_est_dram_bw_gbs > 0 else None
    return (max_power, total_power/count,
            max_gpu_utilization, total_gpu_utilization/count,
            max_mem_utilization, total_mem_utilization/count,
            max_pcie, (total_pcie_rx+total_pcie_tx)/count,
            max_mem_bandwidth_utilization, total_mem_bandwidth_utilization/count,
            (max_mem_clock_mhz if max_mem_clock_mhz > 0 else None),
            avg_mem_clock_mhz,
            mem_bus_width_bits,
            (max_peak_dram_bw_gbs if max_peak_dram_bw_gbs > 0 else None),
            avg_peak_dram_bw_gbs,
            (max_est_dram_bw_gbs if max_est_dram_bw_gbs > 0 else None),
            avg_est_dram_bw_gbs,
            max_pcie_tx, (total_pcie_tx/count),
            max_pcie_rx, (total_pcie_rx/count))


def measure_gpu_info(handle, seconds, sample_interval=0.05):
    """
    与 profile_mps.py 一致的功耗测量窗口。

    Returns:
        (elapsed, power_avg, power_max, clock, sm_util, mem_util, pcie_mb)
    """
    power_samples, clock_samples = [], []
    sm_util_samples, mem_util_samples, pcie_mb_samples = [], [], []

    start_t = time.perf_counter()
    last_sample_t = start_t

    while time.perf_counter() - start_t < seconds:
        try:
            current_t = time.perf_counter()
            if current_t - last_sample_t > 2.0:
                break
            last_sample_t = current_t

            power_samples.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
            clock_samples.append(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))

            utils = pynvml.nvmlDeviceGetUtilizationRates(handle)
            sm_util_samples.append(utils.gpu)
            mem_util_samples.append(utils.memory)

            try:
                tx_bytes = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                rx_bytes = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
                pcie_mb_samples.append((tx_bytes + rx_bytes) / 1000.0)
            except Exception:
                pcie_mb_samples.append(0.0)

            time.sleep(sample_interval)
        except Exception:
            break

    elapsed = time.perf_counter() - start_t

    def _avg(samples):
        return sum(samples) / len(samples) if samples else 0.0

    return (
        elapsed,
        _avg(power_samples),
        max(power_samples) if power_samples else 0.0,
        _avg(clock_samples),
        _avg(sm_util_samples),
        _avg(mem_util_samples),
        _avg(pcie_mb_samples),
    )
    
