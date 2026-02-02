"""
PyTorch direct inference module - fast startup without Docker/Triton overhead.

Pros:
- Startup time drops from ~30-60s to ~1-3s
- No Docker container management
- Simpler code, easier to debug

Cons:
- Loses Triton dynamic batching optimizations
- Manual model loading and batching
"""

import os
import time
import torch
import numpy as np
from torchvision import models
import pynvml

# Global model cache to avoid reloading
_model_cache = {}
_device = None
_IS_WORKER = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" in os.environ

def get_device():
    """Get the GPU device."""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not _IS_WORKER:
            print(f"使用设备: {_device}")
    return _device

def _fuzzy_match_model(model_name: str, model_map: dict) -> str:
    """
    Fuzzy match a model name.

    Priority:
    1. Exact match
    2. Prefix match (maxvit -> maxvit_t)
    3. Contains match
    4. No match: return original name
    """
    # Exact match
    if model_name in model_map:
        return model_name

    # Normalize: lowercase, remove underscores and hyphens
    normalized = model_name.lower().replace('_', '').replace('-', '')

    # Prefix match: shortest key that starts with model_name
    prefix_matches = []
    for key in model_map.keys():
        key_normalized = key.lower().replace('_', '').replace('-', '')
        if key_normalized.startswith(normalized):
            prefix_matches.append((key, len(key)))

    if prefix_matches:
        # Return the shortest (most specific) match
        matched = min(prefix_matches, key=lambda x: x[1])[0]
        return matched

    # Contains match: shortest key containing model_name (or vice versa)
    contain_matches = []
    for key in model_map.keys():
        key_normalized = key.lower().replace('_', '').replace('-', '')
        if normalized in key_normalized or key_normalized in normalized:
            contain_matches.append((key, len(key)))

    if contain_matches:
        matched = min(contain_matches, key=lambda x: x[1])[0]
        return matched

    return model_name


def load_model(model_name, use_cache=True):
    """
    Load a PyTorch model.

    Args:
        model_name: Model name (e.g., 'resnet50', 'efficientnet_v2_m', 'maxvit')
        use_cache: Whether to use cache

    Returns:
        model: Loaded PyTorch model
    """
    global _model_cache

    if use_cache and model_name in _model_cache:
        if not _IS_WORKER:
            # Log cache hits only once per model to reduce spam
            if not hasattr(load_model, 'cache_logged'):
                load_model.cache_logged = set()
            if model_name not in load_model.cache_logged:
                load_model.cache_logged.add(model_name)
                print(f"✓ 模型已缓存: {model_name}")
        return _model_cache[model_name]

    if not _IS_WORKER:
        print(f"首次加载模型: {model_name}")
    device = get_device()

    # Model map
    model_map = {
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'densenet201': models.densenet201,
        'efficientnet_v2_m': models.efficientnet_v2_m,
        'efficientnet_v2_l': models.efficientnet_v2_l,
        'vgg19': models.vgg19,
        'mobilenet_v2': models.mobilenet_v2,
        'mobilenetv2': models.mobilenet_v2,  # Alias for config compatibility
        'alexnet': models.alexnet,
        'mnasnet': models.mnasnet1_0,
        'convnext_base': models.convnext_base,
        'swin_b': models.swin_b,
        'maxvit_t': models.maxvit_t,
    }

    # Fuzzy match to the closest model name
    matched_name = _fuzzy_match_model(model_name, model_map)
    if matched_name != model_name:
        if matched_name in model_map:
            if not _IS_WORKER:
                print(f"  模糊匹配: '{model_name}' -> '{matched_name}'")
            model_name = matched_name
        else:
            raise ValueError(f"不支持的模型: {model_name}。支持的模型: {list(model_map.keys())}")
    
    # Build model
    # --- Official weights ---
    from torchvision.models import (
        ResNet50_Weights, ResNet101_Weights, DenseNet201_Weights, 
        EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights, VGG19_Weights, 
        MobileNet_V2_Weights, AlexNet_Weights, ConvNeXt_Base_Weights,
        Swin_B_Weights, MaxVit_T_Weights, MNASNet1_0_Weights
    )
    weights_map = {
        'resnet50': ResNet50_Weights.IMAGENET1K_V2,
        'resnet101': ResNet101_Weights.IMAGENET1K_V2,
        'densenet201': DenseNet201_Weights.IMAGENET1K_V1,
        'efficientnet_v2_m': EfficientNet_V2_M_Weights.IMAGENET1K_V1,
        'efficientnet_v2_l': EfficientNet_V2_L_Weights.IMAGENET1K_V1,
        'vgg19': VGG19_Weights.IMAGENET1K_V1,
        'mobilenet_v2': MobileNet_V2_Weights.IMAGENET1K_V2,
        'mobilenetv2': MobileNet_V2_Weights.IMAGENET1K_V2,
        'alexnet': AlexNet_Weights.IMAGENET1K_V1,
        'mnasnet': MNASNet1_0_Weights.IMAGENET1K_V1,
        'convnext_base': ConvNeXt_Base_Weights.IMAGENET1K_V1,
        'swin_b': Swin_B_Weights.IMAGENET1K_V1,
        'maxvit_t': MaxVit_T_Weights.IMAGENET1K_V1,
    }
    
    weights = weights_map.get(model_name)
    if weights:
        if not _IS_WORKER:
            print(f"加载官方预训练权重: {weights}")
        model = model_map[model_name](weights=weights)
    else:
        if not _IS_WORKER:
            print(f"警告: 未找到 {model_name} 的官方权重，使用随机初始化")
        model = model_map[model_name](weights=None)

    model = model.to(device)
    model.eval()  # Eval mode
    
    # Mixed precision acceleration
    if torch.cuda.is_available():
        model = model.half()  # FP16
    
    if use_cache:
        _model_cache[model_name] = model
    
    return model

def inference(model_config):
    """
    Run inference with PyTorch (Triton replacement).

    Args:
        model_config: Dict with:
            - model_name: Model name
            - data: Input batches (each element is a NumPy array)
            - batchsize: Batch size
            - resource: GPU allocation (ignored here; handled by MPS)

    Returns:
        (latency_ms, throughput): Latency (ms) and throughput (samples/s)
    """
    model_name = model_config["model_name"]
    input_batches = model_config.get("data", []) or []
    batch_size = int(model_config["batchsize"])
    max_batches = model_config.get("max_batches", None)
    skip_warmup = bool(model_config.get("skip_warmup", False))
    measure_seconds = model_config.get("measure_seconds", None)
    min_iters = int(model_config.get("min_iters", 1) or 1)
    reuse_input = bool(model_config.get("reuse_input", False))
    synthetic = bool(model_config.get("synthetic", False))

    if max_batches is not None:
        try:
            max_batches_int = int(max_batches)
        except Exception:
            max_batches_int = None
        if max_batches_int is not None and max_batches_int > 0:
            input_batches = input_batches[:max_batches_int]

    device = get_device()
    desired_dtype = torch.float16 if device.type == "cuda" else torch.float32
    if synthetic or len(input_batches) == 0:
        # Keep synthetic data lightweight; dtype gets normalized later when tensors are created.
        input_batches = [np.zeros((batch_size, 3, 224, 224), dtype=np.float16)]

    model = load_model(model_name)
    
    total_samples = 0
    inference_times = []
    
    # Warmup (first inference can be slower)
    if (not skip_warmup) and len(input_batches) > 0:
        warmup_data = torch.from_numpy(input_batches[0]).to(device).to(desired_dtype)
        with torch.no_grad():
            _ = model(warmup_data)
        if device.type == "cuda":
            torch.cuda.synchronize()  # Wait for GPU

    # Run inference; supports time-window measurement
    if measure_seconds is not None:
        try:
            measure_seconds = float(measure_seconds)
        except Exception:
            measure_seconds = None
        if measure_seconds is not None and measure_seconds <= 0:
            measure_seconds = None

    if measure_seconds is not None:
        if reuse_input:
            input_tensors = [torch.from_numpy(input_batches[0]).to(device).to(desired_dtype)]
        else:
            # Move inputs to GPU up front to avoid H2D cost in the measurement window
            input_tensors = [torch.from_numpy(b).to(device).to(desired_dtype) for b in input_batches]

        start_time = time.perf_counter()
        iters = 0
        while True:
            input_tensor = input_tensors[iters % len(input_tensors)]
            with torch.no_grad():
                batch_start = time.perf_counter()
                _ = model(input_tensor)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                batch_time = (time.perf_counter() - batch_start) * 1000

            inference_times.append(batch_time)
            total_samples += int(input_tensor.shape[0])
            iters += 1

            elapsed = time.perf_counter() - start_time
            if iters >= min_iters and elapsed >= measure_seconds:
                break

        total_time = time.perf_counter() - start_time
    else:
        start_time = time.perf_counter()
        for input_data in input_batches:
            input_tensor = torch.from_numpy(input_data).to(device).to(desired_dtype)
            with torch.no_grad():
                batch_start = time.perf_counter()
                _ = model(input_tensor)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                batch_time = (time.perf_counter() - batch_start) * 1000
            inference_times.append(batch_time)
            total_samples += input_data.shape[0]
        total_time = time.perf_counter() - start_time
    
    # Average latency and throughput
    avg_latency_ms = np.mean(inference_times) if inference_times else 0
    throughput = total_samples / total_time if total_time > 0 else 0
    
    return avg_latency_ms, throughput

def inference_with_stats(model_config):
    """
    PyTorch inference with stats (compatible with the Triton interface).

    Returns:
        (avg_latency_ms, throughput): Triton-compatible output
    """
    return inference(model_config)

def preload_models(model_names):
    """
    Preload models into GPU memory.
    
    Args:
        model_names: List of model names
    """
    print(f"预加载 {len(model_names)} 个模型...")
    start_time = time.time()
    
    for model_name in model_names:
        try:
            load_model(model_name, use_cache=True)
            print(f"  ✓ {model_name} 加载完成")
        except Exception as e:
            print(f"  ✗ {model_name} 加载失败: {e}")
    
    elapsed = time.time() - start_time
    print(f"预加载完成，耗时: {elapsed:.2f}秒")

def clear_model_cache():
    """Clear the model cache."""
    global _model_cache
    _model_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("模型缓存已清空")

def get_model_info():
    """Get information about loaded models."""
    info = {
        "cached_models": list(_model_cache.keys()),
        "device": str(get_device()),
        "cuda_available": torch.cuda.is_available()
    }
    if torch.cuda.is_available():
        info["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
        info["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024**3  # GB
    return info

# Example usage
if __name__ == "__main__":
    # Test code
    print("="*50)
    print("PyTorch直接推理模块测试")
    print("="*50)
    
    # Generate test data
    test_data = [np.random.randn(32, 3, 224, 224).astype(np.float32)]
    
    model_config = {
        "model_name": "resnet50",
        "data": test_data,
        "batchsize": 32,
        "resource": 100
    }
    
    # Run inference test
    print("\n开始推理测试...")
    latency, throughput = inference(model_config)
    
    print(f"\n结果:")
    print(f"  平均延迟: {latency:.2f} ms")
    print(f"  吞吐量: {throughput:.2f} samples/s")
    
    # Show model info
    print(f"\n模型信息:")
    info = get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
