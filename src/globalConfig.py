import threading
import pynvml
import subprocess
import re
from concurrent.futures import ThreadPoolExecutor


def detect_current_gpu():
    """
    Detect current GPU name via nvidia-smi -L.

    Returns:
        str: GPU name, e.g., "NVIDIA GeForce RTX 3080 Ti"
    """
    try:
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Output format: GPU 0: NVIDIA GeForce RTX 3080 Ti (UUID-...)
            match = re.search(r'GPU \d+:\s*(.+?)\s*\(', result.stdout)
            if match:
                gpu_name = match.group(1).strip()
                print(f"[GlobalConfig] Detected GPU: {gpu_name}")
                return gpu_name
    except Exception as e:
        print(f"[GlobalConfig] GPU detection failed: {e}")
    return None


def _normalize_platform_key(name):
    return "".join(ch for ch in name.lower() if ch.isalnum()) if name else ""


def find_matching_platform(platform_data, platform_hint=None):
    """
    Find matching platform configuration from platform.json for the current GPU.

    Args:
        platform_data: Loaded platform.json data
        platform_hint: Optional platform hint (e.g., "a100", "3080ti")

    Returns:
        tuple: (platform_name, sm_clocks, mem_clocks)
    """
    current_gpu = detect_current_gpu()
    hint_key = _normalize_platform_key(platform_hint) if platform_hint else ""

    # 新格式: platform_data["platforms"]
    if "platforms" in platform_data:
        platforms = platform_data["platforms"]

        if hint_key:
            for platform in platforms:
                platform_name = platform.get("name", "")
                name_key = _normalize_platform_key(platform_name)
                if hint_key == name_key or hint_key in name_key or name_key in hint_key:
                    print(f"[GlobalConfig] Using specified platform: {platform_name}")
                    clocks = platform.get("clocks", {})
                    sm_clocks = clocks.get("SM_Clocks") or clocks.get("Graphics_Clocks", [])
                    mem_clocks = clocks.get("Memory_Clocks") or clocks.get("Memory_Clock", [])
                    if isinstance(sm_clocks, int):
                        sm_clocks = [sm_clocks]
                    if isinstance(mem_clocks, int):
                        mem_clocks = [mem_clocks]
                    return platform_name, sm_clocks, mem_clocks

        # If GPU detected, try to match
        if current_gpu:
            for platform in platforms:
                platform_name = platform.get("name", "")
                # Check if current GPU name contains platform name keywords
                # Supports "3080 Ti" matching "NVIDIA GeForce RTX 3080 Ti"
                # or "RTX 3080 Ti" matching "NVIDIA GeForce RTX 3080 Ti"
                if current_gpu == platform_name:
                    print(f"[GlobalConfig] Exact match platform: {platform_name}")
                    clocks = platform.get("clocks", {})
                    sm_clocks = clocks.get("SM_Clocks") or clocks.get("Graphics_Clocks", [])
                    mem_clocks = clocks.get("Memory_Clocks") or clocks.get("Memory_Clock", [])
                    # Ensure it's a list (fix A100 config being an integer)
                    if isinstance(sm_clocks, int):
                        sm_clocks = [sm_clocks]
                    if isinstance(mem_clocks, int):
                        mem_clocks = [mem_clocks]
                    return platform_name, sm_clocks, mem_clocks

                # Fuzzy match: check keywords
                # Extract keywords from platform name (e.g., "3080 Ti", "A100")
                keywords = [k for k in platform_name.split() if k not in ["NVIDIA", "GeForce", "RTX", "Tesla", "GPU"]]
                if keywords and all(kw in current_gpu for kw in keywords):
                    print(f"[GlobalConfig] Keyword match platform: {platform_name} (matched keywords: {keywords})")
                    clocks = platform.get("clocks", {})
                    sm_clocks = clocks.get("SM_Clocks") or clocks.get("Graphics_Clocks", [])
                    mem_clocks = clocks.get("Memory_Clocks") or clocks.get("Memory_Clock", [])
                    # Ensure it's a list (fix A100 config being an integer)
                    if isinstance(sm_clocks, int):
                        sm_clocks = [sm_clocks]
                    if isinstance(mem_clocks, int):
                        mem_clocks = [mem_clocks]
                    return platform_name, sm_clocks, mem_clocks

        # No match found, use first platform
        first_platform = platforms[0]
        platform_name = first_platform["name"]
        print(f"[GlobalConfig] No matching platform found, using default: {platform_name}")
        clocks = first_platform.get("clocks", {})
        sm_clocks = clocks.get("SM_Clocks") or clocks.get("Graphics_Clocks", [])
        mem_clocks = clocks.get("Memory_Clocks") or clocks.get("Memory_Clock", [])
        # Ensure it's a list
        if isinstance(sm_clocks, int):
            sm_clocks = [sm_clocks]
        if isinstance(mem_clocks, int):
            mem_clocks = [mem_clocks]
        return platform_name, sm_clocks, mem_clocks

    # Legacy format compatibility
    else:
        platform_name = platform_data.get("name", "Unknown")
        sm_clocks = platform_data.get("SM Clocks", [])
        mem_clocks = platform_data.get("Memory Clocks", [])
        print(f"[GlobalConfig] Using legacy format config: {platform_name}")
        return platform_name, sm_clocks, mem_clocks


class GlobalConfig:
    def __init__(self,platform_data,model_data,platform_hint=None):
        # Auto-detect current GPU and match platform config
        platform_name, sm_clocks, mem_clocks = find_matching_platform(platform_data, platform_hint=platform_hint)
        self.platform = PlatformConfig(platform_name, sm_clocks, mem_clocks)
        self.model = ModelConfig(model_data["modelName"],model_data["GPU resources"],model_data["data set path"],model_data["Power Cap"]
                                 ,model_data["model repository"],model_data["batch size list"],model_data["log file path"],model_data["max batchsize"]
                                 ,model_data["SLO"],model_data["alpha"],model_data["belta"])
        self.monitor_executor = ThreadPoolExecutor(max_workers=1)
        self.model_executor = ThreadPoolExecutor(max_workers=len(self.model.resources))
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
class PlatformConfig:
    def __init__(self,name,sm_clocks,mem_clocks):
        self.name = name
        sm_clocks.sort()
        mem_clocks.sort()
        self.sm_clocks = sm_clocks
        self.mem_clocks = mem_clocks
class ModelConfig:
    def __init__(self,model_name,resources,data_path,power_cap,model_repository,batch_size_list,log_path,max_bs,slo,alpha,belta):
        self.model_name = model_name
        self.resources = resources
        self.data_path = data_path
        self.power_cap = power_cap
        self.model_repository = model_repository
        self.batch_size_list = batch_size_list
        self.log_path = log_path
        self.max_bs = max_bs
        self.slo = slo
        self.alpha = alpha
        self.belta = belta
