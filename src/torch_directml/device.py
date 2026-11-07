import torch_directml_native
import torch

def is_built() -> bool:
    r"""Returns a bool indicating if DML is built."""
    return True

def is_available() -> bool:
    r"""Returns a bool indicating if DML is currently available."""
    if not hasattr(torch_directml_native, 'get_device_count'):
        return False
    # This function never throws and returns 0 if driver is missing or can't
    # be initialized
    return torch_directml_native.get_device_count() > 0

def device_count() -> int:
    r"""Returns the number of GPUs available."""
    if is_available():
        return torch_directml_native.get_device_count()
    else:
        return 0

def device_name(device_id) -> str:
    r"""Returns the device name of the specified index."""
    if device_id >= 0 and device_id < device_count():
        return torch_directml_native.get_device_name(device_id)
    else:
        return ""

def default_device() -> int:
    r"""Returns the index of the default selected device."""
    return torch_directml_native.get_default_device()

def device(device_id = None) -> str:
    r"""Returns the torch device at the specified index."""
    num_devices = device_count()
    if device_id is None:
        return torch_directml_native.custom_device(default_device())
    elif device_id >= 0 and device_id < num_devices:
        return torch_directml_native.custom_device(device_id)
    else:
        raise Exception(f"Invalid device_id argument supplied {device_id}. device_id must be in range [0, {num_devices}).")

def disable_tiled_resources(is_disabled):
    torch_directml_native.disable_tiled_resources(is_disabled)

def has_float64_support(device_id = default_device()):
    return torch_directml_native.has_float64_support(device_id)

def gpu_memory(device_id = default_device(), mb_per_tile = 1):
    return torch_directml_native.get_gpu_memory(device_id, mb_per_tile)