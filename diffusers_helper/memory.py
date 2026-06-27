# By lllyasviel
# GPU memory management system that enables running 13B+ parameter models on GPUs
# with as little as 6GB VRAM. Uses a combination of model swapping (only one model
# on GPU at a time) and dynamic parameter streaming (DynamicSwapInstaller).


import os
import torch


cpu = torch.device('cpu')

if torch.cuda.is_available():
    gpu = torch.device(f'cuda:{torch.cuda.current_device()}')
elif torch.backends.mps.is_available():
    gpu = torch.device('mps')
else:
    gpu = torch.device('cpu')
# Tracks which models are fully loaded on GPU so they can be bulk-unloaded later
gpu_complete_modules = []


# DynamicSwapInstaller enables "on-demand" GPU loading of model parameters.
# Instead of loading the entire model to GPU (which might not fit), it monkey-patches
# each module's __getattr__ so that parameters are moved to GPU only when accessed
# during the forward pass. This is 3x faster than HuggingFace's enable_sequential_offload.
#
# How it works:
#   1. The model stays on CPU.
#   2. When the forward pass accesses module.weight, the patched __getattr__ intercepts
#      the access and calls weight.to(device=gpu) on-the-fly.
#   3. After the forward pass, the weight can be garbage collected from GPU.
#   4. Only the currently-executing layer needs to fit in GPU memory at any time.
def _empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs):
        original_class = module.__class__
        module.__dict__['forge_backup_original_class'] = original_class

        # This replaces the default attribute access. When the forward pass
        # does `self.weight`, this function intercepts it and moves the weight
        # to the target device (GPU) before returning it.
        def hacked_get_attr(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    p = _parameters[name]
                    if p is None:
                        return None
                    if p.__class__ == torch.nn.Parameter:
                        return torch.nn.Parameter(p.to(**kwargs), requires_grad=p.requires_grad)
                    else:
                        return p.to(**kwargs)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name].to(**kwargs)
            return super(original_class, self).__getattr__(name)

        # Create a new dynamic subclass with the patched __getattr__
        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
        })

        return

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs):
        for m in model.modules():
            DynamicSwapInstaller._install_module(m, **kwargs)
        return


# Tricks the diffusers library into thinking a model is on GPU by moving just ONE
# small parameter to GPU. Diffusers checks model.device to decide where to put inputs;
# this avoids loading the entire model just to pass the device check.
def fake_diffusers_current_device(model: torch.nn.Module, target_device: torch.device):
    if hasattr(model, 'scale_shift_table'):
        model.scale_shift_table.data = model.scale_shift_table.data.to(target_device)
        return

    for _, p in model.named_modules():
        if hasattr(p, 'weight'):
            p.to(target_device)
            return


# Returns truly available GPU memory in GB. More accurate than just torch.cuda.mem_get_info
# because it also accounts for "inactive reserved" memory — memory that PyTorch's allocator
# has reserved but isn't currently using, which can be reclaimed without a CUDA call.
# Formula: available = free_cuda + (reserved - active)
def get_cuda_free_memory_gb(device=None):
    if device is None:
        device = gpu

    if device.type == 'cuda':
        memory_stats = torch.cuda.memory_stats(device)
        bytes_active = memory_stats['active_bytes.all.current']
        bytes_reserved = memory_stats['reserved_bytes.all.current']
        bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
        bytes_inactive_reserved = bytes_reserved - bytes_active
        bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
        return bytes_total_available / (1024 ** 3)
    elif device.type == 'mps':
        total_mem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        allocated = torch.mps.current_allocated_memory()
        return (total_mem - allocated) / (1024 ** 3)
    else:
        return float('inf')


# Moves model submodules to GPU one-by-one, stopping when free memory drops below
# preserved_memory_gb. This allows partially loading a 26GB model onto a 6GB GPU --
# whatever fits goes to GPU, the rest stays on CPU and is streamed via DynamicSwapInstaller.
def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(f'Moving {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB')

    for m in model.modules():
        if get_cuda_free_memory_gb(target_device) <= preserved_memory_gb:
            _empty_cache()
            return

        if hasattr(m, 'weight'):
            m.to(device=target_device)

    model.to(device=target_device)
    _empty_cache()
    return


# Reverse of move_model_to_device_with_memory_preservation: moves submodules from GPU
# back to CPU until free GPU memory reaches the preservation threshold. Used to make
# room for another model (e.g., offload transformer to make room for VAE decoder).
def offload_model_from_device_for_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(f'Offloading {model.__class__.__name__} from {target_device} to preserve memory: {preserved_memory_gb} GB')

    for m in model.modules():
        if get_cuda_free_memory_gb(target_device) >= preserved_memory_gb:
            _empty_cache()
            return

        if hasattr(m, 'weight'):
            m.to(device=cpu)

    model.to(device=cpu)
    _empty_cache()
    return


def unload_complete_models(*args):
    for m in gpu_complete_modules + list(args):
        m.to(device=cpu)
        print(f'Unloaded {m.__class__.__name__} as complete.')

    gpu_complete_modules.clear()
    _empty_cache()
    return


def load_model_as_complete(model, target_device, unload=True):
    if unload:
        unload_complete_models()

    model.to(device=target_device)
    print(f'Loaded {model.__class__.__name__} to {target_device} as complete.')

    gpu_complete_modules.append(model)
    return
