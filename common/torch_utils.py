import platform
import torch


def get_available_device(verbose=True):
    exec_device = torch.device('cpu')
    if torch.has_mps:
        exec_device = torch.device('mps')
    if torch.has_cuda:
        exec_device = torch.device('cuda')
    if verbose:
        print(f'Platform: {platform.system()}')
        print(f'Release: {platform.release()}')
        print(f'MPS available: {torch.has_mps}')
        print(f'CUDA available: {torch.has_cuda}')
        print(f'Selected device: {exec_device}')
    return exec_device


def tensor2numpy(tensor):
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.numpy()
