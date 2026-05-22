import random, numpy as np, torch


def set_seed(seed: int = 42):
    random.seed(seed)  # Python nativo
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch na CPU
    torch.cuda.manual_seed_all(seed)  # PyTorch na GPU (cuda)
    torch.backends.cudnn.deterministic = True  # força CUDA a ser determinístico
    torch.backends.cudnn.benchmark = False  # desliga otimizações automáticas que quebram determinismo
