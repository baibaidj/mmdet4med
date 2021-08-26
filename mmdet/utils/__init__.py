from .collect_env import collect_env
from .logger import get_root_logger
print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())
__all__ = ['get_root_logger', 'collect_env', 'print_tensor']
