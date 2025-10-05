import sys

class Context:
    def __init__(self):
        self.device = 'cpu'
        self.print_precision = None

_CONTEXT = Context()

def get_device() -> str:
    return _CONTEXT.device

def get_print_precision() -> int:
    if not _CONTEXT.print_precision:
        return sys.float_info.dig

    return _CONTEXT.print_precision
