import sys


class Context:
    def __init__(self):
        self.device = 'cpu'
        self.print_precision = None
        self.print_identities = False


_CONTEXT = Context()


def get_device() -> str:
    return _CONTEXT.device


def get_print_precision() -> int:
    if not _CONTEXT.print_precision:
        return sys.float_info.dig

    return _CONTEXT.print_precision


def get_print_identities() -> bool:
    return _CONTEXT.print_identities
