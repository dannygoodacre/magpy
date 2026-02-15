_PROVIDERS: dict[str, type] = {}


def register(cls: type):
    _PROVIDERS[cls.__name__] = cls

    return cls


def is_type(obj: object, type_name: str) -> bool:
    target_cls = _PROVIDERS.get(type_name)

    return isinstance(obj, target_cls) if target_cls else False
