import numbers
from types import SimpleNamespace

def safe_deep_copy(orig, **kwargs):
    if isinstance(orig, list):
        return [ safe_deep_copy(i, **kwargs) for i in orig ]
    elif isinstance(orig, tuple):
        return tuple([ safe_deep_copy(i, **kwargs) for i in orig ])
    elif isinstance(orig, dict):
        return { k: safe_deep_copy(i, **kwargs) for k, i in orig.items() }
    elif isinstance(orig, SimpleNamespace):
        # We can't handle derived classes here with overriden __init__
        return SimpleNamespace(**{ k: v for k, v in orig.__dict__.items() })
    elif isinstance(orig, numbers.Number):
        return orig
    elif isinstance(orig, str):
        return orig
    elif orig is not None and hasattr(orig, 'copy'):
        return orig.copy(**kwargs)
    elif orig is None:
        return None
    else:
        raise ValueError(f"Cannot make copy of object of type {type(orig).__name__}.")
    