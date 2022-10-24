def safe_deep_compare(a, b, ignore=None):
    """
    Deep-compare two dictionaries. Ignore keys in the
    `ignore` list (independently of depth)
    """

    if not isinstance(a, dict) or not isinstance(b, dict):
        return False

    kk = list(a.keys()) + list(b.keys())

    for k in kk:
        if ignore is not None and k in ignore:
            continue

        if k not in a or k not in b:
            return False

        if isinstance(a[k], dict) or isinstance(b[k], dict):
            if not safe_deep_compare(a[k], b[k], ignore):
                return False
        
        if a[k] != b[k]:
            return False

    return True