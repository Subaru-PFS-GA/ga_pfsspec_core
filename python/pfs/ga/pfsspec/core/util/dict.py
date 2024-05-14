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

def pivot_dict_of_arrays(d):
    """
    Pivot a dictionary of arrays into an array of dictionaries
    """

    keys = list(d.keys())
    n = len(d[keys[0]])
    res = [{} for i in range(n)]

    for k in keys:
        for i in range(n):
            res[i][k] = d[k][i]

    return res

def pivot_array_of_dicts(a):
    """
    Pivot an array of dictionaries into a dictionary of arrays
    """

    keys = a[0].keys()
    res = { k: [] for k in keys }

    for d in a:
        for k in keys:
            res[k].append(d[k])

    return res