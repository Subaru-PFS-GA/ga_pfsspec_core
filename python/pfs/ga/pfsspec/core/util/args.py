def get_arg(name, old_value, args):
    """
    Reads a command-line argument if specified, otherwise uses a default value.

    :param name: Name of the argument.
    :param old_value: Default value of the argument in case in doesn't exist in args.
    :param args: Argument dictionary.
    :return: The parsed command-line argument.
    """
    if name in args and args[name] is not None:
        return args[name]
    else:
        return old_value

def is_arg(name, args):
    """
    Checks if an (optional) command-line argument is specified.

    :param name: Name of the argument.
    :param args: Argument dictionary.
    :return: True if the optional command-line argument is specified.
    """
    return name in args and args[name] is not None