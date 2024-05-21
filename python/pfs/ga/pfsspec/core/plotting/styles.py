from collections.abc import Iterable

def sanitize_style(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}

def __update_style(args, key, value):
    if isinstance(key, str) or not isinstance(key, Iterable):
        key = [ key ]

    if not any([ k in args for k in key]):
        args[key[0]] = value

def dashed_line(**kwargs):
    args = kwargs.copy()

    __update_style(args, 'marker', 'None')
    __update_style(args, ('linestyle', 'ls'), 'dashed')
    __update_style(args, ('linewidth', 'lw'), 0.5)

    return args

def solid_line(**kwargs):
    args = kwargs.copy()

    __update_style(args, 'marker', 'None')
    __update_style(args, ('linestyle', 'ls'), 'solid')
    __update_style(args, ('linewidth', 'lw'), 0.5)

    return args

def thin_line(**kwargs):
    args = kwargs.copy()

    __update_style(args, ('linewidth', 'lw'), 0.2)

    return args

def extra_thin_line(**kwargs):
    args = kwargs.copy()

    __update_style(args, ('linewidth', 'lw'), 0.1)

    return args

def color_line(color, **kwargs):
    args = kwargs.copy()

    __update_style(args, 'color', color)

    return args

def red_line(**kwargs):
    return color_line('red', **kwargs)

def blue_line(**kwargs):
    return color_line('blue', **kwargs)

def lightgray_line(**kwargs):
    return color_line('lightgray', **kwargs)

def lightblue_line(**kwargs):
    return color_line('lightblue', **kwargs)

def open_circle(**kwargs):
    args = kwargs.copy()

    __update_style(args, 'edgecolor', 'b')
    __update_style(args, 'facecolor', 'None')
    __update_style(args, ('linewidth', 'lw'), 0.2)

    return args

def closed_circle(**kwargs):
    args = kwargs.copy()

    __update_style(args, 'edgecolor', 'None')
    __update_style(args, 'facecolor', 'b')
    __update_style(args, ('linewidth', 'lw'), 0)

    return args

def tiny_dots_plot(**kwargs):
    args = kwargs.copy()

    __update_style(args, 'markeredgecolor', 'None')
    __update_style(args, ('markersize', 'ms'), 0.2)
    __update_style(args, ('color', 'c', 'fmt'), 'k')
    __update_style(args, ('marker', 'fmt'), 's')
    __update_style(args, ('linestyle', 'ls', 'fmt'), 'None')
    __update_style(args, ('linewidth', 'lw'), 0.5)

    return args

def tiny_dots_scatter(**kwargs):
    args = kwargs.copy()

    __update_style(args, 'edgecolors', 'None')
    __update_style(args, 'marker', 's')
    __update_style(args, ('color', 'c'), 'k')
    __update_style(args, ('size', 's'), 0.1)

    return args

def histogram_imshow(**kwargs):
    args = kwargs.copy()

    __update_style(args, 'aspect', 'auto')
    __update_style(args, 'origin', 'lower')

    return args

def axis_label_font(**kwargs):
    args = kwargs.copy()

    __update_style(args, 'fontsize', 8)

    return args

def tick_label_font(**kwargs):
    args = kwargs.copy()

    __update_style(args, 'labelsize', 8)

    return args

def mask_label_font(**kwargs):
    args = kwargs.copy()

    __update_style(args, 'labelsize', 5)
    __update_style(args, 'labelcolor', 'red')

    return args

def plot_title_font(**kwargs):
    args = kwargs.copy()

    __update_style(args, 'fontsize', 8)

    return args