################################################################################
# this homework explores the functionality of functional programming in Python #
################################################################################

def sequential_map(*args):
    result = args[-1]
    for i in args[:-1]:
        result = list(map(i, result))
    return result


def consensus_filter(*args):
    result = args[-1]
    for i in args[:-1]:
        result = list(filter(i, result))
    return result


def conditional_reduce(func1, func2, container):
    pre_result = list(filter(func1, container))
    result = func2(pre_result[0], pre_result[1])
    for i in range(2, len(pre_result)):
        result = func2(result, pre_result[2])
    return result


# a way for normal people:
# from functools import reduce
# reduce(lambda a,l: lambda x: l(a(x)), #args)

# abnormality
def func_chain(*args):
    result = args[0]
    function = lambda a, l: lambda x: l(a(x))
    for x in args[1:]:
        result = function(result, x)
    return result


def multiple_partial(*args, **kwargs):
    result = [lambda x: i(*x, **kwargs) for i in args]
    return result


import sys


def printer(*args, sep=' ', end='', file=False):
    if not file:
        for i in args[:-1]:
            sys.stdout.write(i)
            sys.stdout.write(sep)
        sys.stdout.write(args[-1])
        sys.stdout.write(end)
    else:
        write_it_down = ''.join([i + sep for i in args[:-1]]) + args[-1] + end
        with open(file, 'w') as file_to_write:
            file_to_write.write(write_it_down)
