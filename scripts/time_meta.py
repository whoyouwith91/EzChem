import time

import torch

'''
This global dictionary will record run time infos while running, which may help fast the code(find the bottle neck)
'''
function_and_time = {}
function_and_count = {}


def record_data(name, t0, syn=False):
    if syn:
        torch.cuda.synchronize()
    delta_t = time.time() - t0
    if name in function_and_time.keys():
        function_and_time[name] += delta_t
        function_and_count[name] += 1
    else:
        function_and_time[name] = delta_t
        function_and_count[name] = 1
    return time.time()


def print_function_runtime(logger=None):
    _sum = 0.
    for key in function_and_time.keys():
        _sum = _sum + function_and_time[key]
    for key in function_and_time.keys():
        if logger is None:
            print('{}: {:.3e}s({}%)---count: {}'.format(key, function_and_time[key], 100*function_and_time[key]/_sum,
                                                       function_and_count[key]))
        else:
            logger.info('{}: {:.3e}s({}%)---count: {}'.format(key, function_and_time[key],
                                                             100*function_and_time[key]/_sum,
                                                             function_and_count[key]))

    return function_and_time
