import wrapt
from line_profiler import LineProfiler
lp = LineProfiler()

def lp_wrapper():
    """
    Shows time consumed in each line
    """
    @wrapt.decorator
    def wrapper(func, instance, args, kwargs):
        global lp
        lp_wrapper = lp(func)
        res = lp_wrapper(*args, **kwargs)
        lp.print_stats()
        return res

    return wrapper
