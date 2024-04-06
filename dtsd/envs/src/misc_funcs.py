import inspect
from collections import deque
def weighted_quadratic(x, w):
    # inputs should be numpy arrays
    return x.T.dot(w).dot(x) 

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

# online functions
exists_in = lambda gkey,gdict: gkey in gdict.keys()

exists_not_none = lambda gkey,gdict: gkey in gdict.keys() and  not(gdict[gkey] is None)

exists_and_true = lambda gkey,gdict: gkey in gdict.keys() and  gdict[gkey] == True

exists_and_is_equal = lambda gkey,gdict,val: gkey in gdict.keys() and  gdict[gkey] == val

class moving_average_filter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.data_queue = deque(maxlen=window_size)
        self.current_sum = 0

    def update(self, new_data):
        if len(self.data_queue) == self.window_size:
            self.current_sum -= self.data_queue[0]
        self.data_queue.append(new_data)
        self.current_sum += new_data

        return self.current_sum / len(self.data_queue)

class none_filter:
    def __init__(self, window_size):
        pass

    def update(self, new_data):
        return new_data