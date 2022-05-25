from deepem.data.dataset.fibsem.hemibrain import hemibrain_v0 as hemibrain
from deepem.data.dataset.fibsem.fib25 import fib25_v0 as fib25
from deepem.data.dataset.fibsem.fib25 import funke_v0 as funke


def load_data(*args, **kwargs):
    d1 = hemibrain.load_data(*args, **kwargs)
    d2 = fib25.load_data(*args, **kwargs)
    d3 = funke.load_data(*args, **kwargs)

    data = dict()
    data.update(d1)
    data.update(d2)
    data.update(d3)
    return data
