class BatchSizeParam(IntEvoParam):
    _default_values = [16, 32, 64]
    _value_range = [16, 128]
    _key = "batch_size"
    _mean = 32
    _sd = 64

class NEpochsParam(IntEvoParam):
    _default_values = [5, 10, 20]
    _value_range = [5, 30]
    _key = "epochs"
    _mean = 10
    _sd = 20


class HyPaEvoGenome():
    def __init__(self, batch_size, n_epochs):
        # to be continued...