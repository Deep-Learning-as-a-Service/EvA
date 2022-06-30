from utils.Visualizer import Visualizer
from model_representation.EvoParam.FloatEvoParam import FloatEvoParam
from model_representation.EvoParam.IntEvoParam import IntEvoParam
from model_representation.ParametrizedLayer.PLstmLayer import LstmDropoutParam, LstmUnitsParam, PLstmLayer



def test_float_mutate_only_distributions():
    class ExampleFloatEvoParam(FloatEvoParam):
        _default_values = [0.5]
        _value_range = [0.0, 1.0]
        _key = "example"
        _mean = 0.5
        _sd = 500

    param_01 = ExampleFloatEvoParam.create(value=0.5)

    points = []
    def add_point(param, it):
        points.append((param.value, it, "red" , 0.05))

    the_param = param_01
    for it, mutation_intensity in enumerate(["low", "mid", "high", "all"]):

        add_point(the_param, it)
        for i in range(500):
            the_param.mutate(intensity=mutation_intensity)
            add_point(the_param, it)

    Visualizer.scatter_plot(points)

def test_int_mutate_only_distributions():
    class ExampleIntEvoParam(IntEvoParam):
        _default_values = [50]
        _value_range = [0, 100]
        _key = "example"
        _mean = 50
        _sd = 99999

    param_01 = ExampleIntEvoParam.create(value=50)

    points = []
    def add_point(param, it):
        points.append((param.value, it, "red" , 0.05))

    the_param = param_01
    for it, mutation_intensity in enumerate(["low", "mid", "high", "all"]):

        add_point(the_param, it)
        for i in range(500):
            the_param.mutate(intensity=mutation_intensity)
            add_point(the_param, it)

    Visualizer.scatter_plot(points)


def test_random_default_influence():
    param_02 = ExampleFloatEvoParam.create_random_default()
    
def lstm_dropout_test():
    lstm_layer = PLstmLayer(params=[LstmUnitsParam.create(64), LstmDropoutParam.create(0.1)])  
    for _ in range(10):
        lstm_layer.mutate(intensity="high")
        print(lstm_layer)

# test_int_mutate_only_distributions()

    
