from model_representation.EvoParam.CategDEvoParam import CategDEvoParam
from model_representation.EvoParam.IntEvoParam import IntEvoParam
from model_representation.EvoParam.TupleCategDEvoParam import TupleCategDEvoParam
from model_representation.EvoParam.TupleIntEvoParam import TupleIntEvoParam
from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
from model_representation.ParametrizedLayer.PConv1DLayer import PConv1DLayer
from model_representation.ParametrizedLayer.PConv2DLayer import PConv2DLayer
from model_representation.ParametrizedLayer.PLstmLayer import PLstmLayer
from model_representation.ParametrizedLayer.PDenseLayer import PDenseLayer
from optimizer.SeqEvo.SeqEvoGenome import SeqEvoGenome



def get_problematic_genome():
    class Conv1DFiltersParam(IntEvoParam):
        _default_values = [32, 64, 128]
        _value_range = [16, 128]
        _key = "filters"

    class Conv1DKernelSizeParam(IntEvoParam):
        _default_values = [2, 6, 12]
        _value_range = [2, 24]
        _key = "kernel_size"

    class Conv1DStridesParam(CategDEvoParam):
        """
        TODO optional:
        - mutate should go 1 label to the right/ to the left, instead of coinflip on everything
        """

        _dependent_class = Conv1DKernelSizeParam
        _default_values = ["step1", "50%", "100%"]
        _value_range = ["step1", "25%", "50%", "75%", "100%"]
        _key = "strides"

        @property
        def value(self):
            if self._dependent_value == "step1":
                return 1
            else:
                percentage = int(self._dependent_value[:-1]) / 100 # "25%" -> 0.25
                # Conv2DKernelSizeParam is also a tuple
                return max(1, round(self._dependent_on_param.value * percentage))
            
    class Conv2DFiltersParam(IntEvoParam):
        _default_values = [32, 64, 128]
        _value_range = [16, 128]
        _key = "filters"

    # TODO: smart data specific choice?
    # TODO: global dict? should not be bigger, than window size!
    # - responsibility for correct params responsibility here or Model Checker
    class Conv2DKernelSizeParam(TupleIntEvoParam):
        _default_values = [(1, 1), (3, 3), (5, 5)]
        _value_range = [(1, 1), (7, 7)]
        _key = "kernel_size"

    class Conv2DStridesParam(TupleCategDEvoParam):
        """
        TODO
        - mutate should go 1 label to the right/ to the left, instead of coinflip on everything
        """

        _dependent_class = Conv2DKernelSizeParam
        _default_values = [("step1", "step1"), ("100%", "step1")]
        _value_range = ["step1", "25%", "50%", "75%", "100%"]
        _key = "strides"

        def _value_tup_pos(self, tuple_position):
            dependent_value_tup_pos = self._dependent_value[tuple_position]
            if dependent_value_tup_pos == "step1":
                return 1
            else:
                percentage = int(dependent_value_tup_pos[:-1]) / 100 # "25%" -> 0.25
                # Conv2DKernelSizeParam is also a tuple
                return max(1, round(self._dependent_on_param.value[tuple_position] * percentage))

        @property
        def value(self):
            return (self._value_tup_pos(tuple_position=0), self._value_tup_pos(tuple_position=1))
        
    class DenseUnitsParam(IntEvoParam):
        _default_values = [32, 64, 128]
        _value_range = [16, 128]
        _key = "units"
    
    
    class LstmUnitsParam(IntEvoParam):
        _default_values = [4, 8, 16]
        _value_range = [2, 32]
        _key = "units"
        
    first = PDenseLayer(params=[DenseUnitsParam.create(109)])
    second = PLstmLayer(params=[LstmUnitsParam.create(13)])
    third = PConv2DLayer(
        params=[
            Conv2DFiltersParam.create(88),
            Conv2DKernelSizeParam.create((7,6)),
            Conv2DStridesParam.create(("25%","100%"), dependent_on_param=Conv2DKernelSizeParam.create((7,6)))
        ]
    )    
    fourth = PConv2DLayer(
        params=[
            Conv2DFiltersParam.create(103),
            Conv2DKernelSizeParam.create((6,4)),
            Conv2DStridesParam.create(("50%","100%"), dependent_on_param=Conv2DKernelSizeParam.create((6,4)))
        ]
    )    
    fifth = PDenseLayer(params=[DenseUnitsParam.create(104)]) 
    sixth = PLstmLayer(params=[LstmUnitsParam.create(30)])
    seventh = PConv2DLayer(
        params=[
            Conv2DFiltersParam.create(128),
            Conv2DKernelSizeParam.create((5,5)),
            Conv2DStridesParam.create(("100%","step1"), dependent_on_param=Conv2DKernelSizeParam.create((5,5)))
        ]
    ) 
    
    layers: ParametrizedLayer = [
        first, second, third, fourth, fifth, sixth, seventh
    ]
    return SeqEvoGenome(layers=layers)
    