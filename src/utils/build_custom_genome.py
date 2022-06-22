from model_representation.EvoParam.CategDEvoParam import CategDEvoParam
from model_representation.EvoParam.IntEvoParam import IntEvoParam
from model_representation.EvoParam.TupleCategDEvoParam import TupleCategDEvoParam
from model_representation.EvoParam.TupleIntEvoParam import TupleIntEvoParam
from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
from model_representation.ParametrizedLayer.PConv2DLayer import PConv2DLayer, Conv2DFiltersParam, Conv2DStridesParam, Conv2DKernelSizeParam
from model_representation.ParametrizedLayer.PLstmLayer import PLstmLayer, LstmUnitsParam, LstmDropoutParam
from model_representation.ParametrizedLayer.PDenseLayer import PDenseLayer, DenseUnitsParam
from optimizer.SeqEvo.SeqEvoGenome import SeqEvoGenome



def get_problematic_genome():
        
    first = PDenseLayer(params=[DenseUnitsParam.create(109)])
    second = PLstmLayer(params=[
        LstmUnitsParam.create(13),
        LstmDropoutParam.create(0.1)
    ]
    )
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
    # fifth = PDenseLayer(params=[DenseUnitsParam.create(104)]) 
    # sixth = PLstmLayer(params=[LstmUnitsParam.create(30)])
    # seventh = PConv2DLayer(
    #     params=[
    #         Conv2DFiltersParam.create(128),
    #         Conv2DKernelSizeParam.create((5,5)),
    #         Conv2DStridesParam.create(("100%","step1"), dependent_on_param=Conv2DKernelSizeParam.create((5,5)))
    #     ]
    # ) 
    
    layers = [
        first, second, third, fourth
    ]
    # return SeqEvoGenome(layers=layers)
    return layers
    