import utils.nas_settings as nas_settings
from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer 
from keras.layers import concatenate

class ModelNode():
    def __init__(self, layer, parents, childs):
        self.layer = layer
        self.parents = parents
        self.childs = childs
        self.architecture_block = None
    
    @classmethod
    def create_net(cls):
        """
        subclass responsibility
        """
        raise NotImplementedError

    def make_compatible(self) -> None:
        """
        Recursive applied to the whole DAG
        dependent on input and output nodes wraps the keras layer function in a function that concatenates/reshapes the data
        self.architecture_block = lambda input_func_list: keras.layer.Dense(concatenate(input_func_list))
        """
        
        # concatenate func is identity, if there is nothing to concatenate
        concatenate_func = lambda input_func_list: input_func_list[0]

        # concatenate func get a keras.Layer.concatenate if there are multiple input_funcs
        if len(self.parents) > 1:
            concatenate_func = lambda input_func_list: concatenate(input_func_list)
        
        self.architecture_block = lambda input_func_list: self.layer.get_func()(concatenate_func(input_func_list))
        
        for child in self.childs:
            child.make_compatible()