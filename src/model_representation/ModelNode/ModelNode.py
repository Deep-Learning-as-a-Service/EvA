from model_representation.ParametrizedLayer.PConv1DLayer import PConv1DLayer
from model_representation.ParametrizedLayer.PLstmLayer import PLstmLayer
import utils.nas_settings as nas_settings
from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer 
from keras.layers import concatenate, Reshape

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
        def remove_dimensionality_func(input_func):
            output_func = input_func
            if len(input_func.shape) > 3: # if more dimensions than just (batch_size, timesteps, features)
                output_func = Reshape(target_shape=(input_func.shape[1], input_func.shape[2] * input_func.shape[3]))
            return output_func

        
        # concatenate func is identity, if there is nothing to concatenate
        concatenate_func = lambda input_func_list: input_func_list[0]

        # concatenate func get a keras.Layer.concatenate if there are multiple input_funcs
        if len(self.parents) > 1:
            concatenate_func = lambda input_func_list: concatenate(input_func_list)

        # currently only PLstmLayers need a fixed inputdimension of 2 + batch_size
        # add additional conditionals if other layers also have that syntactical/semantical restriction
        if(self.layer.__class__.__name__ == "PLstmLayer"):
            remove_dimensionality_function = remove_dimensionality_func
        else:
            remove_dimensionality_function = lambda input_func: input_func
            
        self.architecture_block = lambda input_func_list: self.layer.get_func()(remove_dimensionality_function(concatenate_func(input_func_list)))
        
        for child in self.childs:
            child.make_compatible()