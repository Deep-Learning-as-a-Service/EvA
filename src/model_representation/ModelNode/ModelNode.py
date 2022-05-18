from model_representation.ParametrizedLayer.PConv1DLayer import PConv1DLayer
from model_representation.ParametrizedLayer.PConv2DLayer import PConv2DLayer

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

        every node says to his childs -> set your architecture_block
        layer.get_func() will return a function that takes a tensor and returns a tensor

        keras if!
        every layer only cares about the input shape

        TODO: refactor
        """
        is_lstm_layer = type(self.layer) is PLstmLayer
        def remove_dimensionality_func(input_tensor):
            if is_lstm_layer and (len(input_tensor.shape) > 3): # if more dimensions than just (batch_size, timesteps, features)
                return Reshape(target_shape=(input_tensor.shape[1], input_tensor.shape[2] * input_tensor.shape[3]))(input_tensor)
            else:
                return input_tensor
        
        is_conv_layer = type(self.layer) in [PConv1DLayer, PConv2DLayer]
        def add_dimensionality_func(input_tensor):
           if is_conv_layer and len(input_tensor.shape) == 3:
               return Reshape(target_shape=(input_tensor.shape[1], input_tensor.shape[2], 1))(input_tensor)
           return input_tensor
        
        def concatenate_func(input_tensor_list):
            if len(self.parents) > 1:
                return concatenate(input_tensor_list)
            return input_tensor_list[0]

        a_block = lambda tensor_list: concatenate_func(tensor_list) # a_block func that takes tensor_list, returns tensor
        b_block = lambda tensor_list: remove_dimensionality_func(a_block(tensor_list))
        c_block = lambda tensor_list: add_dimensionality_func(b_block(tensor_list))
        d_block = lambda tensor_list: self.layer.get_func()(c_block(tensor_list))

        self.architecture_block = d_block
                
        for child in self.childs:
            child.make_compatible()