import math
import random
from model_representation.ParametrizedLayer.PConv1DLayer import Conv1DKernelSizeParam
from model_representation.ParametrizedLayer.PConv2DLayer import Conv2DKernelSizeParam
from optimizer.SeqEvo.SeqEvoGenome import SeqEvoGenome
import utils.settings as settings
class SeqEvoModelChecker():
    """
    - data remaining in the end?
    - remove random conv layers from model if not!
    """

    @classmethod
    def check_model_genome(cls, seqevo_genome: SeqEvoGenome) -> None:
        cls.fix_convolution_dimension_loss(seqevo_genome)
    
    @classmethod
    def fix_convolution_dimension_loss(cls, seqevo_genome: SeqEvoGenome):
        # TODO: get those magic numbers from data (globally accessible?)
        def problematic_input(timesteps_dimension_size_after_convs, features_dimension_size_after_convs, channel_dimension_size_after_convs) -> bool:
            if channel_dimension_size_after_convs:
                if channel_dimension_size_after_convs <= 0:
                    return True
            if timesteps_dimension_size_after_convs <= 0 or features_dimension_size_after_convs <= 0:
                return True
            return False
        
        timesteps_dimension_size = 90
        features_dimension_size = 51
        timesteps_dimension_size_after_convs, features_dimension_size_after_convs, channel_dimension_size_after_convs = cls.__calculate_dimensions_after(timesteps_dimension_size, features_dimension_size, seqevo_genome.layers)

        while(problematic_input(timesteps_dimension_size_after_convs, features_dimension_size_after_convs, channel_dimension_size_after_convs)):
            cls._change_conv_to_other_layer(seqevo_genome)
            timesteps_dimension_size_after_convs, features_dimension_size_after_convs, channel_dimension_size_after_convs = cls.__calculate_dimensions_after(timesteps_dimension_size, features_dimension_size, seqevo_genome.layers)

    @classmethod
    def __calculate_dimensions_after(cls, timesteps_dimension_size, features_dimension_size, layers) -> 'tuple[int, int, int]':
        timesteps_dimension_size_after_convs = timesteps_dimension_size
        features_dimension_size_after_convs = features_dimension_size
        channel_dimension_size_after_convs = None

        for layer in layers:
            # if dimension loss in the middle of model => return -1, -1 to trigger resolver
            if (timesteps_dimension_size_after_convs <= 0 or features_dimension_size_after_convs <= 0):
                return -1, -1, -1
            if channel_dimension_size_after_convs:
                if channel_dimension_size_after_convs <= 0:
                    return -1, -1, -1
            
            # size of features dimension gets reduced to "units" in LSTM/Dense
            if layer.__class__.__name__ == "PDenseLayer":
                units = None
                for param in layer.params:
                    if param._key == "units":
                        units = param.value
                if channel_dimension_size_after_convs:
                    channel_dimension_size_after_convs = units
                else:
                    features_dimension_size_after_convs = units
                    
            if layer.__class__.__name__ == "PLstmLayer":
                if channel_dimension_size_after_convs:
                    features_dimension_size_after_convs *= channel_dimension_size_after_convs
                    channel_dimension_size_after_convs = None
                units = None
                for param in layer.params:
                    if param._key == "units":
                        units = param.value
                features_dimension_size_after_convs = units

            # size of timesteps dimension gets reduced by Conv1D
            if layer.__class__.__name__ == "PConv1DLayer":
                kernel_size = None
                stride = None
                filters = None
                for param in layer.params:
                    if param._key == "kernel_size":
                        kernel_size = param.value
                    if param._key == "strides":
                        stride = param.value  
                    if param._key == "filters":
                        filters = param.value
                channel_dimension_size_after_convs = filters
                timesteps_dimension_size_after_convs = math.floor((timesteps_dimension_size_after_convs - kernel_size) / stride) + 1

            # size of timesteps and features dimension gets reduced by Conv2D
            elif layer.__class__.__name__ == "PConv2DLayer":
                kernel_size = None
                stride = None
                filters = None
                for param in layer.params:
                    if param._key == "kernel_size":
                        kernel_size = param.value
                    if param._key == "strides":
                        stride = param.value 
                    if param._key == "filters":
                        filters = param.value
                timesteps_kernel_size = kernel_size[0]
                features_kernel_size = kernel_size[1]
                timesteps_stride = stride[0]
                features_stride = stride[1]
                
                timesteps_dimension_size_after_convs = math.floor((timesteps_dimension_size_after_convs - timesteps_kernel_size) / timesteps_stride) + 1
                features_dimension_size_after_convs = math.floor((features_dimension_size_after_convs - features_kernel_size) / features_stride) + 1
                channel_dimension_size_after_convs = filters

        return timesteps_dimension_size_after_convs, features_dimension_size_after_convs, channel_dimension_size_after_convs

    @classmethod
    def __remove_random_conv_layer(cls, seqevo_genome) -> None:

        # get random conv layer index from seqevo_genome
        conv_layer_idx_list = list(filter(lambda idx: (seqevo_genome.layers[idx].__class__.__name__ in ["PConv1DLayer", "PConv2DLayer"]), range(len(seqevo_genome.layers))))
        layer_idx_to_remove = random.choice(conv_layer_idx_list)

        seqevo_genome.layers.pop(layer_idx_to_remove)
        
    @classmethod
    def _change_conv_to_other_layer(cls, seqevo_genome) -> None:
        
        conv_layer_idx_list = list(filter(lambda idx: (seqevo_genome.layers[idx].__class__.__name__ in ["PConv1DLayer", "PConv2DLayer"]), range(len(seqevo_genome.layers))))
        layer_idx_to_remove = random.choice(conv_layer_idx_list)

        layer_class = random.choice([x for x in settings.layer_pool if x.__name__ not in ["PConv1DLayer", "PConv2DLayer"]])
        seqevo_genome.layers[layer_idx_to_remove] = layer_class.create_random_default()
    
    @classmethod
    def _lower_high_conv_params(cls, seqevo_genome) -> None:    
        conv_layer_idx_list = list(filter(lambda idx: (seqevo_genome.layers[idx].__class__.__name__ in ["PConv1DLayer", "PConv2DLayer"]), range(len(seqevo_genome.layers))))
        
        highest_kernel_size = 0
        idx_of_highest_kernel_size_layer = None
        
        for layer_idx in conv_layer_idx_list:
            layer = seqevo_genome.layers[layer_idx]
            kernel_size = None
            stride = None

            for param in layer.params:
                if param._key == "kernel_size":
                    kernel_size = param.value
                if param._key == "strides":
                    stride = param.value  
            if layer.__class__.__name__ == "PConv1DLayer":
                if kernel_size > highest_kernel_size:
                    highest_kernel_size = kernel_size
                    idx_of_highest_kernel_size_layer = layer_idx
                    
            elif layer.__class__.__name__ == "PConv2DLayer":
                max_kernel_size = max(kernel_size)
                if max_kernel_size > highest_kernel_size:
                    highest_kernel_size = max_kernel_size
                    idx_of_highest_kernel_size_layer = layer_idx
        
        # create new kernel size param with lower value
        new_kernel_param = None
        if seqevo_genome.layers[idx_of_highest_kernel_size_layer].__class__.__name__ == "PConv1DLayer":            
            new_kernel_param = Conv1DKernelSizeParam.create(2)
        elif seqevo_genome.layers[idx_of_highest_kernel_size_layer].__class__.__name__ == "PConv2DLayer":
            new_kernel_param = Conv2DKernelSizeParam.create((2,2))
            
        # add new kernel size param to layer, update dependency for stride param
        params = seqevo_genome.layers[idx_of_highest_kernel_size_layer].params
        kernel_size_param_idx = [idx for idx, x in enumerate(params) if x._key == "kernel_size"][0]
        stride_size_param = [x for x in params if x._key == "strides"][0]
        params[kernel_size_param_idx] = new_kernel_param
        stride_size_param._dependent_on_param = new_kernel_param
        
            
            


            

    
    

