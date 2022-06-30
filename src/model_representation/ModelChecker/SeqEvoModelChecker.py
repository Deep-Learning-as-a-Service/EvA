import math
import random
from model_representation.ModelGenome.SeqEvoModelGenome import SeqEvoModelGenome
from model_representation.ParametrizedLayer.PConv1DLayer import Conv1DKernelSizeParam
from model_representation.ParametrizedLayer.PConv2DLayer import Conv2DKernelSizeParam
from model_representation.ParametrizedLayer.PDenseLayer import DenseUnitsParam
from model_representation.ParametrizedLayer.PLstmLayer import LstmUnitsParam
from optimizer.SeqEvo.SeqEvoGenome import SeqEvoGenome
import utils.settings as settings
class SeqEvoModelChecker():
    """
    - data remaining in the end?
    - remove random conv layers from model if not!
    """

    @classmethod
    def check_model_genome(cls, seqevo_genome: SeqEvoGenome) -> None:
        cls.fix_convolution_dimension_loss(seqevo_genome) # make model compilable 
        cls.fix_memory_overflow(seqevo_genome) # make model compilable 
        cls.alter_models_with_high_params(seqevo_genome) # reduce time expenditure
        
    @classmethod
    def alter_models_with_high_params(cls, seqevo_genome: SeqEvoGenome):
        threshold = 30_000_000 # 10 mil params as threshold, subject of change
        
        while True:
            model_genome = SeqEvoModelGenome.create_with_default_params(seqevo_genome)
            keras_model = model_genome.get_model()
            num_params = keras_model.count_params()
            if num_params <= threshold:
                break
            else:
                cls.alter_high_param_layers(seqevo_genome)
    
    @classmethod
    def fix_memory_overflow(cls, seqevo_genome: SeqEvoGenome):
        # TODO: get batch size from settings as well
        batch_size = 32
        threshold = pow(2, 30) # if number of neurons in any layer is higher than threshold, reduce big layers
        
        # reduce layer params while GPU would be OOM when compiling
        while True:
            timesteps_dimension_size_after_convs = settings.data_dimension_dict["window_size"]
            features_dimension_size_after_convs = settings.data_dimension_dict["n_features"]
            channel_dimension_size_after_convs = 1
            too_large = False
            too_large_idx = None
            # calculate number of neurons used by each layer
            for idx, layer in enumerate(seqevo_genome.layers):
                timesteps_dimension_size_after_convs, features_dimension_size_after_convs, channel_dimension_size_after_convs = cls.__calculate_dimensions_after(timesteps_dimension_size_after_convs, features_dimension_size_after_convs, [layer], channel_dimension_size_after_convs)
                
                # channel size can be none, set to 1 for calculation afterwards
                if not channel_dimension_size_after_convs:
                    channel_dimension_size_after_convs = 1
                total_neurons = timesteps_dimension_size_after_convs * features_dimension_size_after_convs * channel_dimension_size_after_convs * batch_size
                
                # if total neurons is too large => break and reduce layer params
                if total_neurons >= threshold:
                    too_large = True
                    too_large_idx = idx
                    break
                
            if not too_large:
                break 
            
            cls.alter_high_param_layers_before_idx(seqevo_genome, too_large_idx)
                
    @classmethod
    def alter_high_param_layers_before_idx(cls, seqevo_genome: SeqEvoGenome, idx: int):
        """
        try "alter_high_param_layers" before given idx, if there is no dense/lstm before given idx => remove random layer before given idx
        """     
        high_param_layer_idx_list = list(filter(lambda idx: (seqevo_genome.layers[idx].__class__.__name__ in ["PDenseLayer", "PLstmLayer"]), range(idx)))
        if len(high_param_layer_idx_list) == 0:
            seqevo_genome.layers.pop(random.randrange(idx+1))
        else:
            cls.alter_high_param_layers(seqevo_genome, before_idx=idx+1)

    @classmethod
    def alter_high_param_layers(cls, seqevo_genome: SeqEvoGenome, before_idx=None):
        """
        find Dense/LSTM layer with highest units number and change to a fifth of original units or minimum of value range  
        """
        max_idx_to_alter = len(seqevo_genome.layers)
        if before_idx:
            max_idx_to_alter = before_idx
        high_param_layer_idx_list = list(filter(lambda idx: (seqevo_genome.layers[idx].__class__.__name__ in ["PDenseLayer", "PLstmLayer"]), range(max_idx_to_alter)))
        highest_n_units = 0
        highest_n_units_idx = 0
        
        # retrieve layer with highest units param
        for layer_idx in high_param_layer_idx_list:
            layer = seqevo_genome.layers[layer_idx]
            
            units = None
            for param in layer.params:
                if param._key == "units":
                    units = param.value
            if units > highest_n_units:
                highest_n_units = units
                highest_n_units_idx = layer_idx
                
        layer_to_change = seqevo_genome.layers[highest_n_units_idx]
        
        new_units_param = None
        if layer_to_change.__class__.__name__ == "PLstmLayer":
            new_units_param = LstmUnitsParam.create(max(round(highest_n_units / 5), LstmUnitsParam._value_range[0]))
        elif layer_to_change.__class__.__name__ == "PDenseLayer":
            new_units_param = DenseUnitsParam.create(max(round(highest_n_units / 5), DenseUnitsParam._value_range[0]))
        
        # add new kernel size param to layer, update dependency for stride param
        
        params = layer_to_change.params
        units_param_idx = [idx for idx, x in enumerate(params) if x._key == "units"][0]
        params[units_param_idx] = new_units_param


    
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
        
        timesteps_dimension_size = settings.data_dimension_dict["window_size"]
        features_dimension_size = settings.data_dimension_dict["n_features"]
        timesteps_dimension_size_after_convs, features_dimension_size_after_convs, channel_dimension_size_after_convs = cls.__calculate_dimensions_after(timesteps_dimension_size, features_dimension_size, seqevo_genome.layers)

        while(problematic_input(timesteps_dimension_size_after_convs, features_dimension_size_after_convs, channel_dimension_size_after_convs)):
            cls._change_conv_to_other_layer(seqevo_genome)
            timesteps_dimension_size_after_convs, features_dimension_size_after_convs, channel_dimension_size_after_convs = cls.__calculate_dimensions_after(timesteps_dimension_size, features_dimension_size, seqevo_genome.layers)

    @classmethod
    def __calculate_dimensions_after(cls, timesteps_dimension_size, features_dimension_size, layers, channel_dimension_size=None) -> 'tuple[int, int, int]':
        timesteps_dimension_size_after_convs = timesteps_dimension_size
        features_dimension_size_after_convs = features_dimension_size
        channel_dimension_size_after_convs = channel_dimension_size

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
                max_pooling = None
                for param in layer.params:
                    if param._key == "kernel_size":
                        kernel_size = param.value
                    if param._key == "strides":
                        stride = param.value 
                    if param._key == "filters":
                        filters = param.value
                    if param._key == "max_pooling":
                        max_pooling = 2 if param.value == (2,2) else 4 if param.value == (4,4) else 1 
                timesteps_kernel_size = kernel_size[0]
                features_kernel_size = kernel_size[1]
                timesteps_stride = stride[0]
                features_stride = stride[1]
                
                
                timesteps_dimension_size_after_convs = math.floor((math.floor((timesteps_dimension_size_after_convs - timesteps_kernel_size) / timesteps_stride) + 1) / max_pooling)
                features_dimension_size_after_convs = math.floor((math.floor((features_dimension_size_after_convs - features_kernel_size) / features_stride) + 1) / max_pooling)
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

            for param in layer.params:
                if param._key == "kernel_size":
                    kernel_size = param.value
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
