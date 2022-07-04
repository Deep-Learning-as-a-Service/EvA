from model_representation.ModelNode.ModelNode import ModelNode
import keras.models
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Dense,
    Flatten,
    Dropout,
    LSTM,
    GlobalMaxPooling1D,
    MaxPooling2D,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
import utils.settings as settings
import tensorflow as tf

class ModelGenome():
    """
    this class stores a concrete keras model training execution
    - keras layer architecture
    - keras layer hyperparams
    - training params
    """
    default_n_epochs = 5
    default_batch_size = 32 
    default_learning_rate = 0.001

    def __init__(self, input_model_node, n_epochs, batch_size, learning_rate):
        self.input_model_node = input_model_node
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
    
    @classmethod
    def create_with_default_params(cls):
        """
        Subclass responsibility
        """
        raise NotImplementedError
    
    def reset_to_default_params(self):
        self.n_epochs = ModelGenome.default_n_epochs
        self.batch_size = ModelGenome.default_batch_size
        self.learning_rate = ModelGenome.default_learning_rate
    
    def get_input_model_node(self) -> 'ModelNode':
        return self.input_model_node
    
    def _preprocessing_layer(
        self, input_layer: tf.keras.layers.Layer
    ) -> tf.keras.layers.Layer:
        x = tf.keras.layers.Normalization(
            axis=-1,
            variance=settings.input_distribution_variance,
            mean=settings.input_distribution_mean,
        )(input_layer)
        return x
    
    def get_model(self) -> keras.models.Model:
        """
        compile model from current model_genome instance
        """
        
        assert self.input_model_node.architecture_block is not None, "architecture block didn't get initialized yet, call make_compatible() first"
        
        data_dimension_dict = settings.data_dimension_dict
        window_size = data_dimension_dict["window_size"]
        n_features = data_dimension_dict["n_features"]
        n_classes = data_dimension_dict["n_classes"]
        
        # TODO: No extra Node here - instead the layer mapper should assign the Input layer to our input_model_node
        i = Input(shape=(window_size, n_features))

        x = self._preprocessing_layer(i)
        
        # wrap i into list, since architecture_block expects a list of all inputs
        x = self.input_model_node.architecture_block([x])

        current_nodes_with_output = {self.input_model_node: x}
        calculated_nodes = [self.input_model_node] # Queue: in progress, not all childs considered yet
        finished_nodes = [] # finished, all childs considered
        output_func = x # default output_func is x for models of length 1

        # breadth first search
        # TODO: topological sort easier???
        while True:
            for calculated_node in calculated_nodes:
                for child in calculated_node.childs:
                    
                    # child has already been calculated
                    if child in calculated_nodes or child in finished_nodes:
                        continue
                    
                    # retrieve outputs from all parents of current child
                    # TODO: try catch not in control flow: add check for is key in dict
                    try:
                        inputs_for_child = [current_nodes_with_output[parent] for parent in child.parents]
                        
                    # parent hasn't been calculated yet, skip child, since it has another parent that will look at it later
                    except KeyError:
                        continue
                    
                    # add output of child into dict
                    current_nodes_with_output[child] = child.architecture_block(inputs_for_child)
                    
                    # add child to calculated nodes
                    calculated_nodes.append(child)

                    # FINISH
                    # TODO: model_genome.type == 'output'
                    if(len(child.childs) == 0):
                        output_func = current_nodes_with_output[child]
            
            # all childs of current node have been looked at, remove from calculated and add into finished nodes
            calculated_nodes.remove(calculated_node)
            finished_nodes.append(calculated_node)
            
            # break out of while loop if all children of all nodes have been retrieved already
            if(len(calculated_nodes) == 0):
                break
            
        # add flatten and softmax as last layers to map onto outputs, 
        # TODO: This should be integrated into the last layer of Genome instead of adding another layer
        output_func = Flatten()(output_func)
        out = Dense(n_classes, activation='softmax')(output_func)
        model = Model(i, out)
        
        # TODO: parametrize thatttttttttttttttttttttt
        model.compile(
            optimizer=RMSprop(learning_rate=self.learning_rate), # Adam
            loss="categorical_crossentropy",  # CategoricalCrossentropy (than we have to to the one hot encoding - to_categorical), before: "sparse_categorical_crossentropy"
            metrics=["accuracy"],
        )
        return model

            
            