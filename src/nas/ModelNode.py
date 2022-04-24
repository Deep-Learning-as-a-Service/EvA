import utils.nas_settings as nas_settings
from nas.ParametrizedLayer import ParametrizedLayer 
from keras.layers import concatenate

class ModelNode():
    def __init__(self, neat_node_key, neat_connections, parent, nodes_already_created):
        nodes_already_created.append(self)

        # TODO: at this point the layer_mapper has not complete info about the neighbours - only random layer_mapper possible - will lead to semantic senseless architectures - # save time
        self.layer = nas_settings.layer_mapper.get_layer(neat_node_key)
        self.neat_node_key = neat_node_key

        self.architecture_block = None
        self.parents = [parent]

        # Childs
        connection_tuples = list(map(lambda neat_connection: neat_connection.key, neat_connections))
        output_connection_tuples = list(filter(lambda connection_tuple: connection_tuple[0] == neat_node_key, connection_tuples))
        child_neat_keys = list(map(lambda output_connection_tuple: output_connection_tuple[1], output_connection_tuples))
        keys_already_created = list(map(lambda node: node.neat_node_key, nodes_already_created))
        self.childs = []
        for child_neat_key in child_neat_keys:
            if child_neat_key not in keys_already_created:
                child_node = ModelNode(
                        neat_node_key=child_neat_key, 
                        neat_connections=neat_connections,
                        parent=self, 
                        nodes_already_created=nodes_already_created
                    )
                nodes_already_created.append(child_node)
                self.childs.append(child_node)
            else:
                for node in nodes_already_created:
                    if node.neat_node_key == child_neat_key:
                        node.add_parent(self)
                        self.childs.append(node)
                        break
    
    def add_parent(self, parent):
        self.parents.append(parent)

    def make_compatible(self) -> None:
        """
        Recursive applied to the whole DAG
        dependent on input and output nodes wraps the keras layer function in a function that concatenates/reshapes the data
        self.architecture_block = lambda input_1, input_2: keras.layer.Dense(concatenate([input_1, input_2])
        """
        parametrized_layer: ParametrizedLayer = nas_settings.layer_mapper.get_layer(self.neat_node_key)
        self.layer = parametrized_layer.get_func()
        
        # concatenate inputs if node has multiple inputs
        concatenate_func = lambda input: input[0]
        if(len(self.parents) > 1):
            concatenate_func = lambda input: concatenate(input)
        
        self.architecture_block = lambda input: self.layer(concatenate_func(input))
        
        for child in self.childs:
            child.make_compatible()