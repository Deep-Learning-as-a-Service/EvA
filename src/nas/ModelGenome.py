from nas.ModelNode import ModelNode


class ModelGenome():
    """
    this class stores a concrete keras model training execution
    - keras layer architecture
    - keras layer hyperparams
    - training params
    """
    def __init__(self, input_model_node, neat_genome, n_epochs, batch_size, learning_rate):
        self.input_model_node = input_model_node
        self.neat_genome = neat_genome
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
    
    @staticmethod
    def create_with_default_params(neat_genome) -> 'ModelGenome':
        """
        create a model genome from a neat genome
        with standard trainings params
            - n_epochs=5, batch_size=32, learning_rate=0.001
        
        - we have one input neuron always (config)
        - in the neat implementation this node has the key -1
        """
        nodes_already_created = []
        input_model_node = ModelNode(
            neat_node_key=-1, 
            neat_connections=list(neat_genome.connections.values()), 
            parent=None, 
            nodes_already_created=nodes_already_created
        ) # TODO create_from_connections recursive
        input_model_node.make_compatible()
        return ModelGenome(input_model_node=input_model_node, neat_genome=neat_genome, n_epochs=5, batch_size=32, learning_rate=0.001)
    