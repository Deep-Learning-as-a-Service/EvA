from ModelGenome import ModelGenome

class NeatNASModelGenome(ModelGenome):
    """
    Model Genome, that also stores the connection to its corresponding NeatGenome
    """
    def __init__(self, input_model_node, neat_genome, n_epochs, batch_size, learning_rate):
        super().__init__(
            input_model_node=input_model_node, 
            n_epochs=n_epochs, 
            batch_size=batch_size, 
            learning_rate=learning_rate
        )
        self.neat_genome = neat_genome
    
    @classmethod
    def create_with_default_params(cls, neat_genome) -> 'ModelGenome':
        """
        create a model genome from a neat genome
        with standard trainings params
            - n_epochs=5, batch_size=32, learning_rate=0.001
        
        - we have one input neuron always (config)
        - in the neat implementation this node has the key -1
        """
        nodes_already_created = []
        input_model_node = NeatNASModelNode.create_net(
            neat_node_key=-1, 
            neat_connections=list(neat_genome.connections.values()), 
            parent=None, 
            nodes_already_created=nodes_already_created
        )

        # TODO: input_model_node.assign_layer() - take this from make_compatible
        input_model_node.make_compatible()
        return cls(
            input_model_node=input_model_node, 
            neat_genome=neat_genome, 
            n_epochs=Model_Genome.default_n_epochs, 
            batch_size=Model_Genome.default_batch_size, 
            learning_rate=Model_Genome.default_learning_rate
        )