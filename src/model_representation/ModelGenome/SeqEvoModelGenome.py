from model_representation.ModelGenome.ModelGenome import ModelGenome
from model_representation.ModelNode.SeqEvoModelNode import SeqEvoModelNode

class SeqEvoModelGenome(ModelGenome):
    """
    Model Genome, that also stores the connection to its corresponding NeatGenome
    """
    def __init__(self, input_model_node, seqevo_genome, n_epochs, batch_size, learning_rate):
        super().__init__(
            input_model_node=input_model_node, 
            n_epochs=n_epochs, 
            batch_size=batch_size, 
            learning_rate=learning_rate
        )
        self.seqevo_genome = seqevo_genome
    
    @classmethod
    def create_with_default_params(cls, seqevo_genome) -> 'ModelGenome':

        input_model_node=SeqEvoModelNode.create_net(
                parent=None,
                seqevo_genome=seqevo_genome, 
                node_idx=0
            )
        input_model_node.make_compatible()
    

        return cls(
            input_model_node=input_model_node, 
            seqevo_genome=seqevo_genome, 
            n_epochs=ModelGenome.default_n_epochs, 
            batch_size=ModelGenome.default_batch_size, 
            learning_rate=ModelGenome.default_learning_rate
        )