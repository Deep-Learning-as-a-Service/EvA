from model_representation.ModelNode.ModelNode import ModelNode

class SeqEvoModelNode(ModelNode):

    def add_child(self, child):
        if child is not None:
            self.childs.append(child)
    
    @classmethod
    def create_net(cls, parent, seqevo_genome, node_idx):
        if node_idx == len(seqevo_genome.layers):
            return None

        model_node = cls(
            layer=seqevo_genome.layers[node_idx],
            parents= [parent] if parent is not None else [],
            childs=[]
        )
        model_node.add_child(cls.create_net(
                parent=model_node, 
                seqevo_genome=seqevo_genome, 
                node_idx=node_idx+1
            )
        )
        return model_node