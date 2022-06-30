from optimizer.SeqEvo.SeqEvoGenome import SeqEvoGenome
import utils.settings as settings
from model_representation.ParametrizedLayer.PConv2DLayer import PConv2DLayer
from model_representation.ParametrizedLayer.PDenseLayer import PDenseLayer
from model_representation.ParametrizedLayer.PLstmLayer import PLstmLayer
from model_representation.ModelGenome.SeqEvoModelGenome import SeqEvoModelGenome
from model_representation.ModelChecker.SeqEvoModelChecker import SeqEvoModelChecker


# Config --------------------------------------------------------------------------
window_size = 30*3
n_features = 51
n_classes = 6
num_folds = 5
validation_iterations = 3

layer_pool: 'list[ParametrizedLayer]' = [PConv2DLayer, PDenseLayer, PLstmLayer] #PConv1DLayer
data_dimension_dict = {
    "window_size": window_size,
    "n_features": n_features,
    "n_classes": n_classes
}
settings.init(_layer_pool=layer_pool, _data_dimension_dict=data_dimension_dict)

for _ in range(2):
    seqevo_genome = SeqEvoGenome.create_random_default()
    print("before", seqevo_genome)
    SeqEvoModelChecker.check_model_genome(seqevo_genome)
    print("after", seqevo_genome)
    model_genome = SeqEvoModelGenome.create_with_default_params(seqevo_genome)
    model = model_genome.get_model()
    model.summary()
    


